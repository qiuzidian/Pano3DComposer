"""Image-to-3D object generation (Amodal3R by default, TRELLIS optionally).

The generator turns the perspective crop(s) of an object into a 3D asset in its
own local frame. Besides the Gaussians and the textured mesh, it renders four
reference snapshots of the asset with known local cameras, which are the
"local-frame" inputs of the object-to-world predictor.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..gaussians import Gaussians, build_covariance
from .aligner import PoseInputs

_AMODAL3R = Path(__file__).resolve().parents[2] / "third_party" / "amodal3r"
if str(_AMODAL3R) not in sys.path:
    sys.path.insert(0, str(_AMODAL3R))

# Attention backends of the TRELLIS-style models; set ATTN_BACKEND=flash-attn if flash-attn is installed.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

SNAPSHOT_RESOLUTION = 518
SAMPLER_PARAMS = {
    "sparse_structure_sampler_params": {"steps": 12, "cfg_strength": 7.5},
    "slat_sampler_params": {"steps": 12, "cfg_strength": 3},
}


@dataclass
class GeneratedAsset:
    gaussians: Gaussians  # local frame
    mesh: object | None  # textured trimesh in the same local frame
    snapshots: PoseInputs  # four renderings with their local cameras


def _patch_ratio(mask: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """Fraction of masked pixels in every 14x14 DINOv2 patch of a 518x518 image."""
    mask = F.interpolate(mask.float(), size=(518, 518), mode="nearest")
    patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return patches.mean(dim=(-1, -2)).squeeze(1)  # (B, 37, 37)


def _build_amodal3r():
    from amodal3r.modules.utils import convert_module_to_f16
    from amodal3r.pipelines import Amodal3RImageTo3DPipeline

    class PanoAmodal3R(Amodal3RImageTo3DPipeline):
        """Amodal3R conditioned on RGBA crops; pixels outside the alpha mask count as occluded."""

        def to_fp16(self) -> None:
            for model in self.models.values():
                for p in model.parameters():
                    p.requires_grad = False
                if hasattr(model, "device"):  # the generative models; the DINOv2 encoder stays fp32
                    model.apply(convert_module_to_f16)

        def condition(self, rgba: torch.Tensor) -> dict:
            visible = rgba[:, 3:4] > 0
            image = rgba[:, :3] * visible
            features = self.encode_image(image)
            visible_ratio = _patch_ratio(visible.float()).flatten(1)[..., None].expand(-1, -1, features.shape[-1])
            occluded = torch.zeros_like(visible_ratio)  # no explicit occluder masks for panorama crops
            cond = torch.cat([features, visible_ratio, occluded], dim=1)
            return {"cond": cond.half(), "neg_cond": torch.zeros_like(cond[:1]).half()}

        @torch.no_grad()
        def generate(self, rgba: torch.Tensor, seed: int):
            cond = self.condition(rgba)
            torch.manual_seed(seed)
            ss_params = {**self.sparse_structure_sampler_params, **SAMPLER_PARAMS["sparse_structure_sampler_params"]}
            slat_params = {**self.slat_sampler_params, **SAMPLER_PARAMS["slat_sampler_params"]}
            with self.inject_sampler_multi_image("sparse_structure_sampler", len(rgba), ss_params["steps"]):
                coords = self.sample_sparse_structure(cond, 1, ss_params)
            with self.inject_sampler_multi_image("slat_sampler", len(rgba), slat_params["steps"]):
                slat = self.sample_slat(cond, coords, slat_params)
            out = self.decode_slat(slat, ["gaussian", "mesh"])
            return out["gaussian"][0], out["mesh"][0]

    pipeline = PanoAmodal3R.from_pretrained("Sm0kyWu/Amodal3R")
    pipeline.__class__ = PanoAmodal3R
    pipeline.to_fp16()
    return pipeline


def _build_trellis():
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline

    class Trellis:
        def __init__(self) -> None:
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")

        def to(self, device):
            self.pipeline.to(device)
            return self

        @torch.no_grad()
        def generate(self, rgba: torch.Tensor, seed: int):
            images = [Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"), "RGBA") for x in rgba]
            out = self.pipeline.run_multi_image(images, seed=seed, formats=["gaussian", "mesh"], **SAMPLER_PARAMS)
            return _to_amodal3r_gaussian(out["gaussian"][0]), out["mesh"][0]

    return Trellis()


def _to_amodal3r_gaussian(g):
    """TRELLIS and Amodal3R share the Gaussian layout; reuse the vendored renderer for both."""
    from amodal3r.representations import Gaussian

    out = Gaussian(aabb=g.aabb.tolist(), sh_degree=g.sh_degree, mininum_kernel_size=g.mininum_kernel_size,
                   scaling_bias=g.scaling_bias, opacity_bias=g.opacity_bias, scaling_activation=g.scaling_activation)
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        setattr(out, name, getattr(g, name))
    return out


class ObjectGenerator:
    def __init__(self, name: str = "amodal3r") -> None:
        self.backend = _build_amodal3r() if name == "amodal3r" else _build_trellis()

    def to(self, device) -> "ObjectGenerator":
        self.backend.to(device)
        return self

    @torch.no_grad()
    def __call__(self, rgba: torch.Tensor, seed: int = 42) -> GeneratedAsset:
        """``rgba``: (n, 4, 518, 518) recentered crops of one object, one per view."""
        from amodal3r.utils import postprocessing_utils, render_utils

        with torch.autocast("cuda", enabled=False):
            gaussian, mesh = self.backend.generate(rgba, seed)
            with torch.enable_grad():  # texture baking optimizes the texture with gradients
                glb = postprocessing_utils.to_glb(gaussian, mesh, simplify=0.95, texture_size=1024, verbose=False)
            w2c, K, (colors, depths) = render_utils.render_snapshot(gaussian, resolution=SNAPSHOT_RESOLUTION)

        snapshots = PoseInputs(
            rgb=torch.cat(colors).float(),
            depth=torch.cat(depths).float(),  # (4, H, W) alpha-weighted z-depth
            K=torch.stack(K).float(),
            c2w=torch.linalg.inv(torch.stack(w2c).float()),
        )
        return GeneratedAsset(_local_gaussians(gaussian), glb, snapshots)


def _local_gaussians(g) -> Gaussians:
    scales = g.get_scaling.float()
    rotations = g.get_rotation.float()[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    dc = g.get_features.float().flatten(1)  # (N, 3) DC coefficients
    harmonics = torch.zeros(len(dc), 3, 9, device=dc.device)  # degree-2 layout, only DC is non-zero
    harmonics[..., 0] = dc
    return Gaussians(
        means=g.get_xyz.float(),
        covariances=build_covariance(scales, rotations),
        harmonics=harmonics,
        opacities=g.get_opacity.float().squeeze(-1),
        scales=scales,
        rotations=rotations,
    )
