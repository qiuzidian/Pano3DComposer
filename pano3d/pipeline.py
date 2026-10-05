"""End-to-end Pano3DComposer inference.

    panoramas + instance masks
      -> metric panoramic depth (Depth Anything 3)
      -> per object: image-to-3D asset -> OmniVGGT alignment -> upright prior
                     -> coarse-to-fine refinement -> (optional) contact resolution
      -> background: LaMa inpainting -> background depth -> GS-DPT Gaussians
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance

from .data import Scene, SceneObject
from .gaussians import Gaussians
from .geometry import (
    PANO_C2W,
    distance_to_zdepth,
    equirect_to_perspective,
    keep_yaw_only,
    scale_matrix,
    unproject_pinhole,
)
from .models.aligner import (
    OmniVGGTPoseNet,
    PoseInputs,
    first_pair_transform,
    fuse_candidates,
    relative_update,
)
from .models.background import PanoramaInpainter, build_background_reconstructor
from .models.depth import DepthAnything3Pano, imagenet_normalize
from .models.generator import GeneratedAsset, ObjectGenerator
from .physics import resolve_contacts
from .render import render_pinhole
from .weights import Checkpoint


@dataclass
class Config:
    checkpoint: Path = Path("checkpoints/pano3d.safetensors")
    generator: str = "amodal3r"  # "amodal3r" or "trellis"
    pose_fusion: str = "first"  # "first": first crop x first rendering; "weighted": n x V SE(3) fusion
    fusion_temperature: float = 0.1
    upright: bool = True
    refine: bool = True
    refine_max_steps: int = 5
    refine_min_improvement: float = 1e-3
    physics: bool = False
    background: bool | None = None  # None: only for single-panorama inputs
    seed: int = 42
    low_vram: bool = False  # keep idle models on the CPU
    device: str = "cuda"


@dataclass
class PlacedObject:
    name: str
    transform: torch.Tensor  # (4, 4) local-to-world, including the anisotropic scale
    gaussians: Gaussians  # world space
    mesh: object | None  # trimesh.Trimesh in world space
    refine_steps: int = 0
    chamfer: float = float("nan")  # single-direction CD from the depth points to the asset
    stages: list[tuple[str, torch.Tensor, float]] = field(default_factory=list)  # (stage, transform, chamfer)
    asset: GeneratedAsset | None = None  # local-frame asset and its reference renderings
    observations: PoseInputs | None = None  # real crops with metric depth


@dataclass
class SceneResult:
    objects: list[PlacedObject] = field(default_factory=list)
    background: Gaussians | None = None
    pano_c2w: torch.Tensor | None = None
    pano_depth: torch.Tensor | None = None  # (M, H, W) ray-length depth
    inpainted: torch.Tensor | None = None


class Pano3DComposer:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.device = torch.device(config.device)
        # Models are built and loaded directly on their target device, one at a time,
        # so host memory never holds more than one model.
        load_on = torch.device("cpu") if config.low_vram else self.device
        ckpt = Checkpoint(config.checkpoint)
        self.pano_depth = DepthAnything3Pano.from_state_dict(ckpt.state_dict("pano_depth"), load_on)
        self.aligner = OmniVGGTPoseNet.from_state_dict(ckpt.state_dict("aligner"), load_on)
        self.refiner = OmniVGGTPoseNet.from_state_dict(ckpt.state_dict("refiner"), load_on) if config.refine else None
        self.background = build_background_reconstructor(
            DepthAnything3Pano.from_state_dict(ckpt.state_dict("bg_depth"), load_on), ckpt.state_dict("gs_head"), load_on
        )
        self.inpainter = PanoramaInpainter(ckpt.state_dict("lama")).to(load_on)
        self.generator = ObjectGenerator(config.generator).to(load_on)

    def _modules(self):
        return [m for m in (self.pano_depth, self.aligner, self.refiner, self.inpainter, self.background, self.generator) if m is not None]

    @contextlib.contextmanager
    def _on_gpu(self, module):
        """Temporarily move a model to the GPU when running with ``low_vram``."""
        if self.cfg.low_vram:
            module.to(self.device)
        try:
            yield module
        finally:
            if self.cfg.low_vram:
                module.to("cpu")
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, scene: Scene) -> SceneResult:
        torch.manual_seed(self.cfg.seed)
        panoramas = scene.panoramas.to(self.device)
        result = SceneResult()

        # 1. Metric depth (and, if not given, relative poses) of every panorama.
        with self._on_gpu(self.pano_depth):
            work = F.interpolate(panoramas, size=(504, 1008), mode="bilinear", align_corners=True)
            out = self.pano_depth(imagenet_normalize(work)[None])
        result.pano_depth = F.interpolate(out.depth[:, None], size=panoramas.shape[-2:], mode="bilinear", align_corners=True)[:, 0]
        pano_c2w = scene.pano_c2w.to(self.device) if scene.pano_c2w is not None else PANO_C2W.to(self.device) @ out.c2w
        result.pano_c2w = pano_c2w

        # 2. Objects, placed one after another so that physics can account for earlier ones.
        placed_points: list[torch.Tensor] = []
        for obj in scene.objects:
            placed = self.place_object(obj, result.pano_depth, pano_c2w, placed_points)
            result.objects.append(placed)
            placed_points.append(placed.gaussians.means)
            print(f"[{obj.name}] views={len(obj.views)} refine_steps={placed.refine_steps} cd={placed.chamfer:.4f}")

        # 3. Background Gaussians.
        use_background = self.cfg.background if self.cfg.background is not None else len(panoramas) == 1
        if use_background:
            masks = scene.instance_masks.to(self.device)
            with self._on_gpu(self.inpainter):
                result.inpainted = torch.stack([self.inpainter(p[None], m[None])[0] for p, m in zip(panoramas, masks)])
            with self._on_gpu(self.background):
                result.background, _ = self.background(panoramas, result.inpainted, masks, pano_c2w)
        return result

    # ------------------------------------------------------------------
    def object_observations(self, obj: SceneObject, pano_depth: torch.Tensor, pano_c2w: torch.Tensor) -> PoseInputs:
        """Real crops of an object with metric z-depth from the panoramic depth."""
        rgb, depth, K, c2w = [], [], [], []
        for view in obj.views:
            distance = equirect_to_perspective(
                pano_depth[view.view_index].cpu().numpy(), view.fov, view.theta, view.phi, view.mask.shape, "bilinear"
            )
            z = distance_to_zdepth(torch.from_numpy(distance), view.K) * view.mask
            rgb.append(view.rgb)
            depth.append(z)
            K.append(view.K)
            c2w.append(pano_c2w[view.view_index].cpu() @ view.c2p)
        return PoseInputs(*(torch.stack(x).float().to(self.device) for x in (rgb, depth, K, c2w)))

    def place_object(
        self, obj: SceneObject, pano_depth: torch.Tensor, pano_c2w: torch.Tensor, placed_points: list[torch.Tensor]
    ) -> PlacedObject:
        crops = self.object_observations(obj, pano_depth, pano_c2w)
        with self._on_gpu(self.generator):
            asset = self.generator(torch.stack([v.generator_rgba for v in obj.views]).to(self.device), seed=self.cfg.seed)
        renders = asset.snapshots

        # Coarse alignment: renderings (local frame, known cameras) first, then the real crops.
        with self._on_gpu(self.aligner):
            pred, scale = self.aligner(PoseInputs.cat(renders, crops).resized(), num_known=len(renders.rgb))
        if self.cfg.pose_fusion == "weighted":
            rigid = fuse_candidates(pred, renders.c2w, crops.c2w, self.cfg.fusion_temperature)
        else:
            rigid = first_pair_transform(pred, renders.c2w, crops.c2w)
        scale = scale[0]  # the scale token of the first view carries the object scale
        aligned = rigid
        if self.cfg.upright:
            rigid = keep_yaw_only(rigid)

        # Pseudo point cloud of the visible surface, used to monitor the refinement.
        target = torch.cat([unproject_pinhole(d, k, c) for d, k, c in zip(crops.depth, crops.K, crops.c2w)])
        target = target[torch.randperm(len(target), device=target.device)[:10000]]
        local = asset.gaussians

        def chamfer(T: torch.Tensor) -> torch.Tensor:
            placed = local.means @ (T @ scale_matrix(scale))[:3, :3].T + T[:3, 3]
            return chamfer_distance(target[None], placed[None], single_directional=True)[0]

        S = scale_matrix(scale)
        stages = [("aligned", aligned @ S, float(chamfer(aligned)))]
        if self.cfg.upright:
            stages.append(("upright", rigid @ S, float(chamfer(rigid))))
        best_cd, steps = chamfer(rigid), 0
        if self.refiner is not None:
            rigid, best_cd, steps = self.refine(local, rigid, scale, crops, chamfer, best_cd)
            stages.append(("refined", rigid @ S, float(best_cd)))

        if self.cfg.physics and placed_points:
            gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)  # world z is up
            placed = local.transformed(rigid @ scale_matrix(scale)).means
            shift = resolve_contacts(placed, placed_points, gravity)
            rigid = rigid.clone()
            rigid[:3, 3] += shift
            stages.append(("physics", rigid @ S, float(chamfer(rigid))))

        T = rigid @ S
        mesh = None
        if asset.mesh is not None:
            mesh = asset.mesh.copy()
            mesh.apply_transform(T.double().cpu().numpy())
        return PlacedObject(obj.name, T, local.transformed(T), mesh, steps, float(best_cd), stages, asset, crops)

    def refine(self, local: Gaussians, rigid, scale, crops: PoseInputs, chamfer, best_cd):
        """Coarse-to-fine: render the current placement at the real cameras and predict a rigid update.

        The scale is kept fixed. Iterates while the Chamfer distance to the depth points improves by
        more than ``refine_min_improvement``.
        """
        n = len(crops.rgb)
        real = crops.resized()
        steps = 0
        with self._on_gpu(self.refiner):
            for _ in range(self.cfg.refine_max_steps):
                current = local.transformed(rigid @ scale_matrix(scale))
                rgb, depth = render_pinhole(current, crops.c2w, crops.K, real.rgb.shape[-2:], background=1.0)
                rendered = PoseInputs(rgb, depth[:, 0], crops.K, torch.zeros_like(crops.c2w))
                pred, _ = self.refiner(PoseInputs.cat(real, rendered), num_known=n)
                candidate = relative_update(pred, crops.c2w, n) @ rigid
                if self.cfg.upright:
                    candidate = keep_yaw_only(candidate)
                cd = chamfer(candidate)
                if cd >= best_cd:
                    break
                improvement = best_cd - cd
                rigid, best_cd, steps = candidate, cd, steps + 1
                if improvement < self.cfg.refine_min_improvement:
                    break
        return rigid, best_cd, steps


# ----------------------------------------------------------------------
# glTF is y-up; meshes are exported in the z-up world rotated accordingly (x, z, -y).
Z_UP_TO_GLTF = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def save_result(result: SceneResult, out_dir: str | Path, save_gaussians: bool = True) -> None:
    """Write meshes, Gaussians, transforms and a preview render."""
    import trimesh
    from torchvision.utils import save_image

    from .render import render_panorama

    out_dir = Path(out_dir)
    (out_dir / "objects").mkdir(parents=True, exist_ok=True)
    transforms = {}
    meshes = trimesh.Scene()
    for obj in result.objects:
        transforms[obj.name] = {
            "local_to_world": obj.transform.cpu().tolist(),
            "refine_steps": obj.refine_steps,
            "chamfer_to_depth": obj.chamfer,
            "stages": {name: {"local_to_world": T.cpu().tolist(), "chamfer_to_depth": cd} for name, T, cd in obj.stages},
        }
        if obj.mesh is not None:
            mesh = obj.mesh.copy()
            mesh.apply_transform(Z_UP_TO_GLTF)
            mesh.export(out_dir / "objects" / f"{obj.name}.glb")
            meshes.add_geometry(mesh, node_name=obj.name)
        if save_gaussians:
            obj.gaussians.save_ply(out_dir / "objects" / f"{obj.name}.ply")
    (out_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))
    if len(meshes.geometry):
        meshes.export(out_dir / "scene_objects.glb")

    if result.pano_depth is not None:
        depth = result.pano_depth / result.pano_depth.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        for i, d in enumerate(depth):
            save_image(d[None], out_dir / f"depth_{i}.png")
    if result.inpainted is not None:
        for i, image in enumerate(result.inpainted):
            save_image(image, out_dir / f"inpainted_{i}.png")
    if not save_gaussians:
        return

    parts = [o.gaussians for o in result.objects]
    if result.background is not None:
        result.background.save_ply(out_dir / "background.ply")
        parts.append(result.background)
    if parts:
        scene = Gaussians.concat(parts)
        scene.save_ply(out_dir / "scene.ply")
        save_image(render_panorama(scene, result.pano_c2w[0]), out_dir / "render_panorama.png")
