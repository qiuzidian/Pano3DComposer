"""Background modeling: panorama inpainting, background depth and feed-forward Gaussians."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_scatter import scatter_add, scatter_max

from ..gaussians import Gaussians, build_covariance
from ..geometry import equirect_ray_directions, fibonacci_sphere_grid
from .depth import DepthAnything3Pano, imagenet_normalize
from .gs_head import GSDPT, build_gs_head

SH_DEGREE = 2


def dilate(mask: torch.Tensor, kernel_size: int, iterations: int) -> torch.Tensor:
    """Binary dilation of masks with shape (B, 1, H, W)."""
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    for _ in range(iterations):
        mask = (F.conv2d(mask, kernel, padding=kernel_size // 2) > 0).float()
    return mask


class PanoramaInpainter(nn.Module):
    """LaMa fine-tuned on panoramas; the seam is handled with circular padding."""

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        from simple_lama_inpainting import SimpleLama

        # SimpleLama provides the big-lama architecture (TorchScript); we swap in our weights.
        self.model = SimpleLama(device="cpu").model
        self.model.load_state_dict({f"model.generator.{k}": v for k, v in state.items()}, strict=True)
        self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Inpaint (V, 3, H, W) panoramas inside (V, 1, H, W) masks."""
        w = images.shape[-1]
        pad = w // 8
        images = F.pad(images, (pad, pad, 0, 0), mode="circular")
        masks = dilate(F.pad(masks, (pad, pad, 0, 0), mode="circular"), kernel_size=7, iterations=2)
        return self.model(images, masks)[..., pad : pad + w]


class BackgroundReconstructor(nn.Module):
    """Predict background Gaussians from object-free panoramas.

    The background Depth Anything 3 sees the inpainted (object-free) panoramas and
    predicts metric background depth plus features; the GS-DPT head turns the
    features and the inpainted colors into per-pixel Gaussians, which are sampled
    on a Fibonacci sphere and fused in small voxels.

    Both networks must see the inpainted panorama: with object pixels blanked out,
    the head predicts dark, transparent Gaussians inside the holes.
    """

    work_hw = (504, 1008)  # DA3 input resolution (multiple of the 14-pixel patch)
    voxel_size = 0.002

    def __init__(self, depth_model: DepthAnything3Pano, gs_head: GSDPT) -> None:
        super().__init__()
        self.depth_model = depth_model
        self.gs_head = gs_head
        self.register_buffer("sphere_grid", fibonacci_sphere_grid(self.work_hw[1]), persistent=False)
        sh_mask = torch.ones((SH_DEGREE + 1) ** 2)
        for degree in range(1, SH_DEGREE + 1):
            sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        self.register_buffer("sh_mask", sh_mask, persistent=False)

    @torch.no_grad()
    def forward(
        self, panoramas: torch.Tensor, inpainted: torch.Tensor, object_masks: torch.Tensor, pano_c2w: torch.Tensor
    ) -> tuple[Gaussians, torch.Tensor]:
        """
        Args:
            panoramas, inpainted: (V, 3, H, W) original and inpainted panoramas in [0, 1].
            object_masks: (V, 1, H, W) union of the instance masks.
            pano_c2w: (V, 4, 4) panorama poses; only the first one anchors the prediction.
        Returns:
            Background Gaussians in world space and the completed depth (V, H, W).
        """
        V, _, H, W = panoramas.shape
        resize = lambda x, mode: F.interpolate(x, size=self.work_hw, mode=mode, **({"align_corners": True} if mode == "bilinear" else {}))

        fill = dilate(object_masks, kernel_size=7, iterations=2)
        colors = resize(inpainted * fill + panoramas * (1 - fill), "bilinear")
        out = self.depth_model(imagenet_normalize(colors)[None])
        c2w = pano_c2w[:1] @ out.c2w  # predicted poses are relative to the first panorama

        no_mask = torch.zeros_like(colors[:, :1])  # the head's mask channel is unused at inference
        with torch.autocast("cuda", enabled=False):
            head = self.gs_head(
                out.feats, H=self.work_hw[0], W=self.work_hw[1],
                images=torch.cat([colors, no_mask], dim=1)[None], patch_start_idx=0,
            )
        raw, conf = head["raw_gs"][0], head["raw_gs_conf"][0]  # (V, h, w, C), (V, h, w)

        # Sample every view on the Fibonacci sphere.
        grid = self.sphere_grid[None, :, None].expand(V, -1, -1, -1)  # (V, N, 1, 2)
        sample = lambda x, mode: F.grid_sample(x, grid, mode=mode, padding_mode="border")[..., 0]
        depth = sample(out.depth.clamp(0.05, 10.0)[:, None], "nearest")[:, 0]  # (V, N)
        raw = sample(rearrange(raw, "v h w c -> v c h w"), "bilinear").transpose(1, 2)  # (V, N, C)
        conf = sample(conf[:, None], "bilinear")[:, 0]  # (V, N)

        dirs = equirect_ray_directions(self.sphere_grid / 2 + 0.5)  # (N, 3)
        means = torch.einsum("vij,nj->vni", c2w[:, :3, :3], dirs) * depth[..., None] + c2w[:, None, :3, 3]
        means, feats = self._voxel_fusion(means.flatten(0, 1), raw.flatten(0, 1), conf.flatten())
        depth_full = F.interpolate(out.depth[:, None], size=(H, W), mode="bilinear", align_corners=True)[:, 0]
        return self._to_gaussians(means, feats), depth_full

    def _voxel_fusion(self, points: torch.Tensor, feats: torch.Tensor, conf: torch.Tensor):
        """Merge points falling into the same voxel with confidence-softmax weights."""
        voxels = (points / self.voxel_size).round().int()
        _, inverse = torch.unique(voxels, dim=0, return_inverse=True)
        conf_max, _ = scatter_max(conf, inverse, dim=0)
        w = torch.exp(conf - conf_max[inverse])
        w = (w / (scatter_add(w, inverse, dim=0)[inverse] + 1e-6))[:, None]
        return scatter_add(points * w, inverse, dim=0), scatter_add(feats * w, inverse, dim=0)

    def _to_gaussians(self, means: torch.Tensor, feats: torch.Tensor) -> Gaussians:
        opacity = feats[:, 0].sigmoid()
        scales, rotations, sh = feats[:, 1:].split((3, 4, 3 * (SH_DEGREE + 1) ** 2), dim=-1)
        scales = (0.001 * F.softplus(scales)).clamp_max(0.3)
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        sh = rearrange(sh, "n (c d) -> n c d", c=3) * self.sh_mask
        return Gaussians(
            means=means.float(),
            covariances=build_covariance(scales, rotations).float(),
            harmonics=sh.float(),
            opacities=opacity.float(),
            scales=scales.float(),
            rotations=rotations.float(),
        )


def build_background_reconstructor(
    depth_model: DepthAnything3Pano, gs_head_state: dict[str, torch.Tensor], device
) -> BackgroundReconstructor:
    return BackgroundReconstructor(depth_model, build_gs_head(gs_head_state, device)).to(device)
