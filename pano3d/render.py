"""Gaussian rasterization with gsplat (pinhole views and cube-map panoramas)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization

from .gaussians import Gaussians
from .geometry import equirect_ray_directions, se3_inverse


def render_pinhole(
    gaussians: Gaussians,
    c2w: torch.Tensor,
    K: torch.Tensor,
    image_hw: tuple[int, int],
    background: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render V pinhole views.

    Args:
        c2w: (V, 4, 4) camera-to-world extrinsics.
        K: (V, 3, 3) normalized intrinsics.
    Returns:
        rgb (V, 3, H, W) in [0, 1] and z-depth (V, 1, H, W) in world units (0 where empty).
    """
    h, w = image_hw
    K_pix = K.clone().float()
    K_pix[:, 0] *= w
    K_pix[:, 1] *= h
    d_sh = gaussians.harmonics.shape[-1]
    out, alpha, _ = rasterization(
        means=gaussians.means.float(),
        quats=torch.zeros_like(gaussians.means[:, :1]).expand(-1, 4).float(),  # ignored when covars is given
        scales=gaussians.scales.float(),
        opacities=gaussians.opacities.float(),
        colors=gaussians.harmonics.transpose(-1, -2).float(),  # (N, d_sh, 3)
        viewmats=se3_inverse(c2w.float()),
        Ks=K_pix,
        width=w,
        height=h,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB+ED",
        sh_degree=int(math.isqrt(d_sh)) - 1,
        covars=gaussians.covariances.float(),
        rasterize_mode="classic",
        packed=False,
    )
    rgb = (out[..., :3] + (1 - alpha) * background).clamp(0, 1)
    depth = torch.where(alpha > 0, out[..., 3:4], torch.zeros_like(alpha))
    return rgb.permute(0, 3, 1, 2), depth.permute(0, 3, 1, 2)


# Cube faces as camera-to-panorama rotations (OpenCV cameras): front, right, back, left, up, down.
_CUBE_FACES = [
    torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]),
    torch.tensor([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    torch.tensor([[-1.0, 0, 0], [0, 1, 0], [0, 0, -1]]),
    torch.tensor([[0.0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]),
    torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]),
]


def render_panorama(
    gaussians: Gaussians, pano_c2w: torch.Tensor, height: int = 512, background: float = 1.0
) -> torch.Tensor:
    """Render an equirectangular image (3, H, 2H) by stitching six 90-degree cube faces."""
    device = gaussians.means.device
    face = height
    c2w = []
    for R in _CUBE_FACES:
        T = torch.eye(4, device=device)
        T[:3, :3] = R.to(device)
        c2w.append(pano_c2w.to(device) @ T)
    c2w = torch.stack(c2w)
    margin = 1.02  # slight overlap avoids seams at face borders
    f = 0.5 / math.tan(math.radians(45.0)) / margin
    K = torch.tensor([[f, 0, 0.5], [0, f, 0.5], [0, 0, 1]], device=device).expand(6, 3, 3)
    faces, _ = render_pinhole(gaussians, c2w, K, (face, face), background)

    # Look up every panorama pixel in the face that contains its ray.
    v, u = torch.meshgrid(
        (torch.arange(height, device=device) + 0.5) / height,
        (torch.arange(2 * height, device=device) + 0.5) / (2 * height),
        indexing="ij",
    )
    dirs = equirect_ray_directions(torch.stack([u, v], dim=-1))  # panorama camera frame
    rots = torch.stack([R.to(device) for R in _CUBE_FACES])  # face camera -> panorama camera
    local = torch.einsum("fji,hwj->fhwi", rots, dirs)  # rays in each face camera frame
    z = local[..., 2]
    best = torch.where(z > 0, z, torch.full_like(z, -1)).argmax(0)  # face with the most frontal view
    sel = torch.gather(local, 0, best[None, ..., None].expand(1, *best.shape, 3))[0]
    uv = sel[..., :2] / sel[..., 2:] * f  # in [-0.5, 0.5]
    grid = (uv * 2)[None]
    samples = torch.stack([F.grid_sample(faces[i : i + 1], grid, align_corners=False)[0] for i in range(6)])
    return torch.gather(samples, 0, best[None, None].expand(1, 3, *best.shape))[0]
