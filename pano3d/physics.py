"""Lightweight physics-aware placement: push objects out of collisions and snap floating ones."""

from __future__ import annotations

import torch
from pytorch3d.ops import knn_points, sample_farthest_points


def _downsample(points: torch.Tensor, n: int) -> torch.Tensor:
    if points.shape[0] <= n:
        return points
    return sample_farthest_points(points[None], K=n)[0][0]


@torch.no_grad()
def resolve_contacts(
    object_points: torch.Tensor,
    obstacle_points: list[torch.Tensor],
    gravity: torch.Tensor,
    num_proxy: int = 4096,
    collision_dist: float = 0.05,
    contact_dist: float = 0.05,
    max_step: float = 0.05,
    max_iters: int = 5,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Rigid translation that removes interpenetration and floating.

    At every iteration the object is either pushed out along the direction of its
    deepest penetrating points (distance < ``collision_dist``), or, if it floats
    (distance > ``contact_dist``), pulled towards the nearest obstacle with a bias
    along gravity.

    Args:
        object_points: (N, 3) points of the object to adjust.
        obstacle_points: point clouds of already placed objects (and/or background).
        gravity: (3,) world gravity direction.
    Returns:
        (3,) translation to add to the object.
    """
    device, dtype = object_points.device, object_points.dtype
    total = torch.zeros(3, device=device, dtype=dtype)
    center = object_points.mean(0)
    radius = (object_points - center).norm(dim=-1).max()
    keep = (radius * 2.5 + max_step * (max_iters + 2)).clamp_min(0.3)
    nearby = [p[(p - center).norm(dim=-1) < keep] for p in obstacle_points if p.numel() > 0]
    nearby = [p for p in nearby if p.shape[0] > 0]
    if not nearby:
        return total

    obstacles = _downsample(torch.cat(nearby), num_proxy).contiguous()
    proxy = _downsample(object_points, num_proxy // 2).contiguous()
    gravity = gravity.to(device, dtype) / gravity.norm().clamp_min(1e-6)

    for _ in range(max_iters):
        dist_sq, idx, _ = knn_points(proxy[None], obstacles[None], K=1)
        dists = dist_sq[0, :, 0].clamp_min(1e-12).sqrt()
        nearest = obstacles[idx[0, :, 0]]
        direction = torch.nn.functional.normalize(proxy - nearest, dim=-1)
        # Points that crossed a thin surface point the wrong way; flip them towards the object center.
        flip = (direction * (proxy.mean(0) - nearest)).sum(-1) < 0
        direction[flip] = -direction[flip]

        colliding = dists < collision_dist
        if colliding.any():
            depth = collision_dist - dists[colliding]
            k = max(1, int(depth.shape[0] * 0.05))
            top_depth, top_idx = torch.topk(depth, k)
            push = (direction[colliding][top_idx] * top_depth[:, None]).sum(0)
            delta = torch.nn.functional.normalize(push, dim=0) * (top_depth.mean() + 1e-3)
        elif dists.min() > contact_dist:
            pull = -direction[dists.argmin()]
            gap = dists.min() - contact_dist * 0.5
            if (pull * gravity).sum() > 0.2:  # roughly downwards: prefer falling along gravity
                pull = torch.nn.functional.normalize(0.6 * gravity + 0.4 * pull, dim=0)
            delta = pull * gap
        else:
            break  # in contact without penetration

        norm = delta.norm()
        if norm < tol:
            break
        delta = delta * min(1.0, max_step / norm.item())
        proxy = proxy + delta
        total = total + delta
    return total
