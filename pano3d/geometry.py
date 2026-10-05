"""Camera conventions, panorama <-> perspective projection and SE(3) helpers.

Conventions used throughout the code base:

* Cameras follow the OpenCV convention (x right, y down, z forward).
* Extrinsics are camera-to-world (c2w) 4x4 matrices unless stated otherwise.
* Perspective intrinsics are *normalized*: focal lengths and principal point are
  divided by the image width / height (so cx = cy = 0.5).
* A single panorama is placed at the origin with ``PANO_C2W`` so that the world
  z-axis points up (opposite to gravity).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Camera-to-world rotation of a panorama captured at the origin: camera "down" (+y)
# maps to world -z, i.e. the world z-axis is the up direction.
PANO_C2W = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


# -----------------------------------------------------------------------------
# Perspective crops of an equirectangular panorama
# -----------------------------------------------------------------------------
def _perspective_to_equirect_lonlat(fov_deg: float, theta_deg: float, phi_deg: float, h: int, w: int):
    """Longitude / latitude (radians) of every pixel of a perspective view."""
    hfov = float(h) / w * fov_deg
    w_len = np.tan(np.radians(fov_deg / 2.0))
    h_len = np.tan(np.radians(hfov / 2.0))

    x_map = np.ones([h, w], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, w), [h, 1])
    z_map = -np.tile(np.linspace(-h_len, h_len, h), [w, 1]).T
    xyz = np.stack((x_map, y_map, z_map), axis=2)
    xyz /= np.linalg.norm(xyz, axis=2, keepdims=True)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    R1, _ = cv2.Rodrigues(z_axis * np.radians(theta_deg))
    R2, _ = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi_deg))

    xyz = (R2 @ R1 @ xyz.reshape(-1, 3).T).T
    lat = -np.arcsin(xyz[:, 2]).reshape(h, w)
    lon = np.arctan2(xyz[:, 1], xyz[:, 0]).reshape(h, w)
    return lon, lat


def equirect_to_perspective(
    pano: np.ndarray, fov_deg: float, theta_deg: float, phi_deg: float, out_hw: tuple[int, int], mode: str = "bilinear"
) -> np.ndarray:
    """Sample a perspective view from an equirectangular image of shape (H, W, ...).

    ``theta`` is the yaw in [-180, 180] and ``phi`` the pitch in [-90, 90], in degrees.
    """
    he, we = pano.shape[:2]
    lon, lat = _perspective_to_equirect_lonlat(fov_deg, theta_deg, phi_deg, *out_hw)
    cx, cy = (we - 1) / 2.0, (he - 1) / 2.0
    map_x = (np.degrees(lon) / 180 * cx + cx).astype(np.float32)
    map_y = (np.degrees(lat) / 90 * cy + cy).astype(np.float32)
    interp = {"bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST, "bicubic": cv2.INTER_CUBIC}[mode]
    return cv2.remap(pano, map_x, map_y, interp, borderMode=cv2.BORDER_WRAP)


def crop_camera(fov_deg: float, theta_deg: float, phi_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized intrinsics and camera-to-panorama rotation of a perspective crop."""
    f = 0.5 / np.tan(0.5 * fov_deg / 180.0 * np.pi)
    K = np.array([[f, 0, 0.5], [0, f, 0.5], [0, 0, 1]], np.float32)

    R1, _ = cv2.Rodrigues(np.array([0.0, 1.0, 0.0], np.float32) * np.radians(theta_deg))
    R2, _ = cv2.Rodrigues(np.dot(R1, np.array([1.0, 0.0, 0.0], np.float32)) * np.radians(phi_deg))
    c2p = np.eye(4, dtype=np.float32)
    c2p[:3, :3] = R2 @ R1
    return torch.from_numpy(K), torch.from_numpy(c2p)


def crop_view_from_mask(mask: np.ndarray) -> tuple[float, float, float]:
    """Choose (fov, theta, phi) in degrees of a perspective crop that covers ``mask``.

    The crop is centered on the mask centroid and its field of view covers the mask
    bounding box with a small margin. Masks wrapping around the panorama seam are
    handled by rolling the panorama by half its width.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    wraps_around = xs.min() == 0 and xs.max() == w - 1
    if wraps_around:
        ys, xs = np.nonzero(np.roll(mask, w // 2, axis=1))

    theta = (xs.mean() / w) * 360.0 - 180.0
    phi = 90.0 - (ys.mean() / h) * 180.0
    if wraps_around:
        theta = (theta + 360.0) % 360 - 180

    fov_x = (xs.max() - xs.min()) / w * 360.0 * math.cos(math.radians(phi)) * 1.25
    fov_y = (ys.max() - ys.min()) / h * 180.0 * 1.1
    fov = min(max(fov_x, fov_y, 30.0), 120.0)
    return float(fov), float(theta), float(phi)


def distance_to_zdepth(distance: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Convert ray-length depth (..., H, W) of a pinhole view to z-depth.

    ``K`` holds normalized intrinsics with shape (..., 3, 3).
    """
    h, w = distance.shape[-2:]
    v, u = torch.meshgrid(
        torch.linspace(0, 1, h, device=distance.device), torch.linspace(0, 1, w, device=distance.device), indexing="ij"
    )
    fx, fy = K[..., 0, 0, None, None], K[..., 1, 1, None, None]
    cx, cy = K[..., 0, 2, None, None], K[..., 1, 2, None, None]
    x, y = (u - cx) / fx, (v - cy) / fy
    return distance / torch.sqrt(x**2 + y**2 + 1.0)


def unproject_pinhole(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Back-project the valid pixels (depth > 0) of a z-depth map (H, W) to world points (N, 3)."""
    h, w = depth.shape
    v, u = torch.meshgrid(
        (torch.arange(h, device=depth.device) + 0.5) / h, (torch.arange(w, device=depth.device) + 0.5) / w, indexing="ij"
    )
    valid = depth > 0
    z = depth[valid]
    x = (u[valid] - K[0, 2]) / K[0, 0] * z
    y = (v[valid] - K[1, 2]) / K[1, 1] * z
    points = torch.stack([x, y, z], dim=-1)
    return points @ c2w[:3, :3].T + c2w[:3, 3]


# -----------------------------------------------------------------------------
# Equirectangular rays
# -----------------------------------------------------------------------------
def equirect_ray_directions(uv: torch.Tensor) -> torch.Tensor:
    """Unit ray directions in the panorama camera frame for uv in [0, 1]^2 (u right, v down)."""
    lon = uv[..., 0] * 2 * math.pi - math.pi
    lat = uv[..., 1] * math.pi - math.pi / 2
    return torch.stack([torch.cos(lat) * torch.sin(lon), torch.sin(lat), torch.cos(lat) * torch.cos(lon)], dim=-1)


def fibonacci_sphere_grid(width: int) -> torch.Tensor:
    """Near-uniform samples on the sphere as ``grid_sample`` coordinates in [-1, 1]^2.

    The number of samples matches the pixel density at the equator of an
    equirectangular image of the given width.
    """
    num = int(4 * math.pi * (width / 2 / math.pi) ** 2)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    y = torch.linspace(1, -1, num)
    radius = torch.sqrt(1 - y**2)
    angle = golden_angle * torch.arange(num)
    x, z = torch.cos(angle) * radius, torch.sin(angle) * radius
    lon, lat = torch.atan2(x, z), torch.asin(y)
    return torch.stack([lon / math.pi, lat / (math.pi / 2)], dim=-1)


# -----------------------------------------------------------------------------
# SE(3) helpers
# -----------------------------------------------------------------------------
def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of rigid transforms with shape (..., 4, 4)."""
    R, t = T[..., :3, :3], T[..., :3, 3:]
    inv = torch.zeros_like(T)
    inv[..., :3, :3] = R.transpose(-1, -2)
    inv[..., :3, 3:] = -R.transpose(-1, -2) @ t
    inv[..., 3, 3] = 1.0
    return inv


def scale_matrix(scale: torch.Tensor) -> torch.Tensor:
    """Homogeneous 4x4 matrix diag(sx, sy, sz, 1) from a (..., 3) scale."""
    S = torch.diag_embed(F.pad(scale, (0, 1), value=1.0))
    return S


def keep_yaw_only(T: torch.Tensor) -> torch.Tensor:
    """Upright prior: drop pitch and roll of the rotation (ZYX Euler), keep yaw about world z."""
    R = T[..., :3, :3]
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    out = T.clone()
    out[..., :3, :3] = 0
    out[..., 0, 0], out[..., 0, 1] = c, -s
    out[..., 1, 0], out[..., 1, 1] = s, c
    out[..., 2, 2] = 1.0
    return out


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector of rotation matrices (..., 3, 3)."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos)
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    sin = torch.sin(angle)[..., None]
    small = sin.abs() < 1e-6
    return torch.where(small, 0.5 * w, angle[..., None] * w / (2 * sin.clamp_min(1e-12)))


def _skew(w: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(w[..., 0])
    return torch.stack(
        [zero, -w[..., 2], w[..., 1], w[..., 2], zero, -w[..., 0], -w[..., 1], w[..., 0], zero], dim=-1
    ).reshape(*w.shape[:-1], 3, 3)


def _so3_jacobians(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation matrix exp(w) and the left Jacobian V used by the SE(3) exponential."""
    angle = w.norm(dim=-1, keepdim=True)[..., None].clamp_min(1e-12)
    W = _skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(W)
    a = torch.sin(angle) / angle
    b = (1 - torch.cos(angle)) / angle**2
    c = (angle - torch.sin(angle)) / angle**3
    small = angle < 1e-6
    a = torch.where(small, torch.ones_like(a), a)
    b = torch.where(small, torch.full_like(b, 0.5), b)
    c = torch.where(small, torch.full_like(c, 1 / 6), c)
    R = eye + a * W + b * W @ W
    V = eye + b * W + c * W @ W
    return R, V


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from axis-angle vectors (..., 3)."""
    return _so3_jacobians(w)[0]


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """Twist (v, w) of rigid transforms (..., 4, 4)."""
    w = so3_log(T[..., :3, :3])
    _, V = _so3_jacobians(w)
    v = torch.linalg.solve(V, T[..., :3, 3:])[..., 0]
    return torch.cat([v, w], dim=-1)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Rigid transforms (..., 4, 4) from twists (v, w)."""
    R, V = _so3_jacobians(xi[..., 3:])
    T = torch.zeros(*xi.shape[:-1], 4, 4, dtype=xi.dtype, device=xi.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = (V @ xi[..., :3, None])[..., 0]
    T[..., 3, 3] = 1.0
    return T


def pose_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation angle plus Euclidean translation distance between poses."""
    angle = so3_log(A[..., :3, :3].transpose(-1, -2) @ B[..., :3, :3]).norm(dim=-1)
    return angle + (A[..., :3, 3] - B[..., :3, 3]).norm(dim=-1)


def weighted_se3_mean(transforms: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean of rigid transforms (N, 4, 4) in the tangent space of the heaviest one."""
    T_ref = transforms[weights.argmax()]
    xi = se3_log(se3_inverse(T_ref)[None] @ transforms)
    return T_ref @ se3_exp((weights[:, None] * xi).sum(0))
