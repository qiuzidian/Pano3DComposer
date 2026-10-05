"""Input loading: panoramas, instance masks and per-object perspective crops.

Scene folder layout (one or more panoramas of the same room)::

    scene/
      panoramas/<view>.png            equirectangular RGB images (2:1)
      masks/<view>/<instance>.png     binary instance masks; the same <instance>
                                      name across views means the same object
      poses.json                      optional {"<view>": 4x4 panorama-to-world}

Without ``poses.json`` a single panorama is placed at the origin, and multiple
panoramas use the relative poses estimated by Depth Anything 3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .geometry import PANO_C2W, crop_camera, crop_view_from_mask, equirect_to_perspective

PANO_HW = (512, 1024)  # working resolution of panoramas and masks
CROP_SIZE = 518  # resolution of object crops fed to the generator
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass
class ObjectView:
    """One observation of an object: a perspective crop of one panorama."""

    view_index: int
    fov: float
    theta: float
    phi: float
    K: torch.Tensor  # (3, 3) normalized intrinsics
    c2p: torch.Tensor  # (4, 4) crop camera to panorama camera
    rgb: torch.Tensor  # (3, S, S) crop on white background
    mask: torch.Tensor  # (S, S) bool
    generator_rgba: torch.Tensor  # (4, S, S) recentered crop for the image-to-3D generator
    depth: torch.Tensor | None = None  # (S, S) metric z-depth inside the mask, filled later


@dataclass
class SceneObject:
    name: str
    views: list[ObjectView] = field(default_factory=list)


@dataclass
class Scene:
    name: str
    panoramas: torch.Tensor  # (M, 3, H, W) in [0, 1]
    instance_masks: torch.Tensor  # (M, 1, H, W) union of all instance masks
    objects: list[SceneObject]
    pano_c2w: torch.Tensor | None  # (M, 4, 4), None -> estimate with Depth Anything 3


def recenter_rgba(rgba: np.ndarray, border_ratio: float = 0.15, size: int = CROP_SIZE) -> np.ndarray:
    """Center the masked object in a square canvas with a margin (black background)."""
    mask = rgba[..., 3]
    s = max(rgba.shape[:2])
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    if h == 0 or w == 0:
        raise ValueError("empty object crop")
    scale = int(s * (1 - border_ratio)) / max(h, w)
    h2, w2 = int(h * scale), int(w * scale)
    out = np.zeros((s, s, 4), dtype=np.uint8)
    oy, ox = (s - h2) // 2, (s - w2) // 2
    out[oy : oy + h2, ox : ox + w2] = cv2.resize(rgba[y0:y1, x0:x1], (w2, h2), interpolation=cv2.INTER_AREA)
    alpha = out[..., 3:].astype(np.float32) / 255
    rgb = (out[..., :3] * alpha).clip(0, 255).astype(np.uint8)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(out[..., 3], (size, size), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([rgb, mask[..., None]], axis=-1)


def make_object_view(pano: np.ndarray, mask: np.ndarray, view_index: int) -> ObjectView:
    """Cut a distortion-free perspective crop of one instance out of a panorama.

    Args:
        pano: (H, W, 3) uint8 RGB panorama.
        mask: (H, W) bool instance mask.
    """
    fov, theta, phi = crop_view_from_mask(mask)
    rgba = np.concatenate([pano, mask[..., None].astype(np.uint8) * 255], axis=-1)
    crop = equirect_to_perspective(rgba, fov, theta, phi, (CROP_SIZE, CROP_SIZE), mode="bilinear")
    alpha = crop[..., 3:4] > 0

    rgb_white = np.where(alpha, crop[..., :3], 255).astype(np.float32) / 255
    rgba_gen = recenter_rgba(crop).astype(np.float32) / 255
    K, c2p = crop_camera(fov, theta, phi)
    return ObjectView(
        view_index=view_index,
        fov=fov,
        theta=theta,
        phi=phi,
        K=K,
        c2p=c2p,
        rgb=torch.from_numpy(rgb_white).permute(2, 0, 1),
        mask=torch.from_numpy(alpha[..., 0]),
        generator_rgba=torch.from_numpy(rgba_gen).permute(2, 0, 1),
    )


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(image, PANO_HW[::-1], interpolation=cv2.INTER_AREA)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(mask, PANO_HW[::-1], interpolation=cv2.INTER_NEAREST) > 128


def load_scene(scene_dir: str | Path, min_mask_pixels: int = 64) -> Scene:
    scene_dir = Path(scene_dir)
    pano_paths = sorted(p for p in (scene_dir / "panoramas").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not pano_paths:
        raise FileNotFoundError(f"no panoramas found in {scene_dir / 'panoramas'}")

    panoramas, union_masks, objects = [], [], {}
    for view_index, pano_path in enumerate(pano_paths):
        pano = _read_rgb(pano_path)
        union = np.zeros(PANO_HW, dtype=bool)
        mask_dir = scene_dir / "masks" / pano_path.stem
        mask_paths = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if mask_dir.exists() else []
        for mask_path in mask_paths:
            mask = _read_mask(mask_path)
            union |= mask
            if mask.sum() < min_mask_pixels:
                continue
            obj = objects.setdefault(mask_path.stem, SceneObject(name=mask_path.stem))
            obj.views.append(make_object_view(pano, mask, view_index))
        panoramas.append(torch.from_numpy(pano).permute(2, 0, 1).float() / 255)
        union_masks.append(torch.from_numpy(union)[None].float())

    pose_file = scene_dir / "poses.json"
    if pose_file.exists():
        poses = json.loads(pose_file.read_text())
        pano_c2w = torch.stack([torch.tensor(poses[p.stem], dtype=torch.float32) for p in pano_paths])
    elif len(pano_paths) == 1:
        pano_c2w = PANO_C2W[None].clone()
    else:
        pano_c2w = None

    return Scene(
        name=scene_dir.name,
        panoramas=torch.stack(panoramas),
        instance_masks=torch.stack(union_masks),
        objects=list(objects.values()),
        pano_c2w=pano_c2w,
    )
