"""Dump the intermediate results of every pipeline stage (for inspection and the demo page).

    process/
      panorama.jpg, instances.jpg   input panorama, instance masks and crop frusta
      depth.jpg                     metric panoramic depth
      inpainted.jpg                 object-free panorama
      objects_<stage>.jpg           all objects rendered into the panorama after each stage
      background.jpg, scene.jpg     background Gaussians alone and the full scene
      <object>/
        crop.jpg, crop_depth.jpg    perspective crop and its metric depth
        generator_input.png         recentered RGBA crop fed to the image-to-3D model
        snapshot_<i>.jpg            reference renderings of the generated asset (local frame)
        overlay_<stage>.jpg         the placed asset rendered over the crop after each stage
      process.json                  per-object statistics
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import IMAGE_SUFFIXES, PANO_HW, Scene
from .gaussians import Gaussians
from .geometry import equirect_to_perspective
from .pipeline import SceneResult
from .render import render_panorama, render_pinhole

PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167)]


def _to_uint8(image) -> np.ndarray:
    """(3, H, W) tensor or (H, W, 3) array in [0, 1] -> (H, W, 3) uint8 RGB."""
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu()
        image = image.permute(1, 2, 0).numpy() if image.ndim == 3 and image.shape[0] in (3, 4) else image.numpy()
    return (np.clip(image, 0, 1) * 255).round().astype(np.uint8)


def _write(path: Path, rgb: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGRA if rgb.shape[-1] == 4 else cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix == ".jpg" else [])


def _colorize_depth(depth: torch.Tensor, max_depth: float | None = None) -> np.ndarray:
    """Turbo colormap of a depth map (near = red, far = blue); zeros stay light grey."""
    d = depth.float().cpu().numpy()
    valid = d > 0
    if not valid.any():
        return np.full((*d.shape, 3), 235, np.uint8)
    lo, hi = d[valid].min(), max_depth or np.percentile(d[valid], 99)
    norm = 1 - np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    color = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    color[~valid] = 235
    return color


def _render_with_alpha(gaussians: Gaussians, c2w, K, hw):
    """Premultiplied color and alpha (from renders on black and white backgrounds)."""
    on_black, _ = render_pinhole(gaussians, c2w, K, hw, background=0.0)
    on_white, _ = render_pinhole(gaussians, c2w, K, hw, background=1.0)
    alpha = (1 - (on_white - on_black).mean(1, keepdim=True)).clamp(0, 1)
    return on_black, alpha


def _crop_outline(view, hw) -> list[np.ndarray]:
    """Panorama polylines of the border of a perspective crop (split at the seam)."""
    n = 64
    t = torch.linspace(0, 1, n)
    border = torch.cat([
        torch.stack([t, torch.zeros(n)], -1), torch.stack([torch.ones(n), t], -1),
        torch.stack([1 - t, torch.ones(n)], -1), torch.stack([torch.zeros(n), 1 - t], -1),
    ])
    pix = torch.cat([border, torch.ones(len(border), 1)], -1)
    rays = (view.c2p[:3, :3] @ torch.linalg.inv(view.K) @ pix.T).T
    rays = rays / rays.norm(dim=-1, keepdim=True)
    lon, lat = torch.atan2(rays[:, 0], rays[:, 2]), torch.asin(rays[:, 1].clamp(-1, 1))
    u = ((lon + math.pi) / (2 * math.pi) * hw[1]).numpy()
    v = ((lat + math.pi / 2) / math.pi * hw[0]).numpy()
    pts = np.stack([u, v], -1)
    # Break the polyline where it jumps across the left/right seam.
    breaks = np.nonzero(np.abs(np.diff(u)) > hw[1] / 2)[0] + 1
    return [seg.round().astype(np.int32) for seg in np.split(pts, breaks) if len(seg) > 1]


def _instances_image(scene: Scene, scene_dir: Path, view_index: int) -> np.ndarray:
    pano = _to_uint8(scene.panoramas[view_index]).copy()
    view_name = sorted(p for p in (scene_dir / "panoramas").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[view_index].stem
    overlay = pano.copy()
    for i, obj in enumerate(scene.objects):
        mask_path = next((scene_dir / "masks" / view_name).glob(f"{obj.name}.*"), None)
        if mask_path is None:
            continue
        mask = cv2.resize(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), PANO_HW[::-1], interpolation=cv2.INTER_NEAREST) > 128
        overlay[mask] = PALETTE[i % len(PALETTE)]
    image = cv2.addWeighted(pano, 0.45, overlay, 0.55, 0)
    for i, obj in enumerate(scene.objects):
        for view in obj.views:
            if view.view_index == view_index:
                for seg in _crop_outline(view, PANO_HW):
                    cv2.polylines(image, [seg], False, PALETTE[i % len(PALETTE)], 2, cv2.LINE_AA)
    return image


@torch.no_grad()
def save_process(scene: Scene, scene_dir: str | Path, result: SceneResult, out_dir: str | Path) -> None:
    scene_dir, out = Path(scene_dir), Path(out_dir) / "process"
    pano_c2w = result.pano_c2w[0]
    _write(out / "panorama.jpg", _to_uint8(scene.panoramas[0]))
    _write(out / "instances.jpg", _instances_image(scene, scene_dir, 0))
    _write(out / "depth.jpg", _colorize_depth(result.pano_depth[0]))
    if result.inpainted is not None:
        _write(out / "inpainted.jpg", _to_uint8(result.inpainted[0]))

    stats = {"scene": scene.name, "num_panoramas": len(scene.panoramas), "objects": []}
    stage_names = [name for name, _, _ in result.objects[0].stages] if result.objects else []
    for i, placed in enumerate(result.objects):
        obj = next(o for o in scene.objects if o.name == placed.name)
        view, crops, asset = obj.views[0], placed.observations, placed.asset
        d = out / placed.name
        hw = tuple(crops.rgb.shape[-2:])
        full_crop = equirect_to_perspective(
            _to_uint8(scene.panoramas[view.view_index]), view.fov, view.theta, view.phi, hw, "bilinear"
        )
        _write(d / "crop.jpg", full_crop)
        _write(d / "crop_depth.jpg", _colorize_depth(crops.depth[0]))
        _write(d / "generator_input.png", _to_uint8(view.generator_rgba))
        for k, snapshot in enumerate(asset.snapshots.rgb):
            _write(d / f"snapshot_{k}.jpg", _to_uint8(snapshot))

        mask = view.mask.numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        background = torch.from_numpy(full_crop).permute(2, 0, 1).float().to(crops.rgb.device) / 255
        for stage, T, _ in placed.stages:
            color, alpha = _render_with_alpha(asset.gaussians.transformed(T), crops.c2w[:1], crops.K[:1], hw)
            composite = _to_uint8(0.85 * color[0] + (1 - 0.85 * alpha[0]) * background).copy()
            cv2.drawContours(composite, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.drawContours(composite, contours, -1, PALETTE[i % len(PALETTE)], 2, cv2.LINE_AA)
            _write(d / f"overlay_{stage}.jpg", composite)

        T = placed.transform.cpu()
        stats["objects"].append({
            "name": placed.name,
            "color": "#%02x%02x%02x" % PALETTE[i % len(PALETTE)],
            "views": len(obj.views),
            "crop": {"fov": round(view.fov, 1), "theta": round(view.theta, 1), "phi": round(view.phi, 1)},
            "visible_pixels": int(view.mask.sum()),
            "size": [round(float(x), 3) for x in T[:3, :3].norm(dim=0)],
            "position": [round(float(x), 3) for x in T[:3, 3]],
            "yaw_deg": round(math.degrees(math.atan2(float(T[1, 0]), float(T[0, 0]))), 1),
            "refine_steps": placed.refine_steps,
            "stages": [{"name": name, "chamfer": round(cd, 5)} for name, _, cd in placed.stages],
        })

    # All objects in the panorama after every stage, then background and full scene.
    for stage in stage_names:
        parts = [p.asset.gaussians.transformed(dict((n, T) for n, T, _ in p.stages)[stage]) for p in result.objects]
        _write(out / f"objects_{stage}.jpg", _to_uint8(render_panorama(Gaussians.concat(parts), pano_c2w)))
    if result.background is not None:
        _write(out / "background.jpg", _to_uint8(render_panorama(result.background, pano_c2w)))
        scene_gs = Gaussians.concat([p.gaussians for p in result.objects] + [result.background])
        _write(out / "scene.jpg", _to_uint8(render_panorama(scene_gs, pano_c2w)))
    stats["stages"] = stage_names
    (out / "process.json").write_text(json.dumps(stats, indent=2))
