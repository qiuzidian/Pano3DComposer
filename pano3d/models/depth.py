"""Depth Anything 3 fine-tuned on equirectangular panoramas.

Two fine-tuned copies are used:

* ``da3_pano_depth``: metric (ray-length) depth of the full panorama, used to give
  object crops a metric depth and to estimate relative panorama poses;
* ``da3_bg_depth``: depth completion of panoramas whose object pixels are blanked
  out; its features also feed the background Gaussian head.

Both replace the DPT head convolutions with circular-padding convolutions.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from .erp_conv import erp_convert

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


@dataclass
class DepthOutput:
    depth: torch.Tensor  # (V, H, W) ray-length depth in meters
    c2w: torch.Tensor  # (V, 4, 4) camera-to-world relative to the first view
    feats: list[torch.Tensor]  # backbone features (1, V, N, C) of the 4 output layers


class DepthAnything3Pano(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY

        self.model = create_object(load_config(MODEL_REGISTRY["da3-large"]))
        self.model.head = erp_convert(self.model.head)

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor], device: torch.device) -> "DepthAnything3Pano":
        with torch.device(device):
            model = cls()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"bad DA3 weights: missing={missing[:5]} unexpected={unexpected[:5]}")
        return model.to(device).eval()  # also moves buffers created outside the device context

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> DepthOutput:
        """``images``: (1, V, 3, H, W), ImageNet-normalized, H and W multiples of 14."""
        from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri

        net = self.model
        H, W = images.shape[-2:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats, _ = net.backbone(
                images.to(torch.bfloat16), cam_token=None, export_feat_layers=[], ref_view_strategy="first"
            )
        with torch.autocast("cuda", enabled=False):
            depth = net.head(feats, H, W, patch_start_idx=0)["depth"]
            c2w, _ = pose_encoding_to_extri_intri(net.cam_dec(feats[-1][1]), (H, W))
        c2w = torch.cat([c2w, c2w.new_tensor([0, 0, 0, 1]).expand(*c2w.shape[:-2], 1, 4)], dim=-2)
        return DepthOutput(depth=depth[0].float(), c2w=c2w[0].float(), feats=[f[0] for f in feats])
