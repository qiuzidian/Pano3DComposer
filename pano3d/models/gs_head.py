"""GS-DPT head predicting per-pixel background Gaussians from Depth Anything 3 features."""

from __future__ import annotations

import torch
import torch.nn as nn
from depth_anything_3.model.gsdpt import GSDPT as _GSDPT

from .erp_conv import erp_convert

SH_DEGREE = 2
# opacity + scale + rotation + SH colors + confidence
GS_OUTPUT_DIM = 1 + 3 + 4 + 3 * (SH_DEGREE + 1) ** 2 + 1


class GSDPT(_GSDPT):
    """DA3's GS-DPT whose image branch also takes a mask channel (RGB + mask)."""

    def __init__(self, dim_image: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        first = self.images_merger[0]
        self.images_merger[0] = nn.Conv2d(dim_image, first.out_channels, 3, 1, 1)

    def forward(self, feats: list[torch.Tensor], H: int, W: int, patch_start_idx: int, images: torch.Tensor):
        """``feats``: list of (B, S, N, C) tensors; ``images``: (B, S, 4, H, W)."""
        return super().forward([(f,) for f in feats], H, W, patch_start_idx, images=images)


def build_gs_head(state: dict[str, torch.Tensor], device: torch.device) -> GSDPT:
    with torch.device(device):
        head = GSDPT(
            dim_in=2048, patch_size=14, output_dim=GS_OUTPUT_DIM, activation="linear", conf_activation="expp1", features=256
        )
        head = erp_convert(head)
    head.load_state_dict(state, strict=True)
    return head.to(device).eval()
