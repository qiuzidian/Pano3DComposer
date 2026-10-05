# Copyright (c) Meta Platforms, Inc. and affiliates.
# Trimmed for Pano3DComposer: only the SE(3) inverse used by the aggregator is kept.
import torch


def closed_form_inverse_se3(se3: torch.Tensor) -> torch.Tensor:
    """Invert a batch of (N, 4, 4) or (N, 3, 4) rigid transforms in closed form."""
    if se3.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"se3 must be of shape (N,4,4) or (N,3,4), got {se3.shape}.")
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]
    R_t = R.transpose(1, 2)
    inverted = torch.eye(4, dtype=R.dtype, device=R.device)[None].repeat(len(R), 1, 1)
    inverted[:, :3, :3] = R_t
    inverted[:, :3, 3:] = -torch.bmm(R_t, T)
    return inverted
