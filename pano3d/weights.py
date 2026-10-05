"""All inference weights are stored in one safetensors file, grouped by key prefix:

    aligner.*     OmniVGGT object-to-world predictor (aggregator + camera/scale head)
    refiner.*     OmniVGGT coarse-to-fine refiner, stored as the tensors that differ from the aligner
    pano_depth.*  Depth Anything 3 fine-tuned for metric panoramic depth
    bg_depth.*    Depth Anything 3 for background depth, stored as the tensors that differ from pano_depth
    gs_head.*     GS-DPT head predicting background Gaussians
    lama.*        LaMa generator fine-tuned for panorama inpainting

The matmul weights of the two transformer backbones are stored in bf16. They only feed
bf16 autocast regions, so this is lossless; everything is upcast to fp32 on load.
"""

from __future__ import annotations

from pathlib import Path

import torch


class Checkpoint:
    def __init__(self, path: str | Path) -> None:
        from safetensors import safe_open

        self.path = Path(path)
        self._file = safe_open(str(path), "pt", device="cpu")

    def _group(self, name: str) -> dict[str, torch.Tensor]:
        prefix = name + "."
        state = {k[len(prefix) :]: self._file.get_tensor(k) for k in self._file.keys() if k.startswith(prefix)}
        if not state:
            raise KeyError(f"{self.path} has no weights for '{name}'")
        return {k: v.float() if v.is_floating_point() else v for k, v in state.items()}

    def state_dict(self, name: str) -> dict[str, torch.Tensor]:
        """Full fp32 state dict of one model (``aligner``, ``refiner``, ``pano_depth``, ...)."""
        base = {"refiner": "aligner", "bg_depth": "pano_depth"}.get(name)
        if base is None:
            return self._group(name)
        return self.state_dict(base) | self._group(name)
