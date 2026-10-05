"""Make convolutional heads panorama-aware: circular padding along the longitude."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ERPConv2d(nn.Conv2d):
    """Conv2d that zero-pads vertically and wraps around horizontally."""

    def forward(self, x):
        pad_h, pad_w = self.padding
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="constant")
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


def erp_convert(module: nn.Module) -> nn.Module:
    """Recursively replace every ``nn.Conv2d`` by an ``ERPConv2d`` with the same weights."""
    if isinstance(module, nn.Conv2d) and not isinstance(module, ERPConv2d):
        conv = ERPConv2d(
            module.in_channels, module.out_channels, module.kernel_size, stride=module.stride,
            padding=module.padding, dilation=module.dilation, groups=module.groups, bias=module.bias is not None,
        )
        conv.load_state_dict(module.state_dict())
        return conv
    for name, child in module.named_children():
        setattr(module, name, erp_convert(child))
    return module
