"""
Attention modules for 2D and 3D LIDC-IDRI models.

CBAM is the preferred option because it combines channel attention with
spatial attention. SE is kept as a lightweight comparison baseline.
"""

from __future__ import print_function

import torch
import torch.nn as nn


def _conv(dim):
    return nn.Conv3d if int(dim) == 3 else nn.Conv2d


def _adaptive_avg_pool(dim):
    return nn.AdaptiveAvgPool3d if int(dim) == 3 else nn.AdaptiveAvgPool2d


def _adaptive_max_pool(dim):
    return nn.AdaptiveMaxPool3d if int(dim) == 3 else nn.AdaptiveMaxPool2d


class SEModule(nn.Module):
    def __init__(self, channels, dim=2, reduction=16):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 4)
        conv = _conv(dim)
        pool = _adaptive_avg_pool(dim)
        self.net = nn.Sequential(
            pool(1),
            conv(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            conv(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, dim=2, reduction=16):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 4)
        conv = _conv(dim)
        self.avg_pool = _adaptive_avg_pool(dim)(1)
        self.max_pool = _adaptive_max_pool(dim)(1)
        self.mlp = nn.Sequential(
            conv(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            conv(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, dim=2, kernel_size=7):
        super().__init__()
        padding = int(kernel_size) // 2
        conv = _conv(dim)
        self.conv = conv(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maximum, _ = torch.max(x, dim=1, keepdim=True)
        weights = self.sigmoid(self.conv(torch.cat([avg, maximum], dim=1)))
        return x * weights


class CBAMModule(nn.Module):
    def __init__(self, channels, dim=2, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention(channels, dim=dim, reduction=reduction)
        self.spatial = SpatialAttention(dim=dim, kernel_size=spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


def make_attention(attention, channels, dim):
    name = str(attention or "none").lower()
    if name in ("", "none", "identity"):
        return nn.Identity()
    if name == "se":
        return SEModule(channels, dim=dim)
    if name == "cbam":
        return CBAMModule(channels, dim=dim)
    raise ValueError("Unsupported attention '{}'. Use none, se, or cbam.".format(attention))
