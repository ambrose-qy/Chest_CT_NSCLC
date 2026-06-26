"""3D CBAM candidate false-positive reduction model for LUNA16."""

from __future__ import print_function

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lidc_full_ct_utils import fixed_crop_or_pad, normalise_hu
from lidc_attention import CBAMModule


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class LUNA16CandidateDetector(nn.Module):
    def __init__(
        self,
        base_channels=8,
        dropout=0.25,
        attention="cbam",
        spatial_kernel=3,
    ):
        super().__init__()
        channels = int(base_channels)
        self.features = nn.Sequential(
            ConvBlock3D(1, channels),
            ConvBlock3D(channels, channels * 2),
            ConvBlock3D(channels * 2, channels * 4),
            (
                CBAMModule(
                    channels * 4,
                    dim=3,
                    spatial_kernel=int(spatial_kernel),
                )
                if str(attention).lower() == "cbam"
                else nn.Identity()
            ),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(channels * 4, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_candidate_detector(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    config = payload.get("config", {})
    model = LUNA16CandidateDetector(
        base_channels=int(config.get("base_channels", 8)),
        dropout=float(config.get("dropout", 0.25)),
        attention=config.get("attention", "cbam"),
        spatial_kernel=int(config.get("spatial_kernel", 3)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def candidate_probabilities(model, patches, device, batch_size=32):
    if len(patches) == 0:
        return np.zeros((0,), dtype=np.float32)
    outputs = []
    for start in range(0, len(patches), int(batch_size)):
        batch = np.stack(patches[start:start + int(batch_size)], axis=0)
        tensor = torch.from_numpy(batch).float().unsqueeze(1).to(device)
        probabilities = torch.softmax(model(tensor), dim=1)[:, 1]
        outputs.append(probabilities.cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def candidate_probabilities_from_volume(
    model,
    volume_hu,
    candidates,
    device,
    batch_size=64,
    patch_shape=(32, 32, 32),
):
    if len(candidates) == 0:
        return np.zeros((0,), dtype=np.float32)
    outputs = []
    for start in range(0, len(candidates), int(batch_size)):
        batch_candidates = candidates[start:start + int(batch_size)]
        batch = np.stack(
            [
                normalise_hu(
                    fixed_crop_or_pad(
                        volume_hu,
                        candidate["center_zyx"],
                        patch_shape,
                        pad_value=-1000.0,
                    )
                )
                for candidate in batch_candidates
            ],
            axis=0,
        )
        tensor = torch.from_numpy(batch).float().unsqueeze(1).to(device)
        probabilities = torch.softmax(model(tensor), dim=1)[:, 1]
        outputs.append(probabilities.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def write_detector_config(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
