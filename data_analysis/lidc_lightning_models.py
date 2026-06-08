"""
2D and 3D model factories for LIDC-IDRI Lightning baselines.
"""

from __future__ import print_function

import torch
import torch.nn as nn

from lidc_2d_models import create_lidc_model


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class ResNet3D(nn.Module):
    def __init__(self, layers=(2, 2, 2), channels=(32, 64, 128), in_channels=1, num_classes=2, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
        )
        current = channels[0]
        blocks = []
        for stage_idx, (num_blocks, out_channels) in enumerate(zip(layers, channels)):
            stride = 1 if stage_idx == 0 else 2
            stage = [BasicBlock3D(current, out_channels, stride=stride)]
            current = out_channels
            for _ in range(1, num_blocks):
                stage.append(BasicBlock3D(current, out_channels, stride=1))
            blocks.append(nn.Sequential(*stage))
        self.layers = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(current, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class VNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeats=2):
        super().__init__()
        layers = []
        current = in_channels
        for _ in range(repeats):
            layers.extend([
                nn.Conv3d(current, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.PReLU(out_channels),
            ])
            current = out_channels
        self.block = nn.Sequential(*layers)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)


class VNetClassifier(nn.Module):
    """
    VNet-inspired 3D encoder used as a classifier baseline.
    """

    def __init__(self, in_channels=1, num_classes=2, base_channels=16, dropout=0.25):
        super().__init__()
        c = base_channels
        self.enc1 = VNetBlock(in_channels, c, repeats=2)
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=2, stride=2)
        self.enc2 = VNetBlock(c * 2, c * 2, repeats=2)
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=2, stride=2)
        self.enc3 = VNetBlock(c * 4, c * 4, repeats=3)
        self.down3 = nn.Conv3d(c * 4, c * 8, kernel_size=2, stride=2)
        self.enc4 = VNetBlock(c * 8, c * 8, repeats=3)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * 8, num_classes),
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(self.down1(x))
        x = self.enc3(self.down2(x))
        x = self.enc4(self.down3(x))
        x = self.pool(x)
        return self.classifier(x)


def create_lidc_lightning_model(model_name, input_dim, num_classes=2, pretrained=False, in_channels=None):
    model_name = model_name.lower()
    if input_dim == "2d":
        return create_lidc_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=3 if in_channels is None else in_channels,
        )

    if input_dim != "3d":
        raise ValueError("input_dim must be 2d or 3d.")

    if model_name in ("resnet3d", "resnet3d18"):
        return ResNet3D(layers=(2, 2, 2), channels=(32, 64, 128), in_channels=1, num_classes=num_classes)
    if model_name in ("resnet3d34",):
        return ResNet3D(layers=(3, 4, 6), channels=(32, 64, 128), in_channels=1, num_classes=num_classes)
    if model_name in ("vnet", "vnet3d"):
        return VNetClassifier(in_channels=1, num_classes=num_classes)

    raise ValueError(
        "Unsupported model '{}'. Use 2D resnet18/resnet34/resnet50/densenet121/densenet169 "
        "or 3D resnet3d/resnet3d34/vnet.".format(model_name)
    )


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

