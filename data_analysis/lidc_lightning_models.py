"""
2D and 3D model factories for LIDC-IDRI Lightning baselines.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from lidc_attention import make_attention
from lidc_2d_models import create_lidc_model


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, attention="none"):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.attention = make_attention(attention, out_channels, dim=3)
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
        out = self.attention(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class ResNet3D(nn.Module):
    def __init__(
        self,
        layers=(2, 2, 2),
        channels=(32, 64, 128),
        in_channels=1,
        num_classes=2,
        dropout=0.2,
        attention="none",
        multiscale=False,
    ):
        super().__init__()
        self.multiscale = bool(multiscale)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
        )
        current = channels[0]
        blocks = []
        for stage_idx, (num_blocks, out_channels) in enumerate(zip(layers, channels)):
            stride = 1 if stage_idx == 0 else 2
            stage = [BasicBlock3D(current, out_channels, stride=stride, attention=attention)]
            current = out_channels
            for _ in range(1, num_blocks):
                stage.append(BasicBlock3D(current, out_channels, stride=1, attention=attention))
            blocks.append(nn.Sequential(*stage))
        self.layers = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        classifier_in = sum(channels) if self.multiscale else current
        self.fc = nn.Linear(classifier_in, num_classes)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for layer in self.layers:
            x = layer(x)
            if self.multiscale:
                features.append(self.pool(x).flatten(1))
        if self.multiscale:
            x = torch.cat(features, dim=1)
        else:
            x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class VNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeats=2, attention="none"):
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
        self.attention = make_attention(attention, out_channels, dim=3)

    def forward(self, x):
        return self.attention(self.block(x)) + self.residual(x)


class VNetClassifier(nn.Module):
    """
    VNet-inspired 3D encoder used as a classifier baseline.
    """

    def __init__(self, in_channels=1, num_classes=2, base_channels=16, dropout=0.25, attention="none"):
        super().__init__()
        c = base_channels
        self.enc1 = VNetBlock(in_channels, c, repeats=2, attention=attention)
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=2, stride=2)
        self.enc2 = VNetBlock(c * 2, c * 2, repeats=2, attention=attention)
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=2, stride=2)
        self.enc3 = VNetBlock(c * 4, c * 4, repeats=3, attention=attention)
        self.down3 = nn.Conv3d(c * 4, c * 8, kernel_size=2, stride=2)
        self.enc4 = VNetBlock(c * 8, c * 8, repeats=3, attention=attention)
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


class AttentionWrapped2D(nn.Module):
    def __init__(self, model, model_name, attention="none"):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.attention_name = str(attention or "none").lower()
        channels = None
        if self.attention_name not in ("", "none", "identity"):
            if model_name.startswith("resnet"):
                channels = model.fc[1].in_features if isinstance(model.fc, nn.Sequential) else model.fc.in_features
            elif model_name.startswith("densenet"):
                channels = model.classifier[1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
        self.attention = make_attention(attention, channels, dim=2) if channels else nn.Identity()

    def forward(self, x):
        if self.model_name.startswith("resnet"):
            m = self.model
            x = m.conv1(x)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)
            x = m.layer1(x)
            x = m.layer2(x)
            x = m.layer3(x)
            x = m.layer4(x)
            x = self.attention(x)
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return m.fc(x)

        if self.model_name.startswith("densenet"):
            features = self.model.features(x)
            features = F.relu(features, inplace=True)
            features = self.attention(features)
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            return self.model.classifier(pooled)

        return self.model(x)


class MultiViewFusion3D(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2, attention="none"):
        super().__init__()
        self.axial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            make_attention(attention, 32, dim=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            make_attention(attention, 64, dim=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _view_features(self, volume):
        _, _, d, h, w = volume.shape
        axial = volume[:, :, d // 2, :, :]
        coronal = volume[:, :, :, h // 2, :]
        sagittal = volume[:, :, :, :, w // 2]
        return self.axial(axial), self.axial(coronal), self.axial(sagittal)

    def forward(self, x):
        return self.classifier(torch.cat(self._view_features(x), dim=1))


def create_lidc_lightning_model(
    model_name,
    input_dim,
    num_classes=2,
    pretrained=False,
    in_channels=None,
    dropout=0.2,
    attention="none",
    fusion="none",
):
    model_name = model_name.lower()
    fusion = str(fusion or "none").lower()
    if input_dim == "2d":
        model = create_lidc_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=3 if in_channels is None else in_channels,
        )
        if model_name.startswith("resnet") and hasattr(model, "fc"):
            in_features = model.fc.in_features if hasattr(model.fc, "in_features") else model.fc[-1].in_features
            model.fc = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(in_features, num_classes))
        elif model_name.startswith("densenet") and hasattr(model, "classifier"):
            in_features = model.classifier.in_features if hasattr(model.classifier, "in_features") else model.classifier[-1].in_features
            model.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(in_features, num_classes))
        return AttentionWrapped2D(model, model_name, attention=attention)

    if input_dim != "3d":
        raise ValueError("input_dim must be 2d or 3d.")

    if fusion in ("multiview", "multi_view") or model_name in ("multiview3d", "multi_view3d"):
        return MultiViewFusion3D(num_classes=num_classes, dropout=dropout, attention=attention)
    multiscale = fusion in ("multiscale", "multi_scale") or model_name.endswith("_multiscale")
    base_name = model_name.replace("_multiscale", "")
    if base_name in ("resnet3d", "resnet3d18"):
        return ResNet3D(layers=(2, 2, 2), channels=(32, 64, 128), in_channels=1, num_classes=num_classes, dropout=dropout, attention=attention, multiscale=multiscale)
    if base_name in ("resnet3d34",):
        return ResNet3D(layers=(3, 4, 6), channels=(32, 64, 128), in_channels=1, num_classes=num_classes, dropout=dropout, attention=attention, multiscale=multiscale)
    if model_name in ("vnet", "vnet3d"):
        return VNetClassifier(in_channels=1, num_classes=num_classes, dropout=dropout, attention=attention)

    raise ValueError(
        "Unsupported model '{}'. Use 2D resnet18/resnet34/resnet50/densenet121/densenet169 "
        "or 3D resnet3d/resnet3d34/vnet/multiview3d.".format(model_name)
    )


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
