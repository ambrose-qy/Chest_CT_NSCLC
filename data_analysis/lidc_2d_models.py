"""
Model factory for LIDC-IDRI 2D CNN baselines.
"""

from __future__ import print_function

import torch
import torch.nn as nn


def _torchvision_weights(model_name, pretrained):
    if not pretrained:
        return None

    try:
        from torchvision import models

        mapping = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "densenet121": models.DenseNet121_Weights.DEFAULT,
            "densenet169": models.DenseNet169_Weights.DEFAULT,
        }
        return mapping[model_name]
    except Exception:
        return "legacy_pretrained"


def _adapt_first_conv(conv, in_channels):
    if in_channels == conv.in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels == 1:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        else:
            repeated = conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            new_conv.weight.copy_(repeated)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def create_lidc_model(model_name="resnet18", num_classes=2, pretrained=False, in_channels=3):
    try:
        from torchvision import models
    except Exception as exc:
        raise ImportError("torchvision is required for ResNet/DenseNet baselines.") from exc

    model_name = model_name.lower()
    weights = _torchvision_weights(model_name, pretrained)
    kwargs = {}
    if weights == "legacy_pretrained":
        kwargs["pretrained"] = True
    else:
        kwargs["weights"] = weights

    if model_name in ("resnet18", "resnet34", "resnet50"):
        model = getattr(models, model_name)(**kwargs)
        model.conv1 = _adapt_first_conv(model.conv1, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name in ("densenet121", "densenet169"):
        model = getattr(models, model_name)(**kwargs)
        model.features.conv0 = _adapt_first_conv(model.features.conv0, in_channels)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError("Unsupported model '{}'. Use resnet18/resnet34/resnet50/densenet121/densenet169.".format(model_name))
