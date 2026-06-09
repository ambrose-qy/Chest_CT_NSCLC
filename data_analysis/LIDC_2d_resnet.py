"""
Train the LIDC-IDRI 2D ResNet baseline.

Default architecture: resnet18. Use --model resnet34 or --model resnet50 to
train deeper ResNet variants with the same preprocessing and training strategy.

Edit HYPERPARAMETERS below for ordinary tuning. This file can also be imported
from another Python file:

    from LIDC_2d_resnet import train_2d_resnet
    train_2d_resnet({"lr": 3e-4, "epochs": 20})
"""

from __future__ import print_function

from lidc_model_launchers import run_2d_model


HYPERPARAMETERS = {
    "model": "resnet18",
    "task": "multiclass", # binary or multiclass
    "epochs": 20,
    "batch_size": 16,
    "image_size": 224,
    "lr": 5e-5,
    "weight_decay": 3e-4,
    "scheduler": "cosine",
    "monitor": "val_auc_roc",
    "early_stop_patience": 20,
    "early_stop_min_delta": 0.0,
    "num_workers": 0,
    "seed": 42,
    "precision": "32-true",
    "pretrained": False,
    "normalization_mean": "auto",
    "normalization_std": "auto",
    "normalization_stats_samples": 128,
    "augment_rotate": True,
    "augment_flip": True,
    "augment_scale": True,
    "augment_noise_std": 0.01,
    "augment_intensity_shift": 0.03,
    "augment_contrast_range": 0.05,
    "augment_cutout_fraction": 0.0,
    "class_weight_mode": "sqrt_balanced",
    "custom_class_weights": None,
    "dropout": 0.3,
    "gradient_clip_val": 0.5,
    "gradient_clip_algorithm": "norm",
    "enable_grad_cam": True,
    "grad_cam_max_samples": 12,
    "max_samples_per_split": None,
    "no_class_weights": False,
    "deterministic": False,
    "log_every_n_steps": 10,
}


def merged_hparams(overrides=None):
    params = dict(HYPERPARAMETERS)
    if overrides:
        params.update(overrides)
    return params


def train_2d_resnet(overrides=None, argv=()):
    return run_2d_model(
        "resnet18",
        argv=list(argv),
        output_name="resnet",
        default_hparams=merged_hparams(overrides),
    )


def main(argv=None):
    return run_2d_model(
        "resnet18",
        argv=argv,
        output_name="resnet",
        default_hparams=HYPERPARAMETERS,
    )


if __name__ == "__main__":
    main()
