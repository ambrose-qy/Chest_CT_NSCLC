"""
Train the LIDC-IDRI 3D ResNet baseline.

Default architecture: resnet3d. Use --model resnet3d34 for the deeper variant
with the same 3D preprocessing and training strategy.

Edit HYPERPARAMETERS below for ordinary tuning. This file can also be imported
from another Python file:

    from LIDC_3d_resnet import train_3d_resnet
    train_3d_resnet({"lr": 3e-4, "epochs": 20})
"""

from __future__ import print_function

from lidc_model_launchers import run_3d_model


HYPERPARAMETERS = {
    "model": "resnet3d",
    "task": "multiclass",
    "epochs": 30,
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "monitor": "val_auc_roc",
    "early_stop_patience": 30,
    "early_stop_min_delta": 0.0,
    "num_workers": 0,
    "seed": 42,
    "precision": "32-true",
    "normalization_mean": "auto",
    "normalization_std": "auto",
    "normalization_stats_samples": 128,
    "augment_rotate": True,
    "augment_flip": True,
    "augment_scale": True,
    "augment_noise_std": 0.02,
    "augment_intensity_shift": 0.05,
    "augment_contrast_range": 0.10,
    "augment_cutout_fraction": 0.05,
    "class_weight_mode": "balanced",
    "custom_class_weights": None,
    "dropout": 0.2,
    "gradient_clip_val": 1.0,
    "gradient_clip_algorithm": "norm",
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


def train_3d_resnet(overrides=None, argv=()):
    return run_3d_model(
        "resnet3d",
        argv=list(argv),
        output_name="resnet",
        default_hparams=merged_hparams(overrides),
    )


def main(argv=None):
    return run_3d_model(
        "resnet3d",
        argv=argv,
        output_name="resnet",
        default_hparams=HYPERPARAMETERS,
    )


if __name__ == "__main__":
    main()
