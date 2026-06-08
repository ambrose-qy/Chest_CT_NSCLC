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
    "task": "binary",
    "epochs": 50,
    "batch_size": 16,
    "image_size": 224,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "monitor": "val_auc_roc",
    "early_stop_patience": 8,
    "early_stop_min_delta": 0.0,
    "num_workers": 0,
    "seed": 2026,
    "precision": "32-true",
    "pretrained": False,
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
