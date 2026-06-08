"""
Train the LIDC-IDRI 3D VNet-style classifier baseline.

Edit HYPERPARAMETERS below for ordinary tuning. This file can also be imported
from another Python file:

    from LIDC_3d_vnet import train_3d_vnet
    train_3d_vnet({"lr": 3e-4, "epochs": 20})
"""

from __future__ import print_function

from lidc_model_launchers import run_3d_model


HYPERPARAMETERS = {
    "model": "vnet",
    "task": "binary",
    "epochs": 60,
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "monitor": "val_auc_roc",
    "early_stop_patience": 8,
    "early_stop_min_delta": 0.0,
    "num_workers": 0,
    "seed": 2026,
    "precision": "32-true",
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


def train_3d_vnet(overrides=None, argv=()):
    return run_3d_model(
        "vnet",
        argv=list(argv),
        output_name="vnet",
        default_hparams=merged_hparams(overrides),
    )


def main(argv=None):
    return run_3d_model(
        "vnet",
        argv=argv,
        output_name="vnet",
        default_hparams=HYPERPARAMETERS,
    )


if __name__ == "__main__":
    main()
