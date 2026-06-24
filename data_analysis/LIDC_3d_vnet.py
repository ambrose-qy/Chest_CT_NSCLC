"""
Train the LIDC-IDRI 3D VNet-style classifier baseline.

Edit ``configs/lidc_training.yaml`` for ordinary tuning. This file can also be
imported from another Python file:

    from LIDC_3d_vnet import train_3d_vnet
    train_3d_vnet({"lr": 3e-4, "epochs": 20})
"""

from __future__ import print_function

from lidc_experiment_config import get_model_hyperparameters
from lidc_model_launchers import run_3d_model


HYPERPARAMETERS = get_model_hyperparameters("3d_vnet")


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
