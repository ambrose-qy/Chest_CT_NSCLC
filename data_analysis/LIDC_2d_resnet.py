"""
Train the LIDC-IDRI 2D ResNet baseline.

Default architecture: resnet18. Use --model resnet34 or --model resnet50 to
train deeper ResNet variants with the same preprocessing and training strategy.

Edit ``configs/lidc_training.yaml`` for ordinary tuning. This file can also be
imported from another Python file:

    from LIDC_2d_resnet import train_2d_resnet
    train_2d_resnet({"lr": 3e-4, "epochs": 20})
"""

from __future__ import print_function

from lidc_experiment_config import get_model_hyperparameters
from lidc_model_launchers import run_2d_model


HYPERPARAMETERS = get_model_hyperparameters("2d_resnet")


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
