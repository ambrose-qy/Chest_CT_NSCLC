"""
Train the LIDC-IDRI 2D DenseNet baseline.

Default architecture: densenet121. Use --model densenet169 for the larger
variant with the same preprocessing and training strategy.

Edit ``configs/lidc_training.yaml`` for ordinary tuning. This file can also be
imported from another Python file:

    from LIDC_2d_densenet import train_2d_densenet
    train_2d_densenet({"lr": 3e-4, "epochs": 20})
"""

from __future__ import print_function

from lidc_experiment_config import get_model_hyperparameters
from lidc_model_launchers import run_2d_model


HYPERPARAMETERS = get_model_hyperparameters("2d_densenet")


def merged_hparams(overrides=None):
    params = dict(HYPERPARAMETERS)
    if overrides:
        params.update(overrides)
    return params


def train_2d_densenet(overrides=None, argv=()):
    return run_2d_model(
        "densenet121",
        argv=list(argv),
        output_name="densenet",
        default_hparams=merged_hparams(overrides),
    )


def main(argv=None):
    return run_2d_model(
        "densenet121",
        argv=argv,
        output_name="densenet",
        default_hparams=HYPERPARAMETERS,
    )


if __name__ == "__main__":
    main()
