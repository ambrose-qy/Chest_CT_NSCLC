"""
Train the LIDC-IDRI 2D DenseNet baseline.

Default architecture: densenet121. Use --model densenet169 for the larger
variant with the same preprocessing and training strategy.

Edit HYPERPARAMETERS below for ordinary tuning. This file can also be imported
from another Python file:

    from LIDC_2d_densenet import train_2d_densenet
    train_2d_densenet({"lr": 3e-4, "epochs": 20})
"""

from __future__ import print_function

from lidc_model_launchers import run_2d_model


HYPERPARAMETERS = {
    "model": "densenet121",
    "task": "multiclass", # binary or multiclass
    "epochs": 20,
    "batch_size": 16,
    "image_size": 224,
    "lr": 3e-5,
    "weight_decay": 5e-4,
    "scheduler": "cosine",
    "monitor": "val_auc_roc",
    "early_stop_patience": 10,
    "early_stop_min_delta": 0.001,
    "metric_guard_enabled": 1,
    "metric_guard_warmup_epochs": 3,
    "metric_guard_patience": 2,
    "metric_guard_min_val_auc_roc": 0.50,
    "metric_guard_min_val_f1": 0.10,
    "metric_guard_min_val_accuracy": 0.0,
    "metric_guard_max_val_loss": 5.0,
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
    "augment_scale_range": 0.05,
    "augment_noise_std": 0.005,
    "augment_intensity_shift": 0.02,
    "augment_contrast_range": 0.03,
    "augment_cutout_fraction": 0.0,
    "class_weight_mode": "fixed_multiclass",
    "custom_class_weights": None,
    "dropout": 0.3,
    "attention": "cbam",
    "fusion": "none",
    "gradient_clip_val": 8.0,
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
