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
    "lr": 8e-5,
    "weight_decay": 3e-4,
    "scheduler": "plateau",
    "monitor": "val_f1",
    "early_stop_patience": 5,
    "early_stop_min_delta": 0.001,
    "metric_guard_enabled": 1,
    "metric_guard_warmup_epochs": 6,
    "metric_guard_patience": 3,
    "metric_guard_min_val_auc_roc": 0.53,
    "metric_guard_min_val_f1": 0.12,
    "metric_guard_min_val_accuracy": 0.0,
    "metric_guard_max_val_loss": 2.0,
    "num_workers": 0,
    "seed": 42,
    "precision": "32-true",
    "matmul_precision": "medium",
    "pretrained": False,
    "normalization_mean": "auto",
    "normalization_std": "auto",
    "normalization_stats_samples": 128,
    "augment_rotate": True,
    "augment_flip": True,
    "augment_scale": True,
    "augment_scale_range": 0.03,
    "augment_noise_std": 0.003,
    "augment_intensity_shift": 0.01,
    "augment_contrast_range": 0.02,
    "augment_cutout_fraction": 0.0,
    "class_weight_mode": "fixed_multiclass",
    "custom_class_weights": None,
    "label_smoothing": 0.03,
    "dropout": 0.20,
    "attention": "cbam",
    "fusion": "none",
    "gradient_clip_val": 4.0,
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
