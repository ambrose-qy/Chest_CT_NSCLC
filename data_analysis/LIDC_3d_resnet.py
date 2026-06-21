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
    "task": "multiclass", # binary or multiclass
    "epochs": 25,
    "batch_size": 4,
    "lr": 9e-5,
    "weight_decay": 3e-4,
    "scheduler": "plateau",
    "monitor": "val_f1",
    "early_stop_patience": 6,
    "early_stop_min_delta": 0.002,
    "metric_guard_enabled": 1,
    "metric_guard_warmup_epochs": 6,
    "metric_guard_patience": 3,
    "metric_guard_min_val_auc_roc": 0.55,
    "metric_guard_min_val_f1": 0.12,
    "metric_guard_min_val_accuracy": 0.0,
    "metric_guard_max_val_loss": 2.0,
    "num_workers": 0,
    "seed": 42,
    "precision": "32-true",
    "matmul_precision": "medium",
    "normalization_mean": "auto",
    "normalization_std": "auto",
    "normalization_stats_samples": 128,
    "augment_rotate": True,
    "augment_flip": True,
    "augment_scale": True,
    "augment_scale_range": 0.05,
    "augment_noise_std": 0.010,
    "augment_intensity_shift": 0.02,
    "augment_contrast_range": 0.04,
    "augment_cutout_fraction": 0.0,
    "class_weight_mode": "custom",
    "custom_class_weights": "1.35,0.70,1.55",
    "balanced_sampler": 1,
    "label_smoothing": 0.01,
    "dropout": 0.22,
    "pca_features_enabled": 0,
    "pca_n_components": 0,
    "pca_hidden_dim": 16,
    "attention": "cbam",
    "fusion": "multiscale",
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
