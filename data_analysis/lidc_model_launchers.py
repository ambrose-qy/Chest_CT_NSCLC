"""
Thin launchers for per-model LIDC-IDRI training scripts.

All model-specific scripts use these helpers so preprocessing, augmentation,
logging, early stopping, scheduling, and evaluation stay identical.
"""

from __future__ import print_function

import argparse
from pathlib import Path

from lidc_2d_slices import DEFAULT_OUTPUT_TABLE, build_slice_manifest
from lidc_lightning_train_utils import add_common_training_args, run_lightning_training
from lidc_lightning_utils import EXPERIMENT_DIR, log


TABLE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "tables"
DEFAULT_3D_MANIFEST = TABLE_DIR / "lidc_roi_3d_volume_manifest.csv"


def apply_default_hparams(parser, default_hparams):
    if default_hparams:
        parser.set_defaults(**default_hparams)


def parse_2d_args(model_name, argv=None, output_name=None, default_hparams=None):
    parser = argparse.ArgumentParser(description="Train LIDC-IDRI 2D {} Lightning baseline.".format(model_name))
    add_common_training_args(parser, default_output_dir=EXPERIMENT_DIR / "2d" / (output_name or model_name))
    parser.set_defaults(model=model_name, task="binary", batch_size=16, monitor="val_auc_roc")
    parser.add_argument("--slice-manifest", type=Path, default=DEFAULT_OUTPUT_TABLE, help="2D max-slice manifest.")
    parser.add_argument("--force-build-slice-manifest", action="store_true")
    parser.add_argument("--skip-build-slice-manifest", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained torchvision weights.")
    apply_default_hparams(parser, default_hparams)
    return parser.parse_args(argv)


def run_2d_model(model_name, argv=None, output_name=None, default_hparams=None):
    args = parse_2d_args(model_name, argv=argv, output_name=output_name, default_hparams=default_hparams)
    if args.task != "binary":
        raise ValueError("2D max-slice manifest currently supports binary task only.")
    if args.force_build_slice_manifest or (not args.slice_manifest.exists() and not args.skip_build_slice_manifest):
        log("Building 2D maximum cross-section slice manifest: {}".format(args.slice_manifest))
        build_slice_manifest(output_path=args.slice_manifest)
    if not args.slice_manifest.exists():
        raise FileNotFoundError("Missing 2D slice manifest: {}".format(args.slice_manifest))
    return run_lightning_training(args, input_dim="2d", manifest_path=args.slice_manifest)


def parse_3d_args(model_name, argv=None, output_name=None, default_hparams=None):
    parser = argparse.ArgumentParser(description="Train LIDC-IDRI 3D {} Lightning baseline.".format(model_name))
    add_common_training_args(parser, default_output_dir=EXPERIMENT_DIR / "3d" / (output_name or model_name))
    parser.set_defaults(model=model_name, task="binary", batch_size=4, monitor="val_auc_roc")
    parser.add_argument("--volume-manifest", type=Path, default=DEFAULT_3D_MANIFEST, help="Process5 3D volume manifest.")
    apply_default_hparams(parser, default_hparams)
    return parser.parse_args(argv)


def run_3d_model(model_name, argv=None, output_name=None, default_hparams=None):
    args = parse_3d_args(model_name, argv=argv, output_name=output_name, default_hparams=default_hparams)
    if not args.volume_manifest.exists():
        raise FileNotFoundError(
            "Missing 3D volume manifest: {}. Run data_analysis/LIDC_process5.py first.".format(args.volume_manifest)
        )
    return run_lightning_training(args, input_dim="3d", manifest_path=args.volume_manifest)
