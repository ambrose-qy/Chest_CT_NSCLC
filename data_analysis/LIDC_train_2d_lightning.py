"""
Train LIDC-IDRI 2D Lightning baselines with maximum cross-sectional slices.

Examples:

    conda run -n torch-gpu python data_analysis/LIDC_train_2d_lightning.py --model resnet18 --epochs 30
    conda run -n torch-gpu python data_analysis/LIDC_train_2d_lightning.py --model densenet121 --pretrained
"""

from __future__ import print_function

import argparse
from pathlib import Path

from lidc_2d_slices import DEFAULT_OUTPUT_TABLE, build_slice_manifest
from lidc_lightning_train_utils import add_common_training_args, run_lightning_training
from lidc_lightning_utils import EXPERIMENT_DIR, log


OUTPUT_DIR = EXPERIMENT_DIR / "2d"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train 2D ResNet/DenseNet Lightning baselines on LIDC-IDRI.")
    add_common_training_args(parser, default_output_dir=OUTPUT_DIR)
    parser.set_defaults(model="resnet18", batch_size=16, monitor="val_auc_roc")
    parser.add_argument("--slice-manifest", type=Path, default=DEFAULT_OUTPUT_TABLE, help="2D max-slice manifest.")
    parser.add_argument("--force-build-slice-manifest", action="store_true")
    parser.add_argument("--skip-build-slice-manifest", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained torchvision weights.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.task != "binary":
        raise ValueError("2D max-slice manifest currently supports binary task only.")

    if args.force_build_slice_manifest or (not args.slice_manifest.exists() and not args.skip_build_slice_manifest):
        log("Building 2D maximum cross-section slice manifest: {}".format(args.slice_manifest))
        build_slice_manifest(output_path=args.slice_manifest)
    if not args.slice_manifest.exists():
        raise FileNotFoundError("Missing 2D slice manifest: {}".format(args.slice_manifest))

    run_lightning_training(args, input_dim="2d", manifest_path=args.slice_manifest)


if __name__ == "__main__":
    main()
