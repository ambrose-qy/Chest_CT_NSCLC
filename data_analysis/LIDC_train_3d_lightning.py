"""
Train LIDC-IDRI 3D Lightning baselines with complete standardised ROI volumes.

Examples:

    conda run -n torch-gpu python data_analysis/LIDC_train_3d_lightning.py --model resnet3d --epochs 40
    conda run -n torch-gpu python data_analysis/LIDC_train_3d_lightning.py --model vnet --batch-size 4
"""

from __future__ import print_function

import argparse
from pathlib import Path

from lidc_lightning_train_utils import add_common_training_args, run_lightning_training
from lidc_lightning_utils import EXPERIMENT_DIR


TABLE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "tables"
DEFAULT_3D_MANIFEST = TABLE_DIR / "lidc_roi_3d_volume_manifest.csv"
OUTPUT_DIR = EXPERIMENT_DIR / "3d"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train 3D ResNet/VNet Lightning baselines on LIDC-IDRI ROI volumes.")
    add_common_training_args(parser, default_output_dir=OUTPUT_DIR)
    parser.set_defaults(model="resnet3d", batch_size=4, monitor="val_auc_roc")
    parser.add_argument("--volume-manifest", type=Path, default=DEFAULT_3D_MANIFEST, help="Process5 3D volume manifest.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.volume_manifest.exists():
        raise FileNotFoundError(
            "Missing 3D volume manifest: {}. Run data_analysis/LIDC_process5.py first.".format(args.volume_manifest)
        )
    run_lightning_training(args, input_dim="3d", manifest_path=args.volume_manifest)


if __name__ == "__main__":
    main()
