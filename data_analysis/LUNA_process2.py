"""
LUNA16 subset5-9 preprocessing pipeline.

This file reuses the shared preprocessing implementation in `LUNA_process.py`
but points it at the second LUNA raw-data location:

    D:/LUNA/subset5 ... D:/LUNA/subset9

Outputs are written inside this project:

    data/processed/luna_s5_9/

Pipeline scope only:

* de-identification through SHA1 anonymised scan identifiers
* MetaImage `.mhd/.raw` CT volume loading
* resampling to standard voxel spacing, default 1 x 1 x 1 mm
* HU clipping and normalisation to [0, 1]
* threshold-based lung parenchyma segmentation
* annotation coordinate validation after resampling

Run from the repository root:

    conda run -n torch-gpu python data_analysis/LUNA_process2.py

Useful options:

    conda run -n torch-gpu python data_analysis/LUNA_process2.py --manifest-only
    conda run -n torch-gpu python data_analysis/LUNA_process2.py --max-scans 10 --max-qc-png 2
    conda run -n torch-gpu python data_analysis/LUNA_process2.py --save-volumes
"""

from __future__ import print_function

import argparse
from pathlib import Path

from LUNA_process import (
    HU_CLIP_MAX,
    HU_CLIP_MIN,
    LUNG_THRESHOLD_HU,
    PROJECT_ROOT,
    TARGET_SPACING_XYZ,
    clean_value,
    log,
    preprocess_luna,
)


DEFAULT_LUNA_ROOT = Path("D:/LUNA")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "luna_s5_9"
DEFAULT_OUTPUT_PREFIX = "luna_s5_9"
DEFAULT_SUBSETS = [5, 6, 7, 8, 9]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess unzipped LUNA16 subset5-9 CT volumes.")
    parser.add_argument("--luna-root", type=Path, default=DEFAULT_LUNA_ROOT, help="Root containing subset5 ... subset9 and annotations.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Project-local output directory.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Prefix for generated preprocessing tables.")
    parser.add_argument("--subsets", type=int, nargs="+", default=DEFAULT_SUBSETS, help="Subset numbers to process.")
    parser.add_argument("--max-scans", type=int, default=None, help="Optional limit for runtime testing.")
    parser.add_argument("--target-spacing", type=float, nargs=3, default=list(TARGET_SPACING_XYZ), metavar=("X", "Y", "Z"))
    parser.add_argument("--hu-min", type=int, default=HU_CLIP_MIN, help="HU lower clipping bound.")
    parser.add_argument("--hu-max", type=int, default=HU_CLIP_MAX, help="HU upper clipping bound.")
    parser.add_argument("--lung-threshold", type=int, default=LUNG_THRESHOLD_HU, help="HU threshold for lung air segmentation.")
    parser.add_argument("--save-volumes", action="store_true", help="Save normalised image and lung mask NPZ files. Off by default to protect disk space.")
    parser.add_argument("--max-qc-png", type=int, default=5, help="Maximum QC PNG figures to save if matplotlib is installed.")
    parser.add_argument("--manifest-only", action="store_true", help="Only write preprocessing manifest/de-id map; do not read volume data.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log("LUNA subset5-9 root: {}".format(args.luna_root))
    log("Output dir: {}".format(args.output_dir))

    rows, coord_rows = preprocess_luna(args)
    if args.manifest_only:
        return

    success = [row for row in rows if not clean_value(row.get("error"))]
    log("")
    log("LUNA subset5-9 preprocessing complete")
    log("  attempted scans: {}".format(len(rows)))
    log("  successful scans: {}".format(len(success)))
    log("  annotation coordinates checked: {}".format(len(coord_rows)))
    log("  outputs: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
