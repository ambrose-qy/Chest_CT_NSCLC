"""
Run the stage-4 post-hoc code pipeline after model training is complete.

Default steps:
1. Backfill model predictions and ROC curves.
2. Consolidate Grad-CAM failure modes and calcification evidence.
3. Run clinical threshold analysis.
4. Run ensemble prediction analysis.
5. Rebuild 3D ROI QC table from existing NPZ volumes.

Optional:
    --run-luna 1
will also export LUNA external ROIs and run external-domain inference.
"""

from __future__ import print_function

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(name, command, dry_run=False):
    print("\n=== {} ===".format(name))
    print(" ".join('"{}"'.format(part) if " " in str(part) else str(part) for part in command))
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise RuntimeError("{} failed with return code {}".format(name, completed.returncode))
    return completed.returncode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run stage-4 LIDC post-hoc deliverable scripts.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-backfill", type=int, default=0)
    parser.add_argument("--skip-failure-analysis", type=int, default=0)
    parser.add_argument("--skip-clinical", type=int, default=0)
    parser.add_argument("--skip-ensemble", type=int, default=0)
    parser.add_argument("--skip-roi-qc", type=int, default=0)
    parser.add_argument("--run-luna", type=int, default=0, help="Run LUNA external ROI export and inference.")
    parser.add_argument("--enable-grad-cam", type=int, default=0, help="Regenerate Grad-CAM during backfill.")
    parser.add_argument("--force-backfill", type=int, default=0, help="Force re-evaluation of every run.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    python = sys.executable

    if not args.skip_backfill:
        cmd = [
            python,
            str(SCRIPT_DIR / "LIDC_finalize_missing_outputs.py"),
            "--enable-grad-cam",
            str(args.enable_grad_cam),
        ]
        if args.force_backfill:
            cmd.append("--force")
        run_step("Backfill predictions and ROC", cmd, args.dry_run)

    if not args.skip_failure_analysis:
        run_step(
            "Grad-CAM failure mode analysis",
            [python, str(SCRIPT_DIR / "LIDC_failure_mode_analysis.py")],
            args.dry_run,
        )

    if not args.skip_clinical:
        run_step("Clinical value analysis", [python, str(SCRIPT_DIR / "LIDC_clinical_value_analysis.py")], args.dry_run)

    if not args.skip_ensemble:
        run_step("Model ensemble analysis", [python, str(SCRIPT_DIR / "LIDC_ensemble_predictions.py")], args.dry_run)

    if not args.skip_roi_qc:
        run_step("Rebuild 3D ROI QC", [python, str(SCRIPT_DIR / "LIDC_rebuild_3d_roi_qc.py")], args.dry_run)

    if args.run_luna:
        run_step("Export LUNA external ROIs", [python, str(SCRIPT_DIR / "LUNA_export_lidc_external_rois.py")], args.dry_run)
        run_step("Run LUNA external inference", [python, str(SCRIPT_DIR / "LIDC_validate_luna_external.py"), "--all-runs"], args.dry_run)

    print("\nStage-4 post-hoc pipeline complete.")


if __name__ == "__main__":
    main()
