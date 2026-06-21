"""
Backfill missing LIDC model evaluation artifacts for completed Lightning runs.

This script does not train models. It reads the weekly model summary, finds each
run directory and best checkpoint, then calls ``LIDC_evaluate_lightning.py`` so
missing ``test_predictions.csv`` and ``roc_curve.csv`` files are written back
into the original run folders.

Run from the project root:

    C:\\Users\\Ambro\\.conda\\envs\\torch-gpu\\python.exe data_analysis\\LIDC_finalize_missing_outputs.py
"""

from __future__ import print_function

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "model_results" / "lidc_lightning" / "weekly_report3_final_assets" / "weekly_report3_model_summary.csv"
EVALUATE_SCRIPT = Path(__file__).resolve().parent / "LIDC_evaluate_lightning.py"
POSTHOC_SCRIPT = Path(__file__).resolve().parent / "LIDC_prediction_analysis.py"


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def has_required_outputs(run_dir, task):
    run_dir = Path(run_dir)
    required = [
        run_dir / "test_metrics.json",
        run_dir / "test_predictions.csv",
        run_dir / "confusion_matrix.csv",
    ]
    if task == "binary":
        required.append(run_dir / "roc_curve.csv")
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def command_for_row(row, args):
    run_dir = Path(row["run_dir"])
    checkpoint = Path(row["best_checkpoint"])
    config = run_dir / "config.json"
    if not checkpoint.exists():
        raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint))
    if not config.exists():
        raise FileNotFoundError("Missing config: {}".format(config))

    return [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config),
        "--output-dir",
        str(run_dir.parent),
        "--enable-grad-cam",
        "1" if args.enable_grad_cam else "0",
        "--grad-cam-max-samples",
        str(args.grad_cam_max_samples),
        "--num-workers",
        str(args.num_workers),
    ]


def run_command(cmd, dry_run=False):
    print(" ".join('"{}"'.format(part) if " " in str(part) else str(part) for part in cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(completed.returncode)


def run_posthoc_reports(args):
    cmd = [
        sys.executable,
        str(POSTHOC_SCRIPT),
        "--search-root",
        str(PROJECT_ROOT / "data" / "processed" / "model_results" / "lidc_lightning"),
        "--output-dir",
        str(args.prediction_analysis_output),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Backfill missing LIDC evaluation artifacts.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="CSV produced for the weekly report model summary.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate runs even when prediction outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--enable-grad-cam", type=int, default=0, help="Regenerate Grad-CAM during backfill. Use 0 when existing Grad-CAM is enough.")
    parser.add_argument("--grad-cam-max-samples", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-posthoc-analysis", type=int, default=1, help="Run subgroup/morphology analysis after backfill.")
    parser.add_argument("--prediction-analysis-output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "model_reports" / "lidc_prediction_analysis")
    parser.add_argument("--audit-output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "model_reports" / "lidc_missing_output_backfill")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = read_csv_rows(args.summary)
    records = []
    failures = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        already_complete = has_required_outputs(run_dir, row.get("task", ""))
        record = {
            "input_dim": row.get("input_dim", ""),
            "model": row.get("model", ""),
            "task": row.get("task", ""),
            "run_name": row.get("run_name", ""),
            "run_dir": str(run_dir),
            "already_complete": int(already_complete),
            "status": "skipped_complete" if already_complete and not args.force else "pending",
            "return_code": "",
        }
        if already_complete and not args.force:
            records.append(record)
            continue
        try:
            cmd = command_for_row(row, args)
            code = run_command(cmd, dry_run=args.dry_run)
            record["return_code"] = code
            record["status"] = "dry_run" if args.dry_run else ("complete" if code == 0 else "failed")
            if code != 0:
                failures.append(record)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            failures.append(record)
        records.append(record)

    if args.run_posthoc_analysis and not failures:
        code = run_posthoc_reports(args)
        records.append({
            "input_dim": "",
            "model": "posthoc_prediction_analysis",
            "task": "",
            "run_name": "LIDC_prediction_analysis",
            "run_dir": str(args.prediction_analysis_output),
            "already_complete": "",
            "status": "dry_run" if args.dry_run else ("complete" if code == 0 else "failed"),
            "return_code": code,
        })

    args.audit_output.mkdir(parents=True, exist_ok=True)
    write_csv(args.audit_output / "backfill_records.csv", records)
    write_json(args.audit_output / "backfill_summary.json", {
        "summary_csv": str(args.summary),
        "records_csv": str(args.audit_output / "backfill_records.csv"),
        "run_count": len(rows),
        "failure_count": len(failures),
        "dry_run": bool(args.dry_run),
    })
    if failures:
        print("Backfill finished with {} failure(s). See {}".format(len(failures), args.audit_output / "backfill_records.csv"))
        return 1
    print("Backfill complete. Records: {}".format(args.audit_output / "backfill_records.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
