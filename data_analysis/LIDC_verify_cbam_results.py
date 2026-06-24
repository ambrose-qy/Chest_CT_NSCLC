"""
Audit whether selected report checkpoints actually contain CBAM attention.

The audit checks the report summary, run config, checkpoint hyperparameters,
and CBAM parameter keys. It exits with a non-zero status when any selected
result is not CBAM-compliant unless --allow-incomplete is provided.
"""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "lidc_lightning"
    / "stage4_latest_assets"
    / "stage4_latest_model_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "model_reports" / "lidc_cbam_audit"
)


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def attention_parameter_keys(state_dict):
    keys = []
    for key in state_dict:
        lower = key.lower()
        if ".attention." in lower and (
            ".channel." in lower or ".spatial." in lower
        ):
            keys.append(key)
    return keys


def audit_row(row):
    run_dir = Path(row["run_dir"])
    config_path = run_dir / "config.json"
    checkpoint_path = Path(row.get("selected_checkpoint") or row.get("best_checkpoint", ""))

    config = load_json(config_path) if config_path.exists() else {}
    config_args = config.get("args", {}) if isinstance(config.get("args"), dict) else {}
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    hyperparameters = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint.get("state_dict", {})
    cbam_keys = attention_parameter_keys(state_dict)

    summary_attention = str(row.get("attention", "")).lower()
    config_attention = str(
        config.get("attention") or config_args.get("attention") or ""
    ).lower()
    checkpoint_attention = str(hyperparameters.get("attention") or "").lower()

    checks = {
        "summary_cbam": summary_attention == "cbam",
        "config_cbam": config_attention == "cbam",
        "checkpoint_hparams_cbam": checkpoint_attention == "cbam",
        "checkpoint_has_cbam_weights": bool(cbam_keys),
    }
    return {
        "input_dim": row.get("input_dim", ""),
        "model": row.get("model", ""),
        "task": row.get("task", ""),
        "run_name": row.get("run_name", ""),
        "summary_attention": summary_attention or "missing",
        "config_attention": config_attention or "missing",
        "checkpoint_attention": checkpoint_attention or "missing",
        "cbam_parameter_key_count": len(cbam_keys),
        "summary_cbam": checks["summary_cbam"],
        "config_cbam": checks["config_cbam"],
        "checkpoint_hparams_cbam": checks["checkpoint_hparams_cbam"],
        "checkpoint_has_cbam_weights": checks["checkpoint_has_cbam_weights"],
        "cbam_compliant": all(checks.values()),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit selected LIDC CBAM checkpoints.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = [audit_row(row) for row in read_csv_rows(args.summary)]
    compliant_count = sum(bool(row["cbam_compliant"]) for row in results)
    payload = {
        "selected_result_count": len(results),
        "cbam_compliant_count": compliant_count,
        "noncompliant_count": len(results) - compliant_count,
        "all_selected_results_use_cbam": compliant_count == len(results),
        "noncompliant_runs": [
            row["run_name"] for row in results if not row["cbam_compliant"]
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cbam_checkpoint_audit.csv", results)
    with (args.output_dir / "cbam_checkpoint_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2), flush=True)
    if not payload["all_selected_results_use_cbam"] and not args.allow_incomplete:
        raise SystemExit(1)
    return payload


if __name__ == "__main__":
    main()
