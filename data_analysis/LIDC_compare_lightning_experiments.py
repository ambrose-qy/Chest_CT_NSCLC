"""
Aggregate LIDC-IDRI Lightning experiment metrics into a comparison report.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path

from lidc_lightning_utils import EXPERIMENT_DIR, write_csv, write_json


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_reports" / "lidc_lightning"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_metrics(prefix, metrics):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)):
            out["{}_{}".format(prefix, key)] = value
    return out


def collect_runs(root):
    rows = []
    for config_path in sorted(Path(root).rglob("config.json")):
        run_dir = config_path.parent
        config = read_json(config_path)
        best_path = run_dir / "best_config.json"
        test_path = run_dir / "test_metrics.json"
        row = {
            "run_name": config.get("run_name", run_dir.name),
            "input_dim": config.get("input_dim", ""),
            "model": config.get("args", {}).get("model", ""),
            "task": config.get("args", {}).get("task", ""),
            "lr": config.get("args", {}).get("lr", ""),
            "weight_decay": config.get("args", {}).get("weight_decay", ""),
            "batch_size": config.get("args", {}).get("batch_size", ""),
            "scheduler": config.get("args", {}).get("scheduler", ""),
            "parameter_count": config.get("parameter_count", ""),
            "config_path": str(config_path),
        }
        if best_path.exists():
            row.update(flatten_metrics("best", read_json(best_path)))
        if test_path.exists():
            row.update(flatten_metrics("test", read_json(test_path)))
        rows.append(row)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect LIDC Lightning model comparison metrics.")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = collect_runs(args.experiment_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "lidc_lightning_model_comparison.csv"
    write_csv(comparison_path, rows)
    best = None
    for row in rows:
        score = row.get("test_auc_roc") or row.get("test_f1") or row.get("best_best_score")
        try:
            score_value = float(score)
        except Exception:
            continue
        if best is None or score_value > best["score"]:
            best = {"score": score_value, "row": row}
    write_json(args.output_dir / "lidc_lightning_best_model_summary.json", best or {"status": "no_completed_runs"})
    print("Comparison report: {}".format(comparison_path))


if __name__ == "__main__":
    main()
