"""
Launch and record small LIDC-IDRI Lightning hyperparameter experiments.

By default this writes a tuning plan without executing training. Add --execute
to run each configuration sequentially.
"""

from __future__ import print_function

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from lidc_lightning_utils import EXPERIMENT_DIR, write_csv, write_json


TUNING_DIR = EXPERIMENT_DIR / "hparam_tuning"


def parse_list(text, cast=str):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def default_grid(input_dim):
    if input_dim == "2d":
        return {
            "model": ["resnet18", "densenet121"],
            "lr": [1e-4, 3e-4],
            "weight_decay": [1e-4],
            "batch_size": [16],
            "scheduler": ["cosine", "plateau"],
        }
    return {
        "model": ["resnet3d", "vnet"],
        "lr": [1e-4, 3e-4],
        "weight_decay": [1e-4],
        "batch_size": [4],
        "scheduler": ["cosine", "plateau"],
    }


def load_grid(path, input_dim):
    if path is None:
        return default_grid(input_dim)
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def expand_grid(grid):
    keys = sorted(grid.keys())
    for values in itertools.product(*[grid[key] for key in keys]):
        yield dict(zip(keys, values))


def build_command(args, config, run_output_dir):
    script = "LIDC_train_2d_lightning.py" if args.input_dim == "2d" else "LIDC_train_3d_lightning.py"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / script),
        "--model", str(config["model"]),
        "--task", args.task,
        "--epochs", str(args.epochs),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--scheduler", str(config["scheduler"]),
        "--early-stop-patience", str(args.early_stop_patience),
        "--seed", str(args.seed),
        "--output-dir", str(run_output_dir),
        "--num-workers", str(args.num_workers),
        "--precision", args.precision,
    ]
    if args.max_samples_per_split is not None:
        command.extend(["--max-samples-per-split", str(args.max_samples_per_split)])
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run LIDC Lightning hyperparameter tuning experiments.")
    parser.add_argument("--input-dim", choices=["2d", "3d"], required=True)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--grid-json", type=Path, default=None, help="Optional grid JSON with list values.")
    parser.add_argument("--output-dir", type=Path, default=TUNING_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only writes plan.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = load_grid(args.grid_json, args.input_dim)
    configs = list(expand_grid(grid))

    plan_rows = []
    best = None
    for index, config in enumerate(configs, start=1):
        run_output_dir = args.output_dir / "{}_runs".format(args.input_dim)
        command = build_command(args, config, run_output_dir)
        row = {
            "experiment_id": index,
            "input_dim": args.input_dim,
            "task": args.task,
            "model": config["model"],
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "batch_size": config["batch_size"],
            "scheduler": config["scheduler"],
            "command": " ".join(str(part) for part in command),
            "status": "planned",
            "return_code": "",
            "best_score": "",
            "best_config_path": "",
        }

        if args.execute:
            completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[1]))
            row["return_code"] = completed.returncode
            row["status"] = "complete" if completed.returncode == 0 else "failed"
            run_name = "{}_{}_{}_seed{}".format(args.input_dim, str(config["model"]).lower(), args.task, args.seed)
            best_config_path = run_output_dir / run_name / "best_config.json"
            row["best_config_path"] = str(best_config_path)
            if best_config_path.exists():
                with best_config_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                row["best_score"] = payload.get("best_score", "")
                if row["best_score"] != "" and (best is None or float(row["best_score"]) > float(best["best_score"])):
                    best = row

        plan_rows.append(row)
        write_csv(args.output_dir / "{}_tuning_records.csv".format(args.input_dim), plan_rows)

    write_json(args.output_dir / "{}_tuning_grid.json".format(args.input_dim), grid)
    if best is not None:
        write_json(args.output_dir / "{}_best_hparams.json".format(args.input_dim), best)
    else:
        write_json(args.output_dir / "{}_best_hparams.json".format(args.input_dim), {
            "status": "not_available",
            "reason": "Run with --execute to populate best hyperparameter results.",
        })
    print("Tuning plan/records written to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
