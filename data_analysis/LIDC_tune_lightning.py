"""
Launch and record small LIDC-IDRI Lightning hyperparameter experiments.

By default this writes a tuning plan without executing training. Add --execute
to run each configuration sequentially.
"""

from __future__ import print_function

import argparse
import itertools
import json
from pathlib import Path

from LIDC_2d_densenet import train_2d_densenet
from LIDC_2d_resnet import train_2d_resnet
from LIDC_3d_resnet import train_3d_resnet
from LIDC_3d_vnet import train_3d_vnet
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


def trainer_for_model(input_dim, model_name):
    name = str(model_name).lower()
    if input_dim == "2d" and name.startswith("resnet"):
        return train_2d_resnet
    if input_dim == "2d" and name.startswith("densenet"):
        return train_2d_densenet
    if input_dim == "3d" and name.startswith("resnet"):
        return train_3d_resnet
    if input_dim == "3d" and name.startswith("vnet"):
        return train_3d_vnet
    raise ValueError("No trainer function for input_dim={} model={}".format(input_dim, model_name))


def build_overrides(args, config, run_output_dir):
    overrides = {
        "model": config["model"],
        "task": args.task,
        "epochs": args.epochs,
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "scheduler": config["scheduler"],
        "early_stop_patience": args.early_stop_patience,
        "seed": args.seed,
        "output_dir": run_output_dir,
        "num_workers": args.num_workers,
        "precision": args.precision,
    }
    if args.max_samples_per_split is not None:
        overrides["max_samples_per_split"] = args.max_samples_per_split
    return overrides


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
        overrides = build_overrides(args, config, run_output_dir)
        row = {
            "experiment_id": index,
            "input_dim": args.input_dim,
            "task": args.task,
            "model": config["model"],
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "batch_size": config["batch_size"],
            "scheduler": config["scheduler"],
            "trainer_function": trainer_for_model(args.input_dim, config["model"]).__name__,
            "overrides": json.dumps({key: str(value) for key, value in overrides.items()}, sort_keys=True),
            "status": "planned",
            "return_code": "",
            "best_score": "",
            "best_config_path": "",
        }

        if args.execute:
            try:
                trainer = trainer_for_model(args.input_dim, config["model"])
                payload = trainer(overrides=overrides)
                row["return_code"] = 0
                row["status"] = "complete"
                row["best_config_path"] = payload.get("best_config_path", payload.get("config_path", ""))
                row["best_score"] = payload.get("best_score", "")
                if row["best_score"] != "" and (best is None or float(row["best_score"]) > float(best["best_score"])):
                    best = row
            except Exception as exc:
                row["return_code"] = 1
                row["status"] = "failed"
                row["error"] = str(exc)

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
