"""
Run or plan the required LIDC-IDRI baseline experiment matrix.

The matrix covers:

* 2D ResNet and DenseNet on binary and multiclass tasks
* 3D ResNet and VNet on binary and multiclass tasks

By default this writes a plan only. Add --execute to train sequentially.
"""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

from lidc_2d_slices import DEFAULT_OUTPUT_TABLES, build_slice_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "model_results" / "lidc_lightning"
REQUIRED_EXPERIMENTS = [
    {"input_dim": "2d", "model": "resnet18", "task": "binary", "batch_size": 16},
    {"input_dim": "2d", "model": "resnet18", "task": "multiclass", "batch_size": 16},
    {"input_dim": "2d", "model": "densenet121", "task": "binary", "batch_size": 16},
    {"input_dim": "2d", "model": "densenet121", "task": "multiclass", "batch_size": 16},
    {"input_dim": "3d", "model": "resnet3d", "task": "binary", "batch_size": 4},
    {"input_dim": "3d", "model": "resnet3d", "task": "multiclass", "batch_size": 4},
    {"input_dim": "3d", "model": "vnet", "task": "binary", "batch_size": 4},
    {"input_dim": "3d", "model": "vnet", "task": "multiclass", "batch_size": 4},
]

OUTPUT_DIR = EXPERIMENT_DIR / "required_baselines"


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


def load_trainer_for_model():
    from LIDC_tune_lightning import trainer_for_model
    return trainer_for_model


def maybe_build_2d_manifest(task, force=False):
    path = DEFAULT_OUTPUT_TABLES[task]
    if force or not Path(path).exists():
        build_slice_manifest(output_path=path, task=task)
    return path


def selected_experiments(args):
    rows = []
    for config in REQUIRED_EXPERIMENTS:
        if args.input_dim and config["input_dim"] != args.input_dim:
            continue
        if args.task and config["task"] != args.task:
            continue
        if args.model and config["model"] != args.model:
            continue
        rows.append(dict(config))
    return rows


def build_overrides(args, config):
    epochs = args.epochs_2d if config["input_dim"] == "2d" else args.epochs_3d
    overrides = {
        "model": config["model"],
        "task": config["task"],
        "epochs": epochs,
        "batch_size": config["batch_size"],
        "seed": args.seed,
        "output_dir": args.output_dir / config["input_dim"] / config["model"] / config["task"],
        "num_workers": args.num_workers,
        "precision": args.precision,
        "enable_grad_cam": int(args.enable_grad_cam),
        "attention": args.attention,
        "fusion": args.fusion_3d if config["input_dim"] == "3d" else "none",
    }
    if args.max_samples_per_split is not None:
        overrides["max_samples_per_split"] = args.max_samples_per_split
    return overrides


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plan or run the full required LIDC baseline matrix.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--input-dim", choices=["2d", "3d"], default=None)
    parser.add_argument("--task", choices=["binary", "multiclass"], default=None)
    parser.add_argument("--model", choices=["resnet18", "densenet121", "resnet3d", "vnet"], default=None)
    parser.add_argument("--epochs-2d", type=int, default=20)
    parser.add_argument("--epochs-3d", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--force-build-2d-manifests", action="store_true")
    parser.add_argument("--enable-grad-cam", type=int, default=1)
    parser.add_argument("--attention", choices=["none", "se", "cbam"], default="cbam")
    parser.add_argument("--fusion-3d", choices=["none", "multiscale", "multiview"], default="multiscale")
    parser.add_argument("--execute", action="store_true", help="Train each selected experiment sequentially.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = selected_experiments(args)

    plan_rows = []
    for index, config in enumerate(experiments, start=1):
        if config["input_dim"] == "2d":
            manifest_path = maybe_build_2d_manifest(config["task"], force=args.force_build_2d_manifests)
        else:
            manifest_path = Path("data/processed/tables/lidc_roi_3d_volume_manifest.csv")

        overrides = build_overrides(args, config)
        row = {
            "experiment_id": index,
            "input_dim": config["input_dim"],
            "model": config["model"],
            "task": config["task"],
            "manifest_path": str(manifest_path),
            "overrides": json.dumps({key: str(value) for key, value in overrides.items()}, sort_keys=True),
            "status": "planned",
            "best_score": "",
            "best_config_path": "",
            "error": "",
        }

        if args.execute:
            try:
                trainer = load_trainer_for_model()(config["input_dim"], config["model"])
                payload = trainer(overrides=overrides)
                row["status"] = "complete"
                row["best_score"] = payload.get("best_score", "")
                row["best_config_path"] = payload.get("best_config_path", payload.get("config_path", ""))
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)

        plan_rows.append(row)
        write_csv(args.output_dir / "required_baseline_records.csv", plan_rows)

    write_json(args.output_dir / "required_baseline_plan.json", {
        "execute": bool(args.execute),
        "experiment_count": len(plan_rows),
        "records_csv": str(args.output_dir / "required_baseline_records.csv"),
        "experiments": plan_rows,
    })
    print("Required baseline plan/records written to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
