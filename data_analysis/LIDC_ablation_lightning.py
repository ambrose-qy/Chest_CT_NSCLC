"""
Run or plan LIDC-IDRI ablation experiments.

The default ablations isolate augmentation, class weighting, scheduler, and
gradient-clipping components while reusing the shared Lightning training path.
By default this writes a plan only. Add --execute to train sequentially.
"""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "model_results" / "lidc_lightning"
ABLATION_DIR = EXPERIMENT_DIR / "ablations"

ABLATIONS = {
    "baseline": {},
    "no_augmentation": {
        "augment_rotate": False,
        "augment_flip": False,
        "augment_scale": False,
        "augment_noise_std": 0.0,
        "augment_intensity_shift": 0.0,
        "augment_contrast_range": 0.0,
        "augment_cutout_fraction": 0.0,
    },
    "no_rotation": {"augment_rotate": False},
    "no_flipping": {"augment_flip": False},
    "no_scaling": {"augment_scale": False},
    "no_noise": {"augment_noise_std": 0.0},
    "no_intensity_jitter": {"augment_intensity_shift": 0.0, "augment_contrast_range": 0.0},
    "no_class_weights": {"class_weight_mode": "none", "no_class_weights": True},
    "no_lr_scheduler": {"scheduler": "none"},
    "plateau_scheduler": {"scheduler": "plateau"},
    "no_gradient_clipping": {"gradient_clip_val": 0.0},
    "se_attention": {"attention": "se"},
    "no_attention": {"attention": "none"},
    "multiscale_fusion": {"fusion": "multiscale"},
    "multiview_fusion": {"fusion": "multiview"},
}


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


def build_base_overrides(args):
    return {
        "model": args.model,
        "task": args.task,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "early_stop_patience": args.early_stop_patience,
        "seed": args.seed,
        "output_dir": args.output_dir / args.input_dim / args.model / args.task,
        "num_workers": args.num_workers,
        "precision": args.precision,
        "enable_grad_cam": int(args.enable_grad_cam),
        "class_weight_mode": args.class_weight_mode,
        "gradient_clip_val": args.gradient_clip_val,
        "attention": args.attention,
        "fusion": args.fusion,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plan or run LIDC Lightning ablation experiments.")
    parser.add_argument("--input-dim", choices=["2d", "3d"], required=True)
    parser.add_argument("--model", required=True, help="resnet18/densenet121 for 2D, resnet3d/vnet for 3D.")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--ablations", nargs="+", choices=sorted(ABLATIONS.keys()), default=sorted(ABLATIONS.keys()))
    parser.add_argument("--output-dir", type=Path, default=ABLATION_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--class-weight-mode", choices=["balanced", "sqrt_balanced", "custom", "none"], default="sqrt_balanced")
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--attention", choices=["none", "se", "cbam"], default="cbam")
    parser.add_argument("--fusion", choices=["none", "multiscale", "multiview"], default="none")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--enable-grad-cam", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Train each selected ablation sequentially.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size is None:
        args.batch_size = 16 if args.input_dim == "2d" else 4
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_overrides = build_base_overrides(args)
    if args.max_samples_per_split is not None:
        base_overrides["max_samples_per_split"] = args.max_samples_per_split

    trainer = load_trainer_for_model()(args.input_dim, args.model) if args.execute else None
    records = []
    best = None
    for index, name in enumerate(args.ablations, start=1):
        overrides = dict(base_overrides)
        overrides.update(ABLATIONS[name])
        overrides["output_dir"] = base_overrides["output_dir"] / name

        row = {
            "experiment_id": index,
            "ablation": name,
            "input_dim": args.input_dim,
            "model": args.model,
            "task": args.task,
            "overrides": json.dumps({key: str(value) for key, value in overrides.items()}, sort_keys=True),
            "status": "planned",
            "best_score": "",
            "best_config_path": "",
            "error": "",
        }

        if args.execute:
            try:
                payload = trainer(overrides=overrides)
                row["status"] = "complete"
                row["best_score"] = payload.get("best_score", "")
                row["best_config_path"] = payload.get("best_config_path", payload.get("config_path", ""))
                if row["best_score"] != "" and (best is None or float(row["best_score"]) > float(best["best_score"])):
                    best = row
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)

        records.append(row)
        write_csv(args.output_dir / "ablation_records.csv", records)

    write_json(args.output_dir / "ablation_plan.json", {
        "execute": bool(args.execute),
        "records_csv": str(args.output_dir / "ablation_records.csv"),
        "best": best or {"status": "not_available", "reason": "Run with --execute to populate ablation results."},
        "experiments": records,
    })
    print("Ablation plan/records written to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
