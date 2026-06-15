"""
Validate trained LIDC-IDRI Lightning models on LUNA16 external ROI data.

LUNA16 does not provide LIDC-IDRI malignancy labels, so this script performs
external-domain inference rather than supervised accuracy/F1 evaluation. It
reports prediction distributions, confidence, entropy, and diameter-stratified
summaries to assess behaviour under a different dataset distribution.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lidc_lightning_data import BINARY_LABELS, MULTICLASS_LABELS, normalise_tensor
from lidc_lightning_train_utils import trusted_local_checkpoint_load
from lidc_lightning_utils import EXPERIMENT_DIR, import_lightning, write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LUNA_MANIFEST = PROJECT_ROOT / "data" / "processed" / "luna_lidc_external_rois" / "luna_lidc_external_roi_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports" / "lidc_luna_external"


def clean_value(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("none", "nan"):
        return ""
    return text


def safe_float(value):
    text = clean_value(value)
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class LUNAExternalROIDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        input_dim,
        image_size=224,
        in_channels=3,
        normalization_mean=None,
        normalization_std=None,
        max_samples=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.input_dim = input_dim
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        rows = [
            row for row in read_csv_rows(self.manifest_path)
            if clean_value(row.get("volume_path")) and Path(row.get("volume_path")).exists()
        ]
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(row["volume_path"]) as npz:
            volume = npz["volume"].astype(np.float32)

        if self.input_dim == "2d":
            z = volume.shape[0] // 2
            tensor = torch.from_numpy(volume[z]).float().unsqueeze(0).unsqueeze(0)
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if self.in_channels > 1:
                tensor = tensor.repeat(self.in_channels, 1, 1)
        elif self.input_dim == "3d":
            tensor = torch.from_numpy(volume).float().unsqueeze(0)
        else:
            raise ValueError("input_dim must be 2d or 3d.")

        tensor = normalise_tensor(tensor, self.normalization_mean, self.normalization_std)
        return {
            "image": tensor,
            "luna_roi_id": row.get("luna_roi_id", ""),
            "seriesuid": row.get("seriesuid", ""),
            "anonymised_id": row.get("anonymised_id", ""),
            "diameter_mm": torch.tensor(safe_float(row.get("diameter_mm")) or -1.0, dtype=torch.float32),
        }


def config_normalization(config):
    stats = config.get("normalization_stats", {}) or {}
    return stats.get("mean"), stats.get("std")


def entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    denom = math.log(max(probabilities.shape[1], 2))
    values = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1) / denom
    return values


def diameter_bin(value):
    if value is None or value < 0:
        return "missing"
    if value < 6:
        return "lt_6mm"
    if value < 10:
        return "6_to_10mm"
    return "ge_10mm"


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    rows = []
    probabilities = []
    for batch in dataloader:
        images = batch["image"].to(device)
        probs = torch.softmax(model(images), dim=1).detach().cpu().numpy()
        probabilities.append(probs)
        for idx in range(probs.shape[0]):
            rows.append({
                "luna_roi_id": batch["luna_roi_id"][idx],
                "seriesuid": batch["seriesuid"][idx],
                "anonymised_id": batch["anonymised_id"][idx],
                "diameter_mm": float(batch["diameter_mm"][idx].item()),
            })
    if probabilities:
        probabilities = np.concatenate(probabilities, axis=0)
    else:
        probabilities = np.zeros((0, 2), dtype=np.float32)
    return rows, probabilities


def prediction_rows(base_rows, probabilities, class_names):
    ent = entropy(probabilities) if probabilities.size else np.asarray([])
    predictions = probabilities.argmax(axis=1) if probabilities.size else np.asarray([], dtype=np.int64)
    confidence = probabilities.max(axis=1) if probabilities.size else np.asarray([])
    rows = []
    for idx, row in enumerate(base_rows):
        out = dict(row)
        out["predicted_label_id"] = int(predictions[idx])
        out["predicted_label"] = class_names[int(predictions[idx])]
        out["confidence"] = float(confidence[idx])
        out["entropy"] = float(ent[idx])
        out["diameter_bin"] = diameter_bin(out["diameter_mm"])
        for class_id, class_name in enumerate(class_names):
            out["prob_{}".format(class_name)] = float(probabilities[idx, class_id])
        rows.append(out)
    return rows


def summary_rows(rows, class_names):
    if not rows:
        return [{"section": "overall", "name": "sample_count", "value": 0}]

    out = [{"section": "overall", "name": "sample_count", "value": len(rows)}]
    out.append({"section": "overall", "name": "mean_confidence", "value": sum(row["confidence"] for row in rows) / float(len(rows))})
    out.append({"section": "overall", "name": "mean_entropy", "value": sum(row["entropy"] for row in rows) / float(len(rows))})
    for class_name in class_names:
        count = sum(1 for row in rows if row["predicted_label"] == class_name)
        out.append({"section": "predicted_label", "name": class_name, "value": count})
        out.append({"section": "predicted_label_fraction", "name": class_name, "value": count / float(len(rows))})

    bins = sorted(set(row["diameter_bin"] for row in rows))
    for bin_name in bins:
        bin_rows = [row for row in rows if row["diameter_bin"] == bin_name]
        out.append({"section": "diameter_bin", "name": "{}_sample_count".format(bin_name), "value": len(bin_rows)})
        out.append({"section": "diameter_bin", "name": "{}_mean_confidence".format(bin_name), "value": sum(row["confidence"] for row in bin_rows) / float(len(bin_rows))})
        out.append({"section": "diameter_bin", "name": "{}_mean_entropy".format(bin_name), "value": sum(row["entropy"] for row in bin_rows) / float(len(bin_rows))})
        for class_name in class_names:
            count = sum(1 for row in bin_rows if row["predicted_label"] == class_name)
            out.append({"section": "diameter_bin_predicted_label", "name": "{}_{}".format(bin_name, class_name), "value": count})
    return out


def load_checkpoint_from_config(config_path, checkpoint_path=None):
    config_path = Path(config_path)
    run_dir = config_path.parent
    if checkpoint_path:
        return Path(checkpoint_path)
    best_path = run_dir / "best_config.json"
    if best_path.exists():
        best = read_json(best_path)
        candidate = clean_value(best.get("best_checkpoint"))
        if candidate and Path(candidate).exists():
            return Path(candidate)
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError("Could not find checkpoint for {}".format(config_path))


def evaluate_one(config_path, manifest_path, output_dir, checkpoint_path=None, batch_size=None, max_samples=None, num_workers=0):
    import_lightning()
    from lidc_lightning_module import LIDCClassifier

    config = read_json(config_path)
    config_args = config.get("args", {}) or {}
    input_dim = config.get("input_dim")
    task = config.get("task") or config_args.get("task", "binary")
    class_names = BINARY_LABELS if task == "binary" else MULTICLASS_LABELS
    mean, std = config_normalization(config)
    checkpoint = load_checkpoint_from_config(config_path, checkpoint_path)

    dataset = LUNAExternalROIDataset(
        manifest_path=manifest_path,
        input_dim=input_dim,
        image_size=int(config_args.get("image_size", 224)),
        in_channels=int(config_args.get("in_channels", 3)),
        normalization_mean=mean,
        normalization_std=std,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or config_args.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_kwargs = {}
    if config.get("class_weights") is not None:
        load_kwargs["class_weights"] = config.get("class_weights")
    with trusted_local_checkpoint_load():
        model = LIDCClassifier.load_from_checkpoint(str(checkpoint), map_location=device, **load_kwargs)
    model.to(device)

    base_rows, probabilities = predict(model, loader, device)
    rows = prediction_rows(base_rows, probabilities, class_names)
    summaries = summary_rows(rows, class_names)

    run_name = config.get("run_name", Path(config_path).parent.name)
    target = Path(output_dir) / run_name
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "luna_external_predictions.csv", rows)
    write_csv(target / "luna_external_summary.csv", summaries)
    write_json(target / "luna_external_metadata.json", {
        "run_name": run_name,
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "manifest_path": str(manifest_path),
        "input_dim": input_dim,
        "task": task,
        "sample_count": len(rows),
        "label_note": "LUNA16 has no LIDC malignancy labels; outputs are external-domain inference summaries.",
    })
    return {
        "run_name": run_name,
        "input_dim": input_dim,
        "task": task,
        "model": config_args.get("model", ""),
        "sample_count": len(rows),
        "output_dir": str(target),
        "predictions_csv": str(target / "luna_external_predictions.csv"),
        "summary_csv": str(target / "luna_external_summary.csv"),
    }


def completed_run_configs(run_root):
    configs = []
    for config_path in sorted(Path(run_root).rglob("config.json")):
        try:
            load_checkpoint_from_config(config_path)
            configs.append(config_path)
        except Exception:
            continue
    return configs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run LIDC models on LUNA16 external ROI data.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_LUNA_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--all-runs", action="store_true", help="Evaluate all completed runs under --run-root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.manifest).exists():
        raise FileNotFoundError("Missing LUNA external ROI manifest. Run LUNA_export_lidc_external_rois.py first: {}".format(args.manifest))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if args.all_runs:
        configs = completed_run_configs(args.run_root)
        for config_path in configs:
            try:
                records.append(evaluate_one(
                    config_path=config_path,
                    manifest_path=args.manifest,
                    output_dir=args.output_dir,
                    checkpoint_path=None,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    num_workers=args.num_workers,
                ))
            except Exception as exc:
                records.append({"run_name": Path(config_path).parent.name, "status": "failed", "error": str(exc)})
    else:
        if not args.config:
            raise ValueError("Pass --config for one run, or --all-runs to evaluate every completed run.")
        records.append(evaluate_one(
            config_path=args.config,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
        ))

    write_csv(args.output_dir / "luna_external_validation_records.csv", records)
    write_json(args.output_dir / "luna_external_validation_summary.json", {
        "manifest": str(args.manifest),
        "record_count": len(records),
        "records_csv": str(args.output_dir / "luna_external_validation_records.csv"),
        "label_note": "LUNA16 has no LIDC malignancy labels; these are external-domain inference summaries.",
    })
    print("LUNA external validation records: {}".format(args.output_dir / "luna_external_validation_records.csv"))


if __name__ == "__main__":
    main()
