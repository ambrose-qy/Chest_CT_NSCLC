"""Evaluate selected LIDC models on label-matched LUNA16 image-domain ROIs."""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from lidc_lightning_data import (
    BINARY_LABELS,
    MULTICLASS_LABELS,
    normalise_tensor,
)
from lidc_lightning_train_utils import (
    configure_torch_matmul_precision,
    trusted_local_checkpoint_load,
)
from lidc_lightning_utils import (
    binary_metrics,
    confusion_matrix_rows,
    multiclass_metrics,
    roc_curve_rows,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "luna16_labeled_external_rois"
    / "luna16_labeled_external_roi_manifest.csv"
)
DEFAULT_SELECTION = PROJECT_ROOT / "configs" / "lidc_stage4_selected_runs.yaml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_reports"
    / "lidc_luna16_generalization"
)


def clean_value(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("none", "nan") else text


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def label_fields(task):
    if task == "binary":
        return "binary_label_id", "binary_split", BINARY_LABELS
    return "multiclass_risk_label_id", "multiclass_split", MULTICLASS_LABELS


class LUNA16LabeledROIDataset(Dataset):
    def __init__(self, manifest_path, config, task, split="test"):
        self.config = config
        self.task = task
        self.input_dim = config.get("input_dim")
        self.args = config.get("args", {}) or {}
        self.image_size = int(self.args.get("image_size", 224))
        self.in_channels = int(self.args.get("in_channels", 3))
        stats = config.get("normalization_stats", {}) or {}
        self.mean = stats.get("mean")
        self.std = stats.get("std")
        label_column, split_column, _ = label_fields(task)
        self.label_column = label_column
        self.rows = [
            row
            for row in read_csv_rows(manifest_path)
            if clean_value(row.get(label_column))
            and (split == "all" or clean_value(row.get(split_column)) == split)
            and Path(clean_value(row.get("volume_path"))).exists()
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(clean_value(row["volume_path"])) as npz:
            volume = np.asarray(npz["volume"], dtype=np.float32)
        if self.input_dim == "2d":
            tensor = torch.from_numpy(volume[volume.shape[0] // 2]).float()[None, None]
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
            raise ValueError("Unsupported input_dim: {}".format(self.input_dim))
        tensor = normalise_tensor(tensor, self.mean, self.std)
        return {
            "image": tensor,
            "label": int(float(row[self.label_column])),
            "roi_id": clean_value(row.get("matched_roi_id")),
            "seriesuid": clean_value(row.get("seriesuid")),
            "diameter_mm": float(row.get("diameter_mm") or -1.0),
            "match_distance_mm": float(row.get("match_distance_mm") or -1.0),
        }


def selected_checkpoint(run_dir, item):
    if item.get("checkpoint"):
        path = run_dir / item["checkpoint"]
        if path.exists():
            return path
    best_path = run_dir / "best_config.json"
    if best_path.exists():
        best = read_json(best_path)
        candidate = Path(clean_value(best.get("best_checkpoint")))
        if candidate.exists():
            return candidate
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError("No checkpoint found for {}".format(run_dir))


def load_model(checkpoint, config, device):
    from lidc_lightning_module import LIDCClassifier

    load_kwargs = {}
    if config.get("class_weights") is not None:
        load_kwargs["class_weights"] = config.get("class_weights")
    with trusted_local_checkpoint_load():
        model = LIDCClassifier.load_from_checkpoint(
            str(checkpoint),
            map_location=device,
            **load_kwargs
        )
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device):
    labels = []
    probabilities = []
    metadata = []
    for batch in loader:
        images = batch["image"].to(device)
        probs = torch.softmax(model(images, pca_features=None), dim=1).cpu().numpy()
        probabilities.append(probs)
        labels.extend(batch["label"].cpu().numpy().tolist())
        for index in range(probs.shape[0]):
            metadata.append({
                "roi_id": batch["roi_id"][index],
                "seriesuid": batch["seriesuid"][index],
                "diameter_mm": float(batch["diameter_mm"][index].item()),
                "match_distance_mm": float(batch["match_distance_mm"][index].item()),
            })
    if probabilities:
        probabilities = np.concatenate(probabilities, axis=0)
    else:
        probabilities = np.zeros((0, 2), dtype=np.float32)
    return np.asarray(labels, dtype=np.int64), probabilities, metadata


def prediction_rows(labels, probabilities, metadata, class_names):
    predictions = probabilities.argmax(axis=1)
    rows = []
    for index, meta in enumerate(metadata):
        row = dict(meta)
        row.update({
            "true_label_id": int(labels[index]),
            "true_label": class_names[int(labels[index])],
            "predicted_label_id": int(predictions[index]),
            "predicted_label": class_names[int(predictions[index])],
            "confidence": float(probabilities[index].max()),
            "correct": bool(predictions[index] == labels[index]),
        })
        for class_index, class_name in enumerate(class_names):
            row["prob_{}".format(class_name)] = float(probabilities[index, class_index])
        rows.append(row)
    return rows


def plot_outputs(output_dir, labels, probabilities, class_names, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

    output_dir = Path(output_dir)
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(output_dir / "luna16_confusion_matrix.png"), dpi=240)
    plt.close(fig)

    if len(class_names) == 2 and len(set(labels.tolist())) >= 2:
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        fig, ax = plt.subplots(figsize=(5.2, 4.5))
        ax.plot(fpr, tpr, linewidth=2.2)
        ax.plot([0, 1], [0, 1], "--", color="#777777")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(str(output_dir / "luna16_roc_curve.png"), dpi=240)
        plt.close(fig)


def evaluate_run(item, manifest_path, output_root, split, device, num_workers=0):
    run_dir = PROJECT_ROOT / item["run_dir"]
    config = read_json(run_dir / "config.json")
    task = item["task"]
    _, _, class_names = label_fields(task)
    dataset = LUNA16LabeledROIDataset(manifest_path, config, task, split=split)
    batch_size = int(config.get("args", {}).get("batch_size", 8))
    if config.get("input_dim") == "2d":
        batch_size = max(batch_size, 16)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    checkpoint = selected_checkpoint(run_dir, item)
    model = load_model(checkpoint, config, device)
    labels, probabilities, metadata = predict(model, loader, device)
    if not len(labels):
        raise RuntimeError("No labeled LUNA16 rows for {}.".format(run_dir.name))
    metrics = (
        binary_metrics(labels, probabilities[:, 1])
        if task == "binary"
        else multiclass_metrics(labels, probabilities)
    )

    output_dir = Path(output_root) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = prediction_rows(labels, probabilities, metadata, class_names)
    write_csv(output_dir / "luna16_predictions.csv", rows)
    write_csv(
        output_dir / "luna16_confusion_matrix.csv",
        confusion_matrix_rows(labels, probabilities, class_names),
    )
    if task == "binary":
        write_csv(
            output_dir / "luna16_roc_curve.csv",
            roc_curve_rows(labels, probabilities[:, 1]),
        )
    write_json(output_dir / "luna16_metrics.json", metrics)
    plot_outputs(
        output_dir,
        labels,
        probabilities,
        class_names,
        "{} {} LUNA16".format(item["input_dim"].upper(), item["model"]),
    )

    internal = read_json(run_dir / "test_metrics.json")
    record = {
        "input_dim": item["input_dim"],
        "model": item["model"],
        "task": task,
        "run_name": run_dir.name,
        "sample_count": metrics["sample_count"],
        "accuracy": metrics["accuracy"],
        "auc_roc": metrics["auc_roc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "internal_accuracy": internal.get("test_accuracy", internal.get("accuracy", "")),
        "internal_auc_roc": internal.get("test_auc_roc", internal.get("auc_roc", "")),
        "internal_f1": internal.get("test_f1", internal.get("f1", "")),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
    }
    for metric in ("accuracy", "auc_roc", "f1"):
        external_value = record.get(metric)
        internal_value = record.get("internal_{}".format(metric))
        try:
            record["{}_gap_external_minus_internal".format(metric)] = (
                float(external_value) - float(internal_value)
            )
        except Exception:
            record["{}_gap_external_minus_internal".format(metric)] = ""
    return record


def plot_comparison(records, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "{} {} {}".format(
            row["input_dim"].upper(),
            row["model"],
            "B" if row["task"] == "binary" else "M",
        )
        for row in records
    ]
    external = np.asarray([float(row["f1"]) for row in records])
    internal = np.asarray([float(row["internal_f1"]) for row in records])
    x = np.arange(len(records))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, internal, width, label="LIDC internal test F1", color="#2F6B8A")
    ax.bar(x + width / 2, external, width, label="LUNA16-domain test F1", color="#D98B3A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("F1")
    ax.set_title("LIDC Internal vs LUNA16 Preprocessing-Domain Generalization")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=260)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate selected LIDC models on matched LUNA16 test ROIs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.manifest.exists():
        raise FileNotFoundError(
            "Missing matched LUNA16 manifest. Run LUNA_build_labeled_external_rois.py first."
        )
    with args.selection.open("r", encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    configure_torch_matmul_precision("medium")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in selection["runs"]:
        print(
            "Evaluating {} {} {} on LUNA16 domain...".format(
                item["input_dim"], item["model"], item["task"]
            ),
            flush=True,
        )
        records.append(
            evaluate_run(
                item,
                args.manifest,
                args.output_dir,
                args.split,
                device,
                num_workers=args.num_workers,
            )
        )
    write_csv(args.output_dir / "luna16_generalization_summary.csv", records)
    plot_comparison(records, args.output_dir / "luna16_generalization_f1_comparison.png")
    write_json(
        args.output_dir / "luna16_generalization_metadata.json",
        {
            "manifest": str(args.manifest),
            "selection": str(args.selection),
            "split": args.split,
            "model_count": len(records),
            "summary_csv": str(args.output_dir / "luna16_generalization_summary.csv"),
            "interpretation": (
                "LUNA16 is derived from LIDC-IDRI. Metrics measure robustness to the "
                "LUNA MetaImage/preprocessing domain on the original patient-level "
                "LIDC test split; they are not independent-hospital validation."
            ),
        },
    )
    print("Summary: {}".format(args.output_dir / "luna16_generalization_summary.csv"))


if __name__ == "__main__":
    main()
