"""Train and evaluate the LUNA16 3D CBAM false-positive reduction model."""

from __future__ import print_function

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lidc_lightning_utils import binary_metrics, confusion_matrix_rows, write_csv
from luna16_candidate_detector import (
    LUNA16CandidateDetector,
    write_detector_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "luna16_candidate_dataset"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "luna16_candidate_detector"
)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "luna16_detection.yaml"


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_training_defaults(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("candidate_training", {})


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class CandidatePatchDataset(Dataset):
    def __init__(self, patch_path, label_path, manifest_rows):
        self.patches = np.load(str(patch_path), mmap_mode="r")
        self.labels = np.load(str(label_path), mmap_mode="r")
        self.rows = list(manifest_rows)
        self.indices = [int(row["array_index"]) for row in self.rows]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        array_index = self.indices[index]
        patch = np.asarray(self.patches[array_index], dtype=np.float32).copy()
        label = int(self.labels[array_index])
        return torch.from_numpy(patch), label


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    labels = []
    probabilities = []
    for patches, batch_labels in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(use_amp),
        ):
            logits = model(patches.to(device, non_blocking=True))
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
        labels.extend(batch_labels.numpy().tolist())
    return binary_metrics(labels, probabilities), labels, probabilities


def plot_history(history, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_auc_roc"] for row in history], label="AUC")
    axes[1].plot(epochs, [row["val_f1"] for row in history], label="F1")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=240)
    plt.close(fig)


def plot_test_roc(labels, probabilities, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    auc_value = auc(false_positive_rate, true_positive_rate)
    fig, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label="3D CBAM (AUC={:.3f})".format(auc_value),
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("LUNA16 candidate detector ROC")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=240)
    plt.close(fig)


def parse_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known, _ = pre_parser.parse_known_args(argv)
    defaults = load_training_defaults(known.config)
    parser = argparse.ArgumentParser(
        description="Train LUNA16 candidate false-positive reduction model.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=resolve_project_path(defaults.get("data_dir", DEFAULT_DATA_DIR)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=resolve_project_path(defaults.get("output_dir", DEFAULT_OUTPUT_DIR)),
    )
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 8)))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(defaults.get("batch_size", 48)),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(defaults.get("learning_rate", 2e-4)),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(defaults.get("weight_decay", 3e-4)),
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=int(defaults.get("base_channels", 8)),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=float(defaults.get("dropout", 0.25)),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(defaults.get("patience", 3)),
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(defaults.get("num_workers", 0)),
    )
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.disable_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    manifest = read_csv_rows(args.data_dir / "candidate_patch_manifest.csv")
    train_rows = [row for row in manifest if row["subset"] not in ("subset8", "subset9")]
    val_rows = [row for row in manifest if row["subset"] == "subset8"]
    test_rows = [row for row in manifest if row["subset"] == "subset9"]
    patch_path = args.data_dir / "candidate_patches.npy"
    label_path = args.data_dir / "candidate_labels.npy"
    train_dataset = CandidatePatchDataset(patch_path, label_path, train_rows)
    val_dataset = CandidatePatchDataset(patch_path, label_path, val_rows)
    test_dataset = CandidatePatchDataset(patch_path, label_path, test_rows)

    train_labels = [int(row["label"]) for row in train_rows]
    counts = np.bincount(train_labels, minlength=2)
    weights = [1.0 / max(counts[label], 1) for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    model = LUNA16CandidateDetector(
        base_channels=args.base_channels,
        dropout=args.dropout,
        attention="cbam",
        spatial_kernel=3,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    stale_epochs = 0
    history = []
    best_path = args.output_dir / "best_candidate_detector.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_losses = []
        for batch_index, (patches, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(patches)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
            if (
                args.max_train_batches is not None
                and batch_index >= int(args.max_train_batches)
            ):
                break

        val_metrics, _, _ = evaluate(model, val_loader, device, use_amp=use_amp)
        val_loss_values = []
        model.eval()
        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    val_loss = criterion(model(patches), labels)
                val_loss_values.append(float(val_loss.item()))
        val_auc = float(val_metrics["auc_roc"] or 0.0)
        scheduler.step(val_auc)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_loss_values)),
            "val_accuracy": val_metrics["accuracy"],
            "val_auc_roc": val_metrics["auc_roc"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            stale_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "base_channels": args.base_channels,
                    "dropout": args.dropout,
                    "attention": "cbam",
                    "spatial_kernel": 3,
                    "architecture_version": "deep_cbam_v2",
                    "patch_shape_zyx": [32, 32, 32],
                    "normalization": "HU[-1000,400] to [0,1]",
                },
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, str(best_path))
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    payload = torch.load(
        str(best_path),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["state_dict"])
    test_metrics, test_labels, test_probabilities = evaluate(
        model,
        test_loader,
        device,
        use_amp=use_amp,
    )
    write_csv(args.output_dir / "training_history.csv", history)
    write_csv(
        args.output_dir / "test_confusion_matrix.csv",
        confusion_matrix_rows(
            test_labels,
            np.stack(
                [1.0 - np.asarray(test_probabilities), np.asarray(test_probabilities)],
                axis=1,
            ),
            ["non_nodule", "nodule"],
        ),
    )
    write_csv(
        args.output_dir / "test_predictions.csv",
        [
            {
                "sample_index": index,
                "true_label": int(label),
                "nodule_probability": float(probability),
                "predicted_label": int(float(probability) >= 0.5),
            }
            for index, (label, probability) in enumerate(
                zip(test_labels, test_probabilities)
            )
        ],
    )
    write_detector_config(
        args.output_dir / "candidate_detector_metrics.json",
        {
            "checkpoint": str(best_path),
            "best_epoch": payload["epoch"],
            "validation_metrics": payload["val_metrics"],
            "test_metrics": test_metrics,
            "train_sample_count": len(train_dataset),
            "validation_sample_count": len(val_dataset),
            "test_sample_count": len(test_dataset),
            "device": str(device),
            "mixed_precision": use_amp,
        },
    )
    plot_history(history, args.output_dir / "candidate_detector_training.png")
    plot_test_roc(
        test_labels,
        test_probabilities,
        args.output_dir / "candidate_detector_test_roc.png",
    )
    print(json.dumps(test_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
