"""
Train ResNet/DenseNet 2D CNN baselines on the LIDC-IDRI binary ROI split.

Default first baseline:

    conda run -n torch-gpu python data_analysis/LIDC_train_2d_baseline.py --model resnet18

The script uses train/val/test splits from LIDC_process4.py and nodule-centered
2D slices from lidc_2d_slices.py.
"""

from __future__ import print_function

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lidc_2d_dataset import LIDC2DNoduleDataset, make_lidc_dataloader
from lidc_2d_models import create_lidc_model
from lidc_2d_slices import DEFAULT_OUTPUT_TABLE, build_slice_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "models" / "lidc_2d_baselines"


def log(message):
    print(message)
    sys.stdout.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_counts(dataset):
    return Counter(int(row["binary_label_id"]) for row in dataset.rows)


def class_weight_tensor(dataset, device):
    counts = label_counts(dataset)
    total = sum(counts.values())
    weights = []
    for label in [0, 1]:
        count = counts.get(label, 0)
        weights.append(total / (2.0 * count) if count else 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def maybe_auc(labels, scores):
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / float(max(total, 1)),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    labels_all = []
    scores_all = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
        scores_all.extend(probs.detach().cpu().numpy().astype(float).tolist())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / float(max(total, 1)),
        "auc": maybe_auc(labels_all, scores_all),
    }


def write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_auc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def checkpoint_args(args):
    values = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train LIDC-IDRI 2D ResNet/DenseNet baseline.")
    parser.add_argument("--model", default="resnet18", help="resnet18, resnet34, resnet50, densenet121, or densenet169.")
    parser.add_argument("--slice-manifest", type=Path, default=DEFAULT_OUTPUT_TABLE, help="2D slice manifest CSV.")
    parser.add_argument("--force-build-slice-manifest", action="store_true", help="Regenerate the 2D slice manifest before training.")
    parser.add_argument("--skip-build-slice-manifest", action="store_true", help="Fail if the 2D slice manifest does not already exist.")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR, help="Directory for checkpoints and metric CSVs.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0 on Windows unless multiprocessing is configured.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained torchvision weights.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Debug limit per split.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weights.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    if args.force_build_slice_manifest or (not args.slice_manifest.exists() and not args.skip_build_slice_manifest):
        log("Building 2D slice manifest: {}".format(args.slice_manifest))
        build_slice_manifest(output_path=args.slice_manifest)

    if not args.slice_manifest.exists():
        raise FileNotFoundError("Missing 2D slice manifest: {}".format(args.slice_manifest))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("Device: {}".format(device))
    log("Model: {}".format(args.model))
    log("Slice manifest: {}".format(args.slice_manifest))

    train_dataset = LIDC2DNoduleDataset(
        args.slice_manifest,
        split="train",
        image_size=args.image_size,
        augment=True,
        max_samples=args.max_samples_per_split,
    )
    train_loader = make_lidc_dataloader(
        args.slice_manifest,
        split="train",
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment=True,
        num_workers=args.num_workers,
        max_samples=args.max_samples_per_split,
    )
    val_loader = make_lidc_dataloader(
        args.slice_manifest,
        split="val",
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment=False,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.max_samples_per_split,
    )
    test_loader = make_lidc_dataloader(
        args.slice_manifest,
        split="test",
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment=False,
        shuffle=False,
        num_workers=args.num_workers,
        max_samples=args.max_samples_per_split,
    )

    log("Train samples: {} labels {}".format(len(train_loader.dataset), dict(label_counts(train_dataset))))
    log("Val samples: {}".format(len(val_loader.dataset)))
    log("Test samples: {}".format(len(test_loader.dataset)))

    model = create_lidc_model(args.model, num_classes=2, pretrained=args.pretrained, in_channels=3).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=None if args.no_class_weights else class_weight_tensor(train_dataset, device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    run_name = "{}_binary_seed{}".format(args.model.lower(), args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "{}_best.pt".format(run_name)
    last_path = args.output_dir / "{}_last.pt".format(run_name)
    metrics_path = args.output_dir / "{}_metrics.csv".format(run_name)

    best_score = None
    metrics_rows = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        val_auc = val_metrics["auc"]
        score = val_auc if val_auc is not None else val_metrics["accuracy"]
        is_best = best_score is None or score > best_score
        if is_best:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": checkpoint_args(args),
            }, best_path)

        metrics_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": "" if val_auc is None else val_auc,
        }
        metrics_rows.append(metrics_row)
        write_metrics(metrics_path, metrics_rows)
        log(
            "Epoch {}/{} train_loss={:.4f} train_acc={:.3f} val_loss={:.4f} val_acc={:.3f} val_auc={}".format(
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                "NA" if val_auc is None else "{:.3f}".format(val_auc),
            )
        )

    torch.save({
        "epoch": args.epochs,
        "model_name": args.model,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": checkpoint_args(args),
    }, last_path)

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    log("")
    log("Training complete")
    log("  best checkpoint: {}".format(best_path))
    log("  last checkpoint: {}".format(last_path))
    log("  metrics: {}".format(metrics_path))
    log("  test loss: {:.4f}".format(test_metrics["loss"]))
    log("  test accuracy: {:.3f}".format(test_metrics["accuracy"]))
    log("  test AUC: {}".format("NA" if test_metrics["auc"] is None else "{:.3f}".format(test_metrics["auc"])))


if __name__ == "__main__":
    main()
