"""Evaluate the trained LUNA16 3D CBAM candidate detector."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from LUNA_train_candidate_detector import (
    CandidatePatchDataset,
    plot_test_roc,
    read_csv_rows,
)
from lidc_lightning_utils import binary_metrics, confusion_matrix_rows, write_csv
from luna16_candidate_detector import load_candidate_detector, write_detector_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "luna16_candidate_dataset"
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "luna16_candidate_detector"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the LUNA16 candidate detector on a held-out subset."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--subset", default="subset9")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


@torch.no_grad()
def predict(model, loader, device):
    labels = []
    probabilities = []
    use_amp = device.type == "cuda"
    for patches, batch_labels in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(patches.to(device, non_blocking=True))
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.extend(batch_labels.numpy())
    return np.asarray(labels), np.asarray(probabilities)


def main(argv=None):
    args = parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = args.checkpoint or args.model_dir / "best_candidate_detector.pt"
    model, config = load_candidate_detector(checkpoint, device)

    manifest = read_csv_rows(args.data_dir / "candidate_patch_manifest.csv")
    rows = [row for row in manifest if row["subset"] == args.subset]
    dataset = CandidatePatchDataset(
        args.data_dir / "candidate_patches.npy",
        args.data_dir / "candidate_labels.npy",
        rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    labels, probabilities = predict(model, loader, device)
    metrics = binary_metrics(labels.tolist(), probabilities.tolist())
    args.model_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.model_dir / "{}_predictions.csv".format(args.subset),
        [
            {
                "sample_index": index,
                "array_index": rows[index]["array_index"],
                "seriesuid": rows[index]["seriesuid"],
                "true_label": int(label),
                "nodule_probability": float(probability),
                "predicted_label": int(probability >= 0.5),
            }
            for index, (label, probability) in enumerate(zip(labels, probabilities))
        ],
    )
    probability_matrix = np.stack([1.0 - probabilities, probabilities], axis=1)
    write_csv(
        args.model_dir / "{}_confusion_matrix.csv".format(args.subset),
        confusion_matrix_rows(
            labels.tolist(),
            probability_matrix,
            ["non_nodule", "nodule"],
        ),
    )
    write_detector_config(
        args.model_dir / "{}_metrics.json".format(args.subset),
        {
            "checkpoint": str(checkpoint),
            "subset": args.subset,
            "sample_count": len(dataset),
            "metrics": metrics,
            "model_config": config,
            "device": str(device),
        },
    )
    plot_test_roc(
        labels,
        probabilities,
        args.model_dir / "{}_roc.png".format(args.subset),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
