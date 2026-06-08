"""
Shared utilities for LIDC-IDRI Lightning experiments.
"""

from __future__ import print_function

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "processed" / "model_results" / "lidc_lightning"


def log(message):
    print(message)
    sys.stdout.flush()


def import_lightning():
    try:
        import lightning.pytorch as pl
        return pl
    except Exception:
        try:
            import pytorch_lightning as pl
            return pl
        except Exception as exc:
            raise ImportError(
                "PyTorch Lightning is required for these training scripts. "
                "Install it in torch-gpu, for example: conda run -n torch-gpu pip install lightning"
            ) from exc


def lightning_available():
    try:
        import_lightning()
        return True
    except Exception:
        return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def safe_int(value):
    value = safe_float(value)
    if value is None:
        return None
    return int(value)


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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


def binary_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    total = int(labels.size)

    accuracy = (tp + tn) / float(total) if total else 0.0
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels.tolist())) >= 2:
            auc = float(roc_auc_score(labels, probabilities))
    except Exception:
        auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": "" if auc is None else auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sample_count": total,
    }


def multiclass_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    total = int(labels.size)
    accuracy = float((predictions == labels).sum() / max(total, 1))

    try:
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
        auc = ""
        if len(set(labels.tolist())) > 1:
            auc = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except Exception:
        precision = recall = f1 = 0.0
        auc = ""

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": auc,
        "sample_count": total,
    }


def confusion_matrix_rows(labels, probabilities, class_names):
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim == 1:
        predictions = (probabilities >= 0.5).astype(np.int64)
    else:
        predictions = probabilities.argmax(axis=1)

    rows = []
    for actual_id, actual_name in enumerate(class_names):
        for predicted_id, predicted_name in enumerate(class_names):
            rows.append({
                "actual_id": actual_id,
                "actual_label": actual_name,
                "predicted_id": predicted_id,
                "predicted_label": predicted_name,
                "count": int(((labels == actual_id) & (predictions == predicted_id)).sum()),
            })
    return rows


def roc_curve_rows(labels, probabilities):
    rows = []
    try:
        from sklearn.metrics import roc_curve
        labels = np.asarray(labels, dtype=np.int64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if len(set(labels.tolist())) < 2:
            return rows
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        for idx in range(len(fpr)):
            rows.append({
                "threshold": float(thresholds[idx]),
                "false_positive_rate": float(fpr[idx]),
                "true_positive_rate": float(tpr[idx]),
            })
    except Exception:
        return rows
    return rows
