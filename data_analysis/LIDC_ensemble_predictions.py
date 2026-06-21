"""
Create simple probability-averaging ensembles from completed LIDC predictions.

Run after missing predictions have been backfilled:

    C:\\Users\\Ambro\\.conda\\envs\\torch-gpu\\python.exe data_analysis\\LIDC_ensemble_predictions.py
"""

from __future__ import print_function

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lidc_lightning_utils import binary_metrics, confusion_matrix_rows, multiclass_metrics, roc_curve_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOT = PROJECT_ROOT / "data" / "processed" / "model_results" / "lidc_lightning"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports" / "lidc_ensemble"


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


def discover_prediction_files(search_roots):
    paths = []
    for root in search_roots:
        root = Path(root)
        if root.exists():
            paths.extend(root.rglob("test_predictions.csv"))
    return sorted(set(paths))


def run_name(path):
    return Path(path).parent.name


def probability_columns(row):
    return [key for key in row.keys() if key.startswith("prob_")]


def class_names(rows):
    names = []
    for row in rows:
        for key in probability_columns(row):
            name = key[5:]
            if name not in names:
                names.append(name)
    return names


def task_from_classes(classes):
    if classes == ["benign", "malignant"] or len(classes) == 2:
        return "binary"
    return "multiclass"


def prediction_payload(path):
    rows = read_csv_rows(path)
    classes = class_names(rows)
    if not rows or not classes:
        return None
    by_roi = {}
    for row in rows:
        roi_id = clean_value(row.get("roi_id"))
        if not roi_id:
            continue
        probs = [safe_float(row.get("prob_{}".format(name))) for name in classes]
        if any(value is None for value in probs):
            continue
        by_roi[roi_id] = {
            "row": row,
            "label": int(float(row.get("true_label_id"))),
            "probabilities": np.asarray(probs, dtype=np.float64),
        }
    return {
        "path": Path(path),
        "run_name": run_name(path),
        "classes": classes,
        "task": task_from_classes(classes),
        "by_roi": by_roi,
    }


def group_payloads(payloads):
    grouped = defaultdict(list)
    for payload in payloads:
        grouped[(payload["task"], tuple(payload["classes"]))].append(payload)
    return grouped


def default_weight(payload, metric_name):
    run_dir = payload["path"].parent
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        return 1.0
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        value = metrics.get(metric_name) or metrics.get("test_{}".format(metric_name))
        return max(float(value), 1e-6)
    except Exception:
        return 1.0


def build_ensemble(payloads, classes, task, weighting):
    if len(payloads) < 2:
        return None
    roi_sets = [set(payload["by_roi"].keys()) for payload in payloads]
    common_rois = sorted(set.intersection(*roi_sets)) if roi_sets else []
    if not common_rois:
        return None
    metric_name = "auc_roc" if task == "binary" else "f1"
    weights = np.asarray([
        default_weight(payload, metric_name) if weighting == "metric" else 1.0
        for payload in payloads
    ], dtype=np.float64)
    weights = weights / np.clip(weights.sum(), 1e-12, None)

    rows = []
    labels = []
    probabilities = []
    for roi_id in common_rois:
        label_values = [payload["by_roi"][roi_id]["label"] for payload in payloads]
        if len(set(label_values)) != 1:
            continue
        stacked = np.stack([payload["by_roi"][roi_id]["probabilities"] for payload in payloads], axis=0)
        avg = (stacked * weights.reshape(-1, 1)).sum(axis=0)
        avg = avg / np.clip(avg.sum(), 1e-12, None)
        pred_id = int(avg.argmax())
        base_row = payloads[0]["by_roi"][roi_id]["row"]
        out = {
            "roi_id": roi_id,
            "patient_id": base_row.get("patient_id", ""),
            "true_label_id": int(label_values[0]),
            "true_label": classes[int(label_values[0])] if int(label_values[0]) < len(classes) else str(label_values[0]),
            "predicted_label_id": pred_id,
            "predicted_label": classes[pred_id],
            "ensemble_member_count": len(payloads),
            "ensemble_weighting": weighting,
            "ensemble_members": ";".join(payload["run_name"] for payload in payloads),
            "ensemble_weights": ";".join("{:.6g}".format(value) for value in weights.tolist()),
        }
        for idx, name in enumerate(classes):
            out["prob_{}".format(name)] = float(avg[idx])
        rows.append(out)
        labels.append(int(label_values[0]))
        probabilities.append(avg)
    if not rows:
        return None
    probabilities = np.stack(probabilities, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    if task == "binary":
        metrics = binary_metrics(labels, probabilities[:, 1])
        roc_rows = roc_curve_rows(labels, probabilities[:, 1])
    else:
        metrics = multiclass_metrics(labels, probabilities)
        roc_rows = []
    return {
        "rows": rows,
        "labels": labels,
        "probabilities": probabilities,
        "metrics": metrics,
        "roc_rows": roc_rows,
        "classes": classes,
        "task": task,
        "weighting": weighting,
        "members": [payload["run_name"] for payload in payloads],
        "weights": weights.tolist(),
    }


def write_ensemble(name, payload, output_dir):
    run_dir = Path(output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "ensemble_predictions.csv", payload["rows"])
    write_json(run_dir / "ensemble_metrics.json", payload["metrics"])
    write_csv(run_dir / "ensemble_confusion_matrix.csv", confusion_matrix_rows(payload["labels"], payload["probabilities"], payload["classes"]))
    if payload["roc_rows"]:
        write_csv(run_dir / "ensemble_roc_curve.csv", payload["roc_rows"])
    metadata = {
        "name": name,
        "task": payload["task"],
        "classes": payload["classes"],
        "members": payload["members"],
        "weights": payload["weights"],
        "weighting": payload["weighting"],
        "sample_count": len(payload["rows"]),
        "outputs": {
            "predictions": str(run_dir / "ensemble_predictions.csv"),
            "metrics": str(run_dir / "ensemble_metrics.json"),
            "confusion_matrix": str(run_dir / "ensemble_confusion_matrix.csv"),
            "roc_curve": str(run_dir / "ensemble_roc_curve.csv") if payload["roc_rows"] else "",
        },
    }
    write_json(run_dir / "ensemble_metadata.json", metadata)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Average LIDC model prediction probabilities by task.")
    parser.add_argument("--predictions", type=Path, nargs="*", default=None)
    parser.add_argument("--search-root", type=Path, nargs="*", default=[DEFAULT_SEARCH_ROOT])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weighting", choices=["equal", "metric"], default="equal")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prediction_files = args.predictions if args.predictions else discover_prediction_files(args.search_root)
    payloads = []
    for path in prediction_files:
        payload = prediction_payload(path)
        if payload is not None:
            payloads.append(payload)
    if not payloads:
        raise FileNotFoundError("No usable test_predictions.csv files found. Run LIDC_finalize_missing_outputs.py first.")

    index = []
    for (task, class_tuple), group in sorted(group_payloads(payloads).items()):
        result = build_ensemble(group, list(class_tuple), task, args.weighting)
        if result is None:
            continue
        name = "{}_{}_{}_models".format(task, args.weighting, len(group))
        index.append(write_ensemble(name, result, args.output_dir))
        print("Wrote ensemble {}".format(name))
    write_json(args.output_dir / "ensemble_index.json", {"ensembles": index, "ensemble_count": len(index)})
    print("Ensemble index: {}".format(args.output_dir / "ensemble_index.json"))


if __name__ == "__main__":
    main()
