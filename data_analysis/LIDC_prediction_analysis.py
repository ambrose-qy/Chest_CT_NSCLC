"""
Post-hoc prediction analysis for LIDC-IDRI model outputs.

This script consumes one or more ``test_predictions.csv`` files produced by
``LIDC_evaluate_lightning.py`` and joins them with LIDC ROI, morphology, and
patient-demographic metadata. It writes analysis tables for:

* clinical subgroup performance;
* morphology-feature associations with prediction confidence and errors;
* enriched per-sample prediction records ready for report writing.

The script does not train or evaluate a model. It is safe to run while model
training is still in progress; completed prediction files will be analysed and
missing ones can be added later by rerunning the command.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports" / "lidc_prediction_analysis"
DEFAULT_SEARCH_ROOTS = [
    PROJECT_ROOT / "data" / "processed" / "model_reports" / "lidc_lightning",
    PROJECT_ROOT / "data" / "processed" / "model_results" / "lidc_lightning",
]
DEFAULT_METADATA_TABLES = [
    TABLE_DIR / "lidc_roi_3d_volume_manifest.csv",
    TABLE_DIR / "lidc_roi_binary_2d_slice_manifest.csv",
    TABLE_DIR / "lidc_roi_multiclass_split_manifest.csv",
    TABLE_DIR / "lidc_roi_binary_split_manifest.csv",
    TABLE_DIR / "lidc_roi_labeled_manifest.csv",
    TABLE_DIR / "lidc_roi_consensus_manifest.csv",
]
DEFAULT_DEMOGRAPHICS = TABLE_DIR / "lidc_patient_demographics_model_ready.csv"

MORPHOLOGY_FEATURES = [
    "median_max_diameter_mm",
    "mean_reader_max_diameter_mm",
    "median_x_diameter_mm",
    "median_y_diameter_mm",
    "median_z_diameter_mm",
    "majority_calcification",
    "majority_sphericity",
    "majority_margin",
    "majority_lobulation",
    "majority_spiculation",
    "majority_texture",
    "reader_agreement_category",
    "overall_consistency",
    "reader_count",
    "diameter_approving_reader_count",
    "label_confidence",
]


def log(message):
    print(message, flush=True)


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


def slugify(text):
    text = clean_value(text) or "analysis"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:160]


def discover_prediction_files(search_roots):
    paths = []
    for root in search_roots:
        root = Path(root)
        if root.exists():
            paths.extend(root.rglob("test_predictions.csv"))
    return sorted(set(paths))


def build_roi_metadata_index(metadata_tables):
    index = {}
    source_counts = Counter()
    for table_path in metadata_tables:
        table_path = Path(table_path)
        if not table_path.exists():
            continue
        for row in read_csv_rows(table_path):
            roi_id = clean_value(row.get("roi_id"))
            if not roi_id:
                continue
            if roi_id not in index:
                index[roi_id] = {}
            for key, value in row.items():
                if clean_value(value) != "" and clean_value(index[roi_id].get(key)) == "":
                    index[roi_id][key] = value
            source_counts[table_path.name] += 1
    return index, source_counts


def build_demographics_index(path):
    path = Path(path)
    by_patient_id = {}
    by_patient_folder = {}
    if not path.exists():
        return by_patient_id, by_patient_folder
    for row in read_csv_rows(path):
        patient_id = clean_value(row.get("PatientID"))
        patient_folder = clean_value(row.get("patient_folder"))
        if patient_id:
            by_patient_id[patient_id] = row
        if patient_folder:
            by_patient_folder[patient_folder] = row
    return by_patient_id, by_patient_folder


def prediction_run_name(path):
    path = Path(path)
    parent = path.parent.name
    return parent if parent else path.stem


def probability_columns(row):
    return [key for key in row.keys() if key.startswith("prob_")]


def probability_map(row):
    probs = {}
    for key in probability_columns(row):
        value = safe_float(row.get(key))
        if value is not None:
            probs[key[5:]] = value
    return probs


def confidence_and_true_probability(row):
    probs = probability_map(row)
    if not probs:
        return "", ""
    confidence = max(probs.values())
    true_label = clean_value(row.get("true_label"))
    true_probability = probs.get(true_label, "")
    return confidence, true_probability


def positive_probability(row):
    probs = probability_map(row)
    for name in ("malignant", "high_risk", "1"):
        if name in probs:
            return probs[name]
    if not probs:
        return None
    return probs[sorted(probs.keys())[-1]]


def merge_metadata(prediction_rows, run_name, roi_index, demographics_by_id, demographics_by_folder):
    enriched = []
    for row in prediction_rows:
        out = dict(row)
        out["run_name"] = run_name
        roi_id = clean_value(out.get("roi_id"))
        roi_meta = roi_index.get(roi_id, {})
        for key, value in roi_meta.items():
            if clean_value(out.get(key)) == "" and clean_value(value) != "":
                out[key] = value

        patient_id = clean_value(out.get("patient_id")) or clean_value(out.get("PatientID"))
        patient_folder = clean_value(out.get("patient_folder"))
        demographics = demographics_by_id.get(patient_id) or demographics_by_folder.get(patient_folder) or {}
        for key, value in demographics.items():
            target_key = "demo_{}".format(key)
            if clean_value(value) != "":
                out[target_key] = value
            if clean_value(out.get(key)) == "" and clean_value(value) != "":
                out[key] = value

        true_id = safe_int(out.get("true_label_id"))
        pred_id = safe_int(out.get("predicted_label_id"))
        out["is_correct"] = int(true_id == pred_id) if true_id is not None and pred_id is not None else ""
        out["error_type"] = error_type(out)
        confidence, true_probability = confidence_and_true_probability(out)
        out["confidence"] = confidence
        out["true_label_probability"] = true_probability
        positive_prob = positive_probability(out)
        out["positive_class_probability"] = "" if positive_prob is None else positive_prob
        add_subgroup_fields(out)
        enriched.append(out)
    return enriched


def error_type(row):
    true_label = clean_value(row.get("true_label"))
    predicted_label = clean_value(row.get("predicted_label"))
    if not true_label or not predicted_label:
        return ""
    if true_label == predicted_label:
        return "correct"
    if true_label in ("benign", "low_risk") and predicted_label in ("malignant", "high_risk"):
        return "false_positive_high_risk"
    if true_label in ("malignant", "high_risk") and predicted_label in ("benign", "low_risk"):
        return "false_negative_low_risk"
    return "{}_to_{}".format(true_label, predicted_label)


def add_subgroup_fields(row):
    diameter = safe_float(row.get("median_max_diameter_mm")) or safe_float(row.get("diameter_mm"))
    row["subgroup_size_bin"] = size_bin(diameter)
    row["subgroup_z_location"] = clean_value(row.get("z_location_tertile")) or "missing"
    row["subgroup_image_side"] = clean_value(row.get("image_side")) or clean_value(row.get("patient_side")) or "missing"
    row["subgroup_image_ap_position"] = clean_value(row.get("image_ap_position")) or "missing"
    row["subgroup_reader_agreement"] = clean_value(row.get("reader_agreement_category")) or "missing"
    row["subgroup_overall_consistency"] = clean_value(row.get("overall_consistency")) or "missing"
    row["subgroup_label_confidence"] = clean_value(row.get("label_confidence")) or "missing"
    row["subgroup_true_label"] = clean_value(row.get("true_label")) or "missing"

    age = first_float(row, [
        "PatientAge_years",
        "age_years",
        "demo_PatientAge_years",
        "demo_age_years",
        "age_years_imputed",
        "demo_age_years_imputed",
        "PatientAge_years_imputed",
        "demo_PatientAge_years_imputed",
    ])
    row["subgroup_age_bin"] = age_bin(age)

    sex = (
        clean_value(row.get("PatientSex"))
        or clean_value(row.get("sex"))
        or clean_value(row.get("demo_PatientSex"))
        or clean_value(row.get("demo_sex"))
        or clean_value(row.get("sex_model_ready"))
        or clean_value(row.get("demo_sex_model_ready"))
    )
    row["subgroup_sex"] = sex if sex else "missing"


def first_float(row, keys):
    for key in keys:
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def size_bin(value):
    if value is None:
        return "missing"
    if value < 6.0:
        return "lt_6mm"
    if value < 10.0:
        return "6_to_10mm"
    if value < 20.0:
        return "10_to_20mm"
    return "ge_20mm"


def age_bin(value):
    if value is None:
        return "missing"
    if value < 60:
        return "lt_60"
    if value < 70:
        return "60_to_69"
    if value < 80:
        return "70_to_79"
    return "ge_80"


def class_names_from_rows(rows):
    names = []
    for row in rows:
        for key in probability_columns(row):
            name = key[5:]
            if name not in names:
                names.append(name)
    if names:
        return names
    labels = []
    for row in rows:
        for key in ("true_label", "predicted_label"):
            value = clean_value(row.get(key))
            if value and value not in labels:
                labels.append(value)
    return labels


def labels_and_predictions(rows, class_names):
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    labels = []
    predictions = []
    probability_matrix = []
    for row in rows:
        true_label = clean_value(row.get("true_label"))
        predicted_label = clean_value(row.get("predicted_label"))
        if true_label not in name_to_id or predicted_label not in name_to_id:
            continue
        labels.append(name_to_id[true_label])
        predictions.append(name_to_id[predicted_label])
        probs = probability_map(row)
        if probs:
            probability_matrix.append([probs.get(name, 0.0) for name in class_names])
    probs_array = np.asarray(probability_matrix, dtype=np.float64) if probability_matrix else None
    return np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64), probs_array


def metric_row(rows, class_names, prefix=None):
    labels, predictions, probabilities = labels_and_predictions(rows, class_names)
    total = int(labels.size)
    out = {
        "sample_count": total,
        "accuracy": "",
        "macro_precision": "",
        "macro_recall": "",
        "macro_f1": "",
        "auc_roc": "",
    }
    if prefix:
        out.update(prefix)
    if total == 0:
        return out

    out["accuracy"] = float((labels == predictions).sum() / float(total))
    per_class = []
    for class_id, class_name in enumerate(class_names):
        tp = int(((labels == class_id) & (predictions == class_id)).sum())
        fp = int(((labels != class_id) & (predictions == class_id)).sum())
        fn = int(((labels == class_id) & (predictions != class_id)).sum())
        support = int((labels == class_id).sum())
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        recall = tp / float(tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) else 0.0
        per_class.append((precision, recall, f1))
        out["support_{}".format(class_name)] = support
        out["precision_{}".format(class_name)] = precision
        out["recall_{}".format(class_name)] = recall
        out["f1_{}".format(class_name)] = f1

    out["macro_precision"] = float(np.mean([item[0] for item in per_class]))
    out["macro_recall"] = float(np.mean([item[1] for item in per_class]))
    out["macro_f1"] = float(np.mean([item[2] for item in per_class]))
    out["auc_roc"] = auc_score(labels, probabilities, class_names)
    return out


def auc_score(labels, probabilities, class_names):
    if probabilities is None or probabilities.shape[0] != labels.size:
        return ""
    if len(set(labels.tolist())) < 2:
        return ""
    try:
        from sklearn.metrics import roc_auc_score
        if len(class_names) == 2:
            return float(roc_auc_score(labels, probabilities[:, 1]))
        return float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except Exception:
        return ""


def subgroup_performance(rows):
    variables = [
        "subgroup_size_bin",
        "subgroup_z_location",
        "subgroup_image_side",
        "subgroup_image_ap_position",
        "subgroup_reader_agreement",
        "subgroup_overall_consistency",
        "subgroup_label_confidence",
        "subgroup_age_bin",
        "subgroup_sex",
        "subgroup_true_label",
    ]
    out = []
    rows_by_run = group_rows(rows, "run_name")
    for run_name, run_rows in rows_by_run.items():
        class_names = class_names_from_rows(run_rows)
        out.append(metric_row(run_rows, class_names, {"run_name": run_name, "subgroup_variable": "overall", "subgroup_value": "all"}))
        for variable in variables:
            for value, value_rows in sorted(group_rows(run_rows, variable).items()):
                out.append(metric_row(value_rows, class_names, {
                    "run_name": run_name,
                    "subgroup_variable": variable,
                    "subgroup_value": value,
                }))
    return out


def group_rows(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[clean_value(row.get(key)) or "missing"].append(row)
    return grouped


def morphology_category_summary(rows):
    features = [
        "majority_calcification",
        "majority_sphericity",
        "majority_margin",
        "majority_lobulation",
        "majority_spiculation",
        "majority_texture",
        "reader_agreement_category",
        "overall_consistency",
        "label_confidence",
        "subgroup_size_bin",
    ]
    out = []
    for run_name, run_rows in group_rows(rows, "run_name").items():
        class_names = class_names_from_rows(run_rows)
        for feature in features:
            for value, value_rows in sorted(group_rows(run_rows, feature).items()):
                summary = metric_row(value_rows, class_names, {
                    "run_name": run_name,
                    "feature": feature,
                    "feature_value": value,
                })
                summary.update(confidence_summary(value_rows))
                summary["error_count"] = sum(1 for row in value_rows if clean_value(row.get("is_correct")) == "0")
                summary["error_rate"] = summary["error_count"] / float(len(value_rows)) if value_rows else ""
                out.append(summary)
    return out


def confidence_summary(rows):
    confidences = [safe_float(row.get("confidence")) for row in rows]
    true_probs = [safe_float(row.get("true_label_probability")) for row in rows]
    positive_probs = [safe_float(row.get("positive_class_probability")) for row in rows]
    return {
        "mean_confidence": mean([value for value in confidences if value is not None]),
        "mean_true_label_probability": mean([value for value in true_probs if value is not None]),
        "mean_positive_class_probability": mean([value for value in positive_probs if value is not None]),
    }


def morphology_numeric_correlations(rows):
    features = [
        "median_max_diameter_mm",
        "mean_reader_max_diameter_mm",
        "median_x_diameter_mm",
        "median_y_diameter_mm",
        "median_z_diameter_mm",
        "reader_count",
        "diameter_approving_reader_count",
        "majority_calcification",
        "majority_sphericity",
        "majority_margin",
        "majority_lobulation",
        "majority_spiculation",
        "majority_texture",
        "median_malignancy_score",
        "mean_malignancy_score",
    ]
    targets = [
        ("is_correct", "prediction_correctness"),
        ("confidence", "confidence"),
        ("true_label_probability", "true_label_probability"),
        ("positive_class_probability", "positive_class_probability"),
    ]
    out = []
    for run_name, run_rows in group_rows(rows, "run_name").items():
        for feature in features:
            x = [safe_float(row.get(feature)) for row in run_rows]
            for target_key, target_name in targets:
                y = [safe_float(row.get(target_key)) for row in run_rows]
                corr, count = pearson_corr(x, y)
                out.append({
                    "run_name": run_name,
                    "feature": feature,
                    "target": target_name,
                    "sample_count": count,
                    "pearson_correlation": corr,
                })
    return out


def error_distribution(rows):
    out = []
    for run_name, run_rows in group_rows(rows, "run_name").items():
        total = len(run_rows)
        for error, error_rows in sorted(group_rows(run_rows, "error_type").items()):
            out.append({
                "run_name": run_name,
                "error_type": error,
                "sample_count": len(error_rows),
                "fraction": len(error_rows) / float(total) if total else "",
                "mean_confidence": confidence_summary(error_rows)["mean_confidence"],
                "mean_true_label_probability": confidence_summary(error_rows)["mean_true_label_probability"],
            })
    return out


def mean(values):
    values = [value for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return ""
    return float(sum(values) / float(len(values)))


def pearson_corr(x_values, y_values):
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return "", len(pairs)
    x = np.asarray([item[0] for item in pairs], dtype=np.float64)
    y = np.asarray([item[1] for item in pairs], dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return "", len(pairs)
    return float(np.corrcoef(x, y)[0, 1]), len(pairs)


def analyse_prediction_file(path, roi_index, demographics_by_id, demographics_by_folder, output_dir):
    path = Path(path)
    run_name = prediction_run_name(path)
    rows = read_csv_rows(path)
    enriched = merge_metadata(rows, run_name, roi_index, demographics_by_id, demographics_by_folder)
    run_output = Path(output_dir) / slugify(run_name)
    run_output.mkdir(parents=True, exist_ok=True)

    write_csv(run_output / "enriched_predictions.csv", enriched)
    write_csv(run_output / "subgroup_performance.csv", subgroup_performance(enriched))
    write_csv(run_output / "morphology_category_summary.csv", morphology_category_summary(enriched))
    write_csv(run_output / "morphology_numeric_correlations.csv", morphology_numeric_correlations(enriched))
    write_csv(run_output / "error_distribution.csv", error_distribution(enriched))

    metadata = {
        "run_name": run_name,
        "prediction_file": str(path),
        "sample_count": len(enriched),
        "class_names": class_names_from_rows(enriched),
        "outputs": {
            "enriched_predictions": str(run_output / "enriched_predictions.csv"),
            "subgroup_performance": str(run_output / "subgroup_performance.csv"),
            "morphology_category_summary": str(run_output / "morphology_category_summary.csv"),
            "morphology_numeric_correlations": str(run_output / "morphology_numeric_correlations.csv"),
            "error_distribution": str(run_output / "error_distribution.csv"),
        },
    }
    write_json(run_output / "analysis_metadata.json", metadata)
    return metadata


def combine_outputs(run_payloads, output_dir):
    combined = []
    for payload in run_payloads:
        path = Path(payload["outputs"]["subgroup_performance"])
        if path.exists():
            combined.extend(read_csv_rows(path))
    write_csv(Path(output_dir) / "all_runs_subgroup_performance.csv", combined)

    combined = []
    for payload in run_payloads:
        path = Path(payload["outputs"]["morphology_category_summary"])
        if path.exists():
            combined.extend(read_csv_rows(path))
    write_csv(Path(output_dir) / "all_runs_morphology_category_summary.csv", combined)

    combined = []
    for payload in run_payloads:
        path = Path(payload["outputs"]["morphology_numeric_correlations"])
        if path.exists():
            combined.extend(read_csv_rows(path))
    write_csv(Path(output_dir) / "all_runs_morphology_numeric_correlations.csv", combined)

    combined = []
    for payload in run_payloads:
        path = Path(payload["outputs"]["error_distribution"])
        if path.exists():
            combined.extend(read_csv_rows(path))
    write_csv(Path(output_dir) / "all_runs_error_distribution.csv", combined)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyse LIDC prediction files by subgroup and morphology.")
    parser.add_argument("--predictions", type=Path, nargs="*", default=None, help="One or more test_predictions.csv files.")
    parser.add_argument("--search-root", type=Path, nargs="*", default=DEFAULT_SEARCH_ROOTS, help="Roots searched when --predictions is omitted.")
    parser.add_argument("--metadata-tables", type=Path, nargs="*", default=DEFAULT_METADATA_TABLES, help="ROI metadata tables used for joining prediction rows.")
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS, help="Patient demographics table.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = args.predictions if args.predictions else discover_prediction_files(args.search_root)
    prediction_files = [Path(path) for path in prediction_files if Path(path).exists()]
    if not prediction_files:
        raise FileNotFoundError("No test_predictions.csv files found. Run LIDC_evaluate_lightning.py first or pass --predictions.")

    roi_index, source_counts = build_roi_metadata_index(args.metadata_tables)
    demographics_by_id, demographics_by_folder = build_demographics_index(args.demographics)
    payloads = []
    for prediction_file in prediction_files:
        log("Analysing {}".format(prediction_file))
        payloads.append(analyse_prediction_file(
            prediction_file,
            roi_index,
            demographics_by_id,
            demographics_by_folder,
            args.output_dir,
        ))

    combine_outputs(payloads, args.output_dir)
    write_json(args.output_dir / "analysis_index.json", {
        "prediction_files": [str(path) for path in prediction_files],
        "run_count": len(payloads),
        "roi_metadata_rows_by_source": dict(source_counts),
        "roi_metadata_index_size": len(roi_index),
        "demographics_patient_id_count": len(demographics_by_id),
        "demographics_patient_folder_count": len(demographics_by_folder),
        "runs": payloads,
    })
    log("Analysis complete: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
