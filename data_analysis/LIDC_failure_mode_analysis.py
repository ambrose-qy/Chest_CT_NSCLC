"""
Consolidate Grad-CAM error cases and restore calcification evidence.

The current model manifests may not contain majority calcification values.
This script joins Grad-CAM failure cases to the reader-level ROI annotation
manifest, derives calcification consensus, and writes report-ready summaries.
No model training or inference is performed.
"""

from __future__ import print_function

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"
DEFAULT_MODEL_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "lidc_lightning"
    / "weekly_report3_final_assets"
    / "weekly_report3_model_summary.csv"
)
DEFAULT_READER_ANNOTATIONS = TABLE_DIR / "lidc_roi_reader_annotation_manifest.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "model_reports" / "lidc_failure_mode_analysis"
)

CALCIFICATION_LABELS = {
    1: "popcorn",
    2: "laminated",
    3: "solid",
    4: "non_central",
    5: "central",
    6: "absent",
}


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_code(value):
    try:
        code = int(float(str(value).strip()))
    except Exception:
        return None
    return code if code in CALCIFICATION_LABELS else None


def majority_code(codes):
    if not codes:
        return None
    counts = Counter(codes)
    return sorted(counts, key=lambda code: (-counts[code], code))[0]


def build_calcification_index(rows):
    scores_by_roi = defaultdict(list)
    for row in rows:
        roi_id = str(row.get("roi_id", "")).strip()
        code = safe_code(row.get("calcification"))
        if roi_id and code is not None:
            scores_by_roi[roi_id].append(code)

    index = {}
    for roi_id, codes in scores_by_roi.items():
        present_count = sum(code in (1, 2, 3, 4, 5) for code in codes)
        absent_count = sum(code == 6 for code in codes)
        if present_count > absent_count:
            status = "calcified"
        elif absent_count > present_count:
            status = "non_calcified"
        else:
            status = "mixed_or_tied"
        majority = majority_code(codes)
        index[roi_id] = {
            "reader_calcification_scores": ",".join(str(code) for code in codes),
            "majority_calcification": majority if majority is not None else "",
            "majority_calcification_label": CALCIFICATION_LABELS.get(majority, ""),
            "calcification_present_votes": present_count,
            "calcification_absent_votes": absent_count,
            "calcification_present_vote_fraction": present_count / float(len(codes)),
            "calcification_status": status,
        }
    return index


def standard_failure_patterns(row):
    true_label = str(row.get("true_label", "")).strip()
    predicted_label = str(row.get("predicted_label", "")).strip()
    patterns = ["misclassified"]
    if true_label in ("benign", "low_risk") and predicted_label in ("malignant", "high_risk"):
        patterns.append("false_positive_as_malignant_or_high_risk")
    if true_label in ("malignant", "high_risk") and predicted_label in ("benign", "low_risk"):
        patterns.append("false_negative_as_benign_or_low_risk")
    if row.get("calcification_status") == "calcified":
        patterns.append("calcified_nodule_misclassified")
        if "false_positive_as_malignant_or_high_risk" in patterns:
            patterns.append("calcified_nodule_flagged_as_malignant_or_high_risk")
    try:
        if float(row.get("median_max_diameter_mm", "")) < 6.0:
            patterns.append("small_nodule_lt_6mm")
    except Exception:
        pass
    if str(row.get("overall_consistency", "")).strip().lower() == "low":
        patterns.append("low_reader_consistency")
    return patterns


def consolidate_failures(model_rows, calcification_index):
    consolidated = []
    for model_row in model_rows:
        failure_path = Path(model_row["run_dir"]) / "grad_cam" / "grad_cam_failure_cases.csv"
        if not failure_path.exists():
            continue
        for row in read_csv_rows(failure_path):
            out = {
                "input_dim": model_row.get("input_dim", ""),
                "model": model_row.get("model", ""),
                "task": model_row.get("task", ""),
                "run_name": model_row.get("run_name", ""),
                "failure_case_path": str(failure_path),
            }
            out.update(row)
            out.update(calcification_index.get(str(row.get("roi_id", "")).strip(), {
                "reader_calcification_scores": "",
                "majority_calcification": "",
                "majority_calcification_label": "",
                "calcification_present_votes": "",
                "calcification_absent_votes": "",
                "calcification_present_vote_fraction": "",
                "calcification_status": "unknown",
            }))
            out["standard_failure_patterns"] = ";".join(standard_failure_patterns(out))
            consolidated.append(out)
    return consolidated


def summarize_patterns(rows):
    counts = Counter()
    model_counts = Counter()
    for row in rows:
        key = (row["input_dim"], row["model"], row["task"])
        model_counts[key] += 1
        for pattern in str(row.get("standard_failure_patterns", "")).split(";"):
            if pattern:
                counts[key + (pattern,)] += 1
    output = []
    for key, count in sorted(counts.items()):
        total = model_counts[key[:3]]
        output.append({
            "input_dim": key[0],
            "model": key[1],
            "task": key[2],
            "failure_pattern": key[3],
            "count": count,
            "analysed_failure_count": total,
            "fraction": count / float(total) if total else "",
        })
    return output


def model_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["input_dim"], row["model"], row["task"])].append(row)
    output = []
    for key, items in sorted(grouped.items()):
        calcified = [row for row in items if row.get("calcification_status") == "calcified"]
        calcified_high_risk = [
            row
            for row in items
            if "calcified_nodule_flagged_as_malignant_or_high_risk"
            in str(row.get("standard_failure_patterns", ""))
        ]
        output.append({
            "input_dim": key[0],
            "model": key[1],
            "task": key[2],
            "analysed_failure_count": len(items),
            "calcified_failure_count": len(calcified),
            "calcified_flagged_as_malignant_or_high_risk_count": len(calcified_high_risk),
            "small_nodule_failure_count": sum(
                "small_nodule_lt_6mm" in str(row.get("standard_failure_patterns", ""))
                for row in items
            ),
            "low_reader_consistency_failure_count": sum(
                "low_reader_consistency" in str(row.get("standard_failure_patterns", ""))
                for row in items
            ),
        })
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Consolidate LIDC Grad-CAM failure modes.")
    parser.add_argument("--model-summary", type=Path, default=DEFAULT_MODEL_SUMMARY)
    parser.add_argument("--reader-annotations", type=Path, default=DEFAULT_READER_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_rows = read_csv_rows(args.model_summary)
    reader_rows = read_csv_rows(args.reader_annotations)
    calcification_index = build_calcification_index(reader_rows)
    failures = consolidate_failures(model_rows, calcification_index)
    pattern_rows = summarize_patterns(failures)
    model_rows_out = model_summary(failures)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all_grad_cam_failure_cases_enriched.csv", failures)
    write_csv(args.output_dir / "failure_pattern_summary.csv", pattern_rows)
    write_csv(args.output_dir / "model_failure_summary.csv", model_rows_out)
    payload = {
        "model_count": len(model_rows_out),
        "failure_case_count": len(failures),
        "roi_calcification_metadata_count": len(calcification_index),
        "calcified_failure_count": sum(
            row.get("calcification_status") == "calcified" for row in failures
        ),
        "calcified_flagged_as_malignant_or_high_risk_count": sum(
            "calcified_nodule_flagged_as_malignant_or_high_risk"
            in str(row.get("standard_failure_patterns", ""))
            for row in failures
        ),
        "output_dir": str(args.output_dir),
    }
    with (args.output_dir / "failure_analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
