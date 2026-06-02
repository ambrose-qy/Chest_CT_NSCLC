"""
LIDC-IDRI process4: malignancy labels and patient-level dataset splits.

This step consumes the ROI manifests from process3:

    data/processed/tables/lidc_roi_consensus_manifest.csv
    data/processed/tables/lidc_roi_reader_annotation_manifest.csv

Goal:

    Based on radiologists' malignancy scores (1-5 points), construct:

    * binary labels:
        benign    = lesion median malignancy score <= 2
        malignant = lesion median malignancy score >= 4
        score 3 / mixed intermediate lesions are excluded from binary training

    * multi-category labels:
        low risk          = lesion median malignancy score <= 2
        intermediate risk = lesion median malignancy score > 2 and < 4
        high risk         = lesion median malignancy score >= 4

Then split the ROI dataset into train/validation/test partitions at patient
level, so ROIs from the same patient never appear in different splits.

Run from the repository root:

    conda run -n torch-gpu python data_analysis/LIDC_process4.py

Useful options:

    conda run -n torch-gpu python data_analysis/LIDC_process4.py --train-frac 0.70 --val-frac 0.10 --test-frac 0.20
    conda run -n torch-gpu python data_analysis/LIDC_process4.py --seed 2026
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"

CONSENSUS_ROI_TABLE = TABLE_DIR / "lidc_roi_consensus_manifest.csv"
READER_ROI_TABLE = TABLE_DIR / "lidc_roi_reader_annotation_manifest.csv"

LABELED_ROI_TABLE = TABLE_DIR / "lidc_roi_labeled_manifest.csv"
BINARY_SPLIT_TABLE = TABLE_DIR / "lidc_roi_binary_split_manifest.csv"
MULTICLASS_SPLIT_TABLE = TABLE_DIR / "lidc_roi_multiclass_split_manifest.csv"
PATIENT_SPLIT_TABLE = TABLE_DIR / "lidc_patient_split_assignments.csv"
LABEL_CRITERIA_TABLE = TABLE_DIR / "lidc_label_construction_criteria.csv"
LABEL_SUMMARY_TABLE = TABLE_DIR / "lidc_label_split_summary.csv"
LABEL_BALANCE_TABLE = TABLE_DIR / "lidc_split_label_balance_diagnostics.csv"


def log(message):
    print(message)
    sys.stdout.flush()


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


def median(values):
    values = sorted([value for value in values if value is not None])
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stable_float_from_text(text, seed):
    payload = "{}::{}".format(seed, text).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def score_counts(scores):
    counts = Counter(scores)
    return ";".join("{}:{}".format(score, counts[score]) for score in sorted(counts.keys()))


def majority_score(scores):
    if not scores:
        return None
    counts = Counter(scores)
    max_count = max(counts.values())
    candidates = sorted([score for score, count in counts.items() if count == max_count])
    if len(candidates) == 1:
        return candidates[0]

    med = median(scores)
    return min(candidates, key=lambda score: (abs(score - med), score))


def multiclass_label_from_score(score):
    if score is None:
        return ""
    if score <= 2:
        return "low_risk"
    if score >= 4:
        return "high_risk"
    return "intermediate_risk"


def binary_label_from_score(score):
    if score is None:
        return ""
    if score <= 2:
        return "benign"
    if score >= 4:
        return "malignant"
    return ""


def binary_label_id(label):
    if label == "benign":
        return 0
    if label == "malignant":
        return 1
    return ""


def multiclass_label_id(label):
    mapping = {
        "low_risk": 0,
        "intermediate_risk": 1,
        "high_risk": 2,
    }
    return mapping.get(label, "")


def malignant_vote_fraction(scores):
    if not scores:
        return ""
    malignant_votes = sum(1 for score in scores if score >= 4)
    return malignant_votes / float(len(scores))


def benign_vote_fraction(scores):
    if not scores:
        return ""
    benign_votes = sum(1 for score in scores if score <= 2)
    return benign_votes / float(len(scores))


def label_confidence(scores):
    if not scores:
        return "missing"
    label_votes = Counter(multiclass_label_from_score(score) for score in scores)
    top_count = max(label_votes.values())
    if top_count == len(scores):
        return "unanimous"
    if top_count >= 3:
        return "majority_3_plus"
    if top_count >= 2:
        return "partial_majority"
    return "discordant"


def group_reader_scores(reader_rows):
    by_roi = defaultdict(list)
    for row in reader_rows:
        roi_id = clean_value(row.get("roi_id"))
        score = safe_int(row.get("malignancy"))
        if roi_id and score is not None:
            by_roi[roi_id].append(score)
    return by_roi


def construct_labels(consensus_rows, reader_rows):
    scores_by_roi = group_reader_scores(reader_rows)
    labeled = []

    for row in consensus_rows:
        roi_id = clean_value(row.get("roi_id"))
        scores = scores_by_roi.get(roi_id, [])
        med_score = median(scores)
        mean_score = mean(scores)
        majority = majority_score(scores)
        multi_label = multiclass_label_from_score(med_score)
        binary_label = binary_label_from_score(med_score)

        out = dict(row)
        out.update({
            "reader_malignancy_scores": ",".join(str(score) for score in scores),
            "reader_malignancy_score_counts": score_counts(scores),
            "reader_malignancy_score_count": len(scores),
            "median_malignancy_score": med_score if med_score is not None else "",
            "mean_malignancy_score": mean_score if mean_score is not None else "",
            "majority_malignancy_score": majority if majority is not None else "",
            "benign_vote_fraction": benign_vote_fraction(scores),
            "malignant_vote_fraction": malignant_vote_fraction(scores),
            "label_confidence": label_confidence(scores),
            "multiclass_risk_label": multi_label,
            "multiclass_risk_label_id": multiclass_label_id(multi_label),
            "binary_label": binary_label,
            "binary_label_id": binary_label_id(binary_label),
            "binary_label_status": "included" if binary_label else "excluded_intermediate",
        })
        labeled.append(out)

    write_csv(LABELED_ROI_TABLE, labeled)
    return labeled


def patient_label_counts(rows, label_col):
    counts_by_patient = defaultdict(Counter)
    for row in rows:
        patient = clean_value(row.get("PatientID")) or clean_value(row.get("patient_folder"))
        label = clean_value(row.get(label_col))
        if patient and label:
            counts_by_patient[patient][label] += 1
    return counts_by_patient


def target_split_counts(total, train_frac, val_frac, test_frac):
    return {
        "train": total * train_frac,
        "val": total * val_frac,
        "test": total * test_frac,
    }


def normalized_abs(value, target):
    denominator = target if target > 0 else 1.0
    return abs(value - target) / denominator


def normalized_delta(current, added, target):
    denominator = target if target > 0 else 1.0
    before = abs(current - target) / denominator
    after = abs(current + added - target) / denominator
    return after - before


def choose_best_split(
    patient_counts,
    split_label_counts,
    split_patient_counts,
    split_roi_counts,
    target_label_counts,
    target_roi_totals,
    target_patient_counts,
):
    best_split = None
    best_score = None
    patient_roi_count = sum(patient_counts.values())

    for split_name in ["train", "val", "test"]:
        label_delta = 0.0
        for label, count in patient_counts.items():
            current_label = split_label_counts[split_name][label]
            target_label = target_label_counts[split_name].get(label, 0.0)
            label_delta += normalized_delta(current_label, count, target_label)

        roi_delta = normalized_delta(
            split_roi_counts[split_name],
            patient_roi_count,
            target_roi_totals[split_name],
        )
        patient_delta = normalized_delta(
            split_patient_counts[split_name],
            1,
            target_patient_counts[split_name],
        )
        total_penalty = 0.60 * label_delta + 0.30 * roi_delta + 0.10 * patient_delta

        if best_score is None or total_penalty < best_score:
            best_score = total_penalty
            best_split = split_name

    return best_split


def compute_split_stats(assignments, counts_by_patient):
    split_label_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    split_roi_counts = Counter()
    split_patient_counts = Counter()

    for patient, split_name in assignments.items():
        split_label_counts[split_name].update(counts_by_patient[patient])
        split_roi_counts[split_name] += sum(counts_by_patient[patient].values())
        split_patient_counts[split_name] += 1

    return split_label_counts, split_roi_counts, split_patient_counts


def split_assignment_objective(
    assignments,
    counts_by_patient,
    target_label_counts,
    target_roi_totals,
    target_patient_counts,
):
    split_label_counts, split_roi_counts, split_patient_counts = compute_split_stats(
        assignments,
        counts_by_patient,
    )

    score = 0.0
    all_labels = sorted(set(label for counts in target_label_counts.values() for label in counts.keys()))
    for split_name in ["train", "val", "test"]:
        for label in all_labels:
            score += 0.70 * normalized_abs(
                split_label_counts[split_name].get(label, 0),
                target_label_counts[split_name].get(label, 0.0),
            )
        score += 0.20 * normalized_abs(split_roi_counts[split_name], target_roi_totals[split_name])
        score += 0.10 * normalized_abs(split_patient_counts[split_name], target_patient_counts[split_name])

    return score


def improve_assignments_by_local_search(
    assignments,
    counts_by_patient,
    target_label_counts,
    target_roi_totals,
    target_patient_counts,
    seed,
    max_passes=30,
):
    """Refine patient splits to reduce risk-label imbalance without leakage."""
    rng = random.Random(seed)
    patients = list(assignments.keys())
    best_score = split_assignment_objective(
        assignments,
        counts_by_patient,
        target_label_counts,
        target_roi_totals,
        target_patient_counts,
    )

    for _pass_number in range(max_passes):
        improved = False
        rng.shuffle(patients)

        for patient in patients:
            current_split = assignments[patient]
            candidate_splits = [name for name in ["train", "val", "test"] if name != current_split]
            rng.shuffle(candidate_splits)

            for candidate_split in candidate_splits:
                assignments[patient] = candidate_split
                candidate_score = split_assignment_objective(
                    assignments,
                    counts_by_patient,
                    target_label_counts,
                    target_roi_totals,
                    target_patient_counts,
                )
                if candidate_score + 1e-12 < best_score:
                    best_score = candidate_score
                    current_split = candidate_split
                    improved = True
                    break

                assignments[patient] = current_split

        if not improved:
            break

    return assignments


def make_patient_splits(rows, label_col, train_frac, val_frac, test_frac, seed):
    counts_by_patient = patient_label_counts(rows, label_col)
    patients = sorted(counts_by_patient.keys())

    total_label_counts = Counter()
    for counts in counts_by_patient.values():
        total_label_counts.update(counts)

    target_label_counts = {
        "train": Counter({label: count * train_frac for label, count in total_label_counts.items()}),
        "val": Counter({label: count * val_frac for label, count in total_label_counts.items()}),
        "test": Counter({label: count * test_frac for label, count in total_label_counts.items()}),
    }
    total_roi_count = sum(total_label_counts.values())
    target_roi_totals = target_split_counts(total_roi_count, train_frac, val_frac, test_frac)
    target_patient_counts = target_split_counts(len(patients), train_frac, val_frac, test_frac)

    patients_sorted = sorted(
        patients,
        key=lambda patient: (
            -sum(counts_by_patient[patient].values()),
            -max(counts_by_patient[patient].values()),
            stable_float_from_text(patient, seed),
        ),
    )

    split_label_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    split_roi_counts = Counter()
    split_patient_counts = Counter()
    assignments = {}

    for patient in patients_sorted:
        chosen = choose_best_split(
            counts_by_patient[patient],
            split_label_counts,
            split_patient_counts,
            split_roi_counts,
            target_label_counts,
            target_roi_totals,
            target_patient_counts,
        )
        assignments[patient] = chosen
        split_label_counts[chosen].update(counts_by_patient[patient])
        split_roi_counts[chosen] += sum(counts_by_patient[patient].values())
        split_patient_counts[chosen] += 1

    assignments = improve_assignments_by_local_search(
        assignments,
        counts_by_patient,
        target_label_counts,
        target_roi_totals,
        target_patient_counts,
        seed=seed,
    )
    split_label_counts, _split_roi_counts, split_patient_counts = compute_split_stats(
        assignments,
        counts_by_patient,
    )

    return assignments, split_label_counts, split_patient_counts


def apply_split(rows, assignments, split_col):
    output = []
    for row in rows:
        patient = clean_value(row.get("PatientID")) or clean_value(row.get("patient_folder"))
        out = dict(row)
        out[split_col] = assignments.get(patient, "")
        output.append(out)
    return output


def split_and_write_manifests(labeled_rows, train_frac, val_frac, test_frac, seed):
    multiclass_rows = [row for row in labeled_rows if clean_value(row.get("multiclass_risk_label"))]
    binary_rows = [row for row in labeled_rows if clean_value(row.get("binary_label"))]

    multiclass_assignments, multiclass_label_counts, multiclass_patient_counts = make_patient_splits(
        multiclass_rows,
        label_col="multiclass_risk_label",
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    binary_assignments, binary_label_counts, binary_patient_counts = make_patient_splits(
        binary_rows,
        label_col="binary_label",
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    multiclass_split_rows = apply_split(multiclass_rows, multiclass_assignments, "multiclass_split")
    binary_split_rows = apply_split(binary_rows, binary_assignments, "binary_split")

    write_csv(MULTICLASS_SPLIT_TABLE, multiclass_split_rows)
    write_csv(BINARY_SPLIT_TABLE, binary_split_rows)

    patients = sorted(set(list(multiclass_assignments.keys()) + list(binary_assignments.keys())))
    patient_rows = []
    for patient in patients:
        patient_rows.append({
            "PatientID": patient,
            "multiclass_split": multiclass_assignments.get(patient, ""),
            "binary_split": binary_assignments.get(patient, ""),
        })
    write_csv(PATIENT_SPLIT_TABLE, patient_rows)

    return {
        "multiclass_rows": multiclass_split_rows,
        "binary_rows": binary_split_rows,
        "multiclass_label_counts": multiclass_label_counts,
        "binary_label_counts": binary_label_counts,
        "multiclass_patient_counts": multiclass_patient_counts,
        "binary_patient_counts": binary_patient_counts,
    }


def count_by_split_and_label(rows, split_col, label_col):
    counts = defaultdict(Counter)
    for row in rows:
        split_name = clean_value(row.get(split_col)) or "unassigned"
        label = clean_value(row.get(label_col)) or "missing"
        counts[split_name][label] += 1
    return counts


def patient_counts_by_split(rows, split_col):
    patients = defaultdict(set)
    for row in rows:
        split_name = clean_value(row.get(split_col)) or "unassigned"
        patient = clean_value(row.get("PatientID")) or clean_value(row.get("patient_folder"))
        if patient:
            patients[split_name].add(patient)
    return {split_name: len(values) for split_name, values in patients.items()}


def build_label_balance_rows(rows, task_name, split_col, label_col, train_frac, val_frac, test_frac):
    fractions = {
        "train": train_frac,
        "val": val_frac,
        "test": test_frac,
    }
    labels = sorted(set(clean_value(row.get(label_col)) for row in rows if clean_value(row.get(label_col))))
    overall_counts = Counter(clean_value(row.get(label_col)) for row in rows if clean_value(row.get(label_col)))
    split_counts = count_by_split_and_label(rows, split_col, label_col)

    diagnostics = []
    for split_name in ["train", "val", "test"]:
        for label in labels:
            actual = split_counts[split_name].get(label, 0)
            target = overall_counts[label] * fractions[split_name]
            deviation = actual - target
            diagnostics.append({
                "task": task_name,
                "split": split_name,
                "label": label,
                "actual_roi_count": actual,
                "target_roi_count": target,
                "count_deviation": deviation,
                "absolute_count_deviation": abs(deviation),
                "actual_fraction_within_label": actual / float(overall_counts[label]) if overall_counts[label] else "",
                "target_fraction_within_label": fractions[split_name],
            })
    return diagnostics


def write_label_balance_diagnostics(split_result, train_frac, val_frac, test_frac):
    rows = []
    rows.extend(build_label_balance_rows(
        split_result["multiclass_rows"],
        task_name="multiclass",
        split_col="multiclass_split",
        label_col="multiclass_risk_label",
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    ))
    rows.extend(build_label_balance_rows(
        split_result["binary_rows"],
        task_name="binary",
        split_col="binary_split",
        label_col="binary_label",
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    ))
    write_csv(LABEL_BALANCE_TABLE, rows)


def write_label_criteria(train_frac, val_frac, test_frac, seed):
    rows = [
        {
            "criterion": "Malignancy score source",
            "definition": "Uses per-reader LIDC XML malignancy scores from the process3 reader ROI manifest.",
        },
        {
            "criterion": "Lesion-level score",
            "definition": "The primary lesion score is the median of available radiologist malignancy scores.",
        },
        {
            "criterion": "Binary benign label",
            "definition": "Median malignancy score <= 2.",
        },
        {
            "criterion": "Binary malignant label",
            "definition": "Median malignancy score >= 4.",
        },
        {
            "criterion": "Binary exclusion",
            "definition": "Median malignancy score > 2 and < 4 is excluded from binary training/evaluation.",
        },
        {
            "criterion": "Multi-category low risk",
            "definition": "Median malignancy score <= 2.",
        },
        {
            "criterion": "Multi-category intermediate risk",
            "definition": "Median malignancy score > 2 and < 4.",
        },
        {
            "criterion": "Multi-category high risk",
            "definition": "Median malignancy score >= 4.",
        },
        {
            "criterion": "Split unit",
            "definition": "Splits are assigned by PatientID to avoid patient leakage across train/validation/test.",
        },
        {
            "criterion": "Risk-label balance",
            "definition": "Patient-level splits are optimized so each risk level follows the requested train/validation/test proportions as closely as possible.",
        },
        {
            "criterion": "Split fractions",
            "definition": "train={}, val={}, test={}, seed={}.".format(train_frac, val_frac, test_frac, seed),
        },
    ]
    write_csv(LABEL_CRITERIA_TABLE, rows)


def write_summary(labeled_rows, split_result):
    label_counts = Counter(row.get("multiclass_risk_label") or "missing" for row in labeled_rows)
    binary_counts = Counter(row.get("binary_label") or "excluded_intermediate" for row in labeled_rows)
    confidence_counts = Counter(row.get("label_confidence") or "missing" for row in labeled_rows)

    multiclass_split_counts = count_by_split_and_label(
        split_result["multiclass_rows"],
        split_col="multiclass_split",
        label_col="multiclass_risk_label",
    )
    binary_split_counts = count_by_split_and_label(
        split_result["binary_rows"],
        split_col="binary_split",
        label_col="binary_label",
    )
    multiclass_patients = patient_counts_by_split(split_result["multiclass_rows"], "multiclass_split")
    binary_patients = patient_counts_by_split(split_result["binary_rows"], "binary_split")

    rows = []
    rows.append({
        "section": "overall_multiclass",
        "split": "all",
        "label": "low_risk",
        "roi_count": label_counts.get("low_risk", 0),
        "patient_count": "",
    })
    rows.append({
        "section": "overall_multiclass",
        "split": "all",
        "label": "intermediate_risk",
        "roi_count": label_counts.get("intermediate_risk", 0),
        "patient_count": "",
    })
    rows.append({
        "section": "overall_multiclass",
        "split": "all",
        "label": "high_risk",
        "roi_count": label_counts.get("high_risk", 0),
        "patient_count": "",
    })
    rows.append({
        "section": "overall_binary",
        "split": "all",
        "label": "benign",
        "roi_count": binary_counts.get("benign", 0),
        "patient_count": "",
    })
    rows.append({
        "section": "overall_binary",
        "split": "all",
        "label": "malignant",
        "roi_count": binary_counts.get("malignant", 0),
        "patient_count": "",
    })
    rows.append({
        "section": "overall_binary",
        "split": "all",
        "label": "excluded_intermediate",
        "roi_count": binary_counts.get("excluded_intermediate", 0),
        "patient_count": "",
    })

    for split_name in ["train", "val", "test"]:
        for label in ["low_risk", "intermediate_risk", "high_risk"]:
            rows.append({
                "section": "multiclass_split",
                "split": split_name,
                "label": label,
                "roi_count": multiclass_split_counts[split_name].get(label, 0),
                "patient_count": multiclass_patients.get(split_name, 0),
            })
        for label in ["benign", "malignant"]:
            rows.append({
                "section": "binary_split",
                "split": split_name,
                "label": label,
                "roi_count": binary_split_counts[split_name].get(label, 0),
                "patient_count": binary_patients.get(split_name, 0),
            })

    for confidence, count in sorted(confidence_counts.items()):
        rows.append({
            "section": "label_confidence",
            "split": "all",
            "label": confidence,
            "roi_count": count,
            "patient_count": "",
        })

    write_csv(LABEL_SUMMARY_TABLE, rows)


def validate_split_fractions(train_frac, val_frac, test_frac):
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0; got {}".format(total))
    if min(train_frac, val_frac, test_frac) <= 0:
        raise ValueError("Split fractions must all be positive.")


def print_summary(labeled_rows, split_result):
    label_counts = Counter(row.get("multiclass_risk_label") or "missing" for row in labeled_rows)
    binary_counts = Counter(row.get("binary_label") or "excluded_intermediate" for row in labeled_rows)
    multiclass_counts = count_by_split_and_label(
        split_result["multiclass_rows"],
        split_col="multiclass_split",
        label_col="multiclass_risk_label",
    )
    binary_split_counts = count_by_split_and_label(
        split_result["binary_rows"],
        split_col="binary_split",
        label_col="binary_label",
    )

    log("")
    log("LIDC malignancy label construction and dataset split complete")
    log("  Labeled ROI count: {}".format(len(labeled_rows)))
    log("  Multi-category labels: {}".format(dict(label_counts)))
    log("  Binary labels: {}".format(dict(binary_counts)))
    log("  Multiclass split counts: {}".format({key: dict(value) for key, value in multiclass_counts.items()}))
    log("  Binary split counts: {}".format({key: dict(value) for key, value in binary_split_counts.items()}))
    log("  Labeled ROI manifest: {}".format(LABELED_ROI_TABLE))
    log("  Binary split manifest: {}".format(BINARY_SPLIT_TABLE))
    log("  Multi-category split manifest: {}".format(MULTICLASS_SPLIT_TABLE))
    log("  Patient split table: {}".format(PATIENT_SPLIT_TABLE))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Construct LIDC malignancy labels and dataset splits.")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Training split fraction.")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Validation split fraction.")
    parser.add_argument("--test-frac", type=float, default=0.20, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic seed used for patient ordering.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_split_fractions(args.train_frac, args.val_frac, args.test_frac)

    if not CONSENSUS_ROI_TABLE.exists():
        raise FileNotFoundError("Missing process3 consensus ROI manifest: {}".format(CONSENSUS_ROI_TABLE))
    if not READER_ROI_TABLE.exists():
        raise FileNotFoundError("Missing process3 reader ROI manifest: {}".format(READER_ROI_TABLE))

    log("Project root: {}".format(PROJECT_ROOT))
    log("Consensus ROI table: {}".format(CONSENSUS_ROI_TABLE))
    log("Reader ROI table: {}".format(READER_ROI_TABLE))

    consensus_rows = read_csv_rows(CONSENSUS_ROI_TABLE)
    reader_rows = read_csv_rows(READER_ROI_TABLE)
    log("Loaded consensus ROIs: {}".format(len(consensus_rows)))
    log("Loaded reader ROI annotations: {}".format(len(reader_rows)))

    labeled_rows = construct_labels(consensus_rows, reader_rows)
    split_result = split_and_write_manifests(
        labeled_rows,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    write_label_criteria(args.train_frac, args.val_frac, args.test_frac, args.seed)
    write_label_balance_diagnostics(split_result, args.train_frac, args.val_frac, args.test_frac)
    write_summary(labeled_rows, split_result)
    print_summary(labeled_rows, split_result)


if __name__ == "__main__":
    main()
