"""
LIDC-IDRI process6: data-quality issue and imputation planning report.

This step turns the known missing-data and outlier checks into explicit report
tables. It also writes a conservative model-ready demographics table:

* age is imputed with cohort mean/median values and a missingness flag;
* DICOM PatientSex is not guessed from image or lesion similarity. Missing sex
  is kept as "Unknown" with a missingness flag, because fabricating demographic
  labels can bias analysis and fairness reporting.

Run from the repository root:

    conda run -n torch-gpu python data_analysis/LIDC_process6.py
"""

from __future__ import print_function

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"

DEMOGRAPHICS_TABLE = TABLE_DIR / "lidc_patient_demographics.csv"
FEATURE_TABLE = TABLE_DIR / "lidc_nodule_size_location_morphology.csv"
ROI_LABEL_TABLE = TABLE_DIR / "lidc_roi_labeled_manifest.csv"
ROI_3D_QC_TABLE = TABLE_DIR / "lidc_roi_3d_preprocessing_qc.csv"
ROI_3D_OUTLIER_TABLE = TABLE_DIR / "lidc_roi_3d_outlier_report.csv"

QUALITY_SUMMARY_TABLE = TABLE_DIR / "lidc_data_quality_issue_summary.csv"
DEMOGRAPHICS_MODEL_READY_TABLE = TABLE_DIR / "lidc_patient_demographics_model_ready.csv"
DEMOGRAPHICS_IMPUTATION_PLAN_TABLE = TABLE_DIR / "lidc_demographics_imputation_plan.csv"
ANNOTATION_OUTLIER_TABLE = TABLE_DIR / "lidc_annotation_outlier_rows.csv"
ROI_QUALITY_ISSUE_TABLE = TABLE_DIR / "lidc_roi_quality_issue_rows.csv"


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


def read_csv_rows(path, missing_ok=False):
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
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


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def median(values):
    values = sorted([value for value in values if value is not None])
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def fraction(count, total):
    return count / float(total) if total else ""


def patient_sex_category(value):
    text = clean_value(value).upper()
    if text in ("M", "F", "O"):
        return text
    return "Unknown"


def build_demographics_model_ready(demographics):
    ages = [safe_float(row.get("PatientAge_years")) for row in demographics]
    available_ages = [age for age in ages if age is not None]
    age_mean = mean(available_ages)
    age_median = median(available_ages)

    output_rows = []
    for row in demographics:
        age = safe_float(row.get("PatientAge_years"))
        sex = patient_sex_category(row.get("PatientSex"))
        age_missing = age is None
        sex_missing = sex == "Unknown"

        out = dict(row)
        out.update({
            "PatientAge_missing": int(age_missing),
            "PatientAge_mean_imputed": age if age is not None else age_mean,
            "PatientAge_median_imputed": age if age is not None else age_median,
            "PatientAge_imputation_method": "observed" if age is not None else "cohort_mean_or_median_with_missingness_flag",
            "PatientSex_model_category": sex,
            "PatientSex_missing": int(sex_missing),
            "PatientSex_imputation_method": "observed" if not sex_missing else "Unknown_category_no_similarity_guess",
        })
        output_rows.append(out)

    return output_rows, age_mean, age_median


def build_imputation_plan(demographics, age_mean, age_median):
    total = len(demographics)
    age_missing = sum(1 for row in demographics if safe_float(row.get("PatientAge_years")) is None)
    sex_missing = sum(1 for row in demographics if patient_sex_category(row.get("PatientSex")) == "Unknown")
    sex_counts = Counter(patient_sex_category(row.get("PatientSex")) for row in demographics)

    return [
        {
            "field": "PatientAge_years",
            "missing_count": age_missing,
            "missing_fraction": fraction(age_missing, total),
            "recommended_for_descriptive_report": "Report observed age distribution and missingness; do not hide missingness.",
            "recommended_for_model_covariate": "Use median or mean imputation plus PatientAge_missing flag.",
            "default_model_value": age_median,
            "alternative_model_value": age_mean,
            "not_recommended": "Complete-case analysis only, because it would keep only {} of {} patients.".format(total - age_missing, total),
        },
        {
            "field": "PatientSex",
            "missing_count": sex_missing,
            "missing_fraction": fraction(sex_missing, total),
            "observed_distribution": ";".join("{}:{}".format(key, sex_counts[key]) for key in sorted(sex_counts.keys())),
            "recommended_for_descriptive_report": "Report M/F/Unknown counts explicitly.",
            "recommended_for_model_covariate": "Use categorical values M, F, O, Unknown plus PatientSex_missing flag.",
            "default_model_value": "Unknown",
            "alternative_model_value": "",
            "not_recommended": "Do not infer sex/gender from lesion similarity or image appearance without validated external labels.",
        },
    ]


def annotation_issue_flags(row):
    flags = []
    if safe_float(row.get("x_spacing_mm")) is None or safe_float(row.get("y_spacing_mm")) is None:
        flags.append("missing_spacing")
    diameter = safe_float(row.get("max_diameter_mm"))
    if diameter is None:
        flags.append("missing_diameter")
    elif diameter <= 0:
        flags.append("diameter_le_zero")
    elif diameter > 60:
        flags.append("diameter_gt_60mm")
    if safe_float(row.get("malignancy")) is None:
        flags.append("missing_malignancy_score")
    return flags


def build_annotation_outlier_rows(features):
    rows = []
    for row in features:
        flags = annotation_issue_flags(row)
        if not flags:
            continue
        rows.append({
            "issue_flags": ";".join(flags),
            "patient_folder": row.get("patient_folder", ""),
            "PatientID": row.get("PatientID", ""),
            "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
            "xml_name": row.get("xml_name", ""),
            "reader_id": row.get("reader_id", ""),
            "nodule_id": row.get("nodule_id", ""),
            "max_diameter_mm": row.get("max_diameter_mm", ""),
            "x_spacing_mm": row.get("x_spacing_mm", ""),
            "y_spacing_mm": row.get("y_spacing_mm", ""),
            "malignancy": row.get("malignancy", ""),
            "recommended_handling": annotation_recommendation(flags),
        })
    return rows


def annotation_recommendation(flags):
    recommendations = []
    if "missing_spacing" in flags:
        recommendations.append("recover spacing from matching DICOM series or exclude from size/location summaries")
    if "diameter_le_zero" in flags or "missing_diameter" in flags:
        recommendations.append("exclude from diameter-based analysis")
    if "diameter_gt_60mm" in flags:
        recommendations.append("manual review; keep only if confirmed as true lesion extent")
    if "missing_malignancy_score" in flags:
        recommendations.append("exclude from label construction; keep for unlabeled exploration only")
    return "; ".join(recommendations)


def roi_issue_flags(row):
    flags = []
    if clean_value(row.get("multiclass_risk_label")) == "":
        flags.append("missing_multiclass_label")
    if clean_value(row.get("binary_label")) == "":
        flags.append("binary_excluded_intermediate_or_missing")
    if clean_value(row.get("label_confidence")) == "discordant":
        flags.append("discordant_malignancy_votes")
    if clean_value(row.get("overall_consistency")) == "low":
        flags.append("low_annotation_consistency")
    diameter = safe_float(row.get("median_max_diameter_mm"))
    if diameter is not None and diameter > 60:
        flags.append("roi_diameter_gt_60mm")
    return flags


def build_roi_issue_rows(roi_rows, roi_3d_outliers):
    rows = []
    for row in roi_rows:
        flags = roi_issue_flags(row)
        if not flags:
            continue
        rows.append({
            "issue_flags": ";".join(flags),
            "roi_id": row.get("roi_id", ""),
            "patient_folder": row.get("patient_folder", ""),
            "PatientID": row.get("PatientID", ""),
            "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
            "reader_count": row.get("reader_count", ""),
            "overall_consistency": row.get("overall_consistency", ""),
            "label_confidence": row.get("label_confidence", ""),
            "median_max_diameter_mm": row.get("median_max_diameter_mm", ""),
            "multiclass_risk_label": row.get("multiclass_risk_label", ""),
            "binary_label": row.get("binary_label", ""),
            "recommended_handling": roi_recommendation(flags),
        })

    for row in roi_3d_outliers:
        rows.append({
            "issue_flags": row.get("outlier_flags", ""),
            "roi_id": row.get("roi_id", ""),
            "patient_folder": "",
            "PatientID": row.get("PatientID", ""),
            "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
            "reader_count": "",
            "overall_consistency": "",
            "label_confidence": "",
            "median_max_diameter_mm": "",
            "multiclass_risk_label": "",
            "binary_label": "",
            "recommended_handling": row.get("handling", ""),
            "source": "process5_3d_preprocessing",
        })
    return rows


def roi_recommendation(flags):
    recommendations = []
    if "missing_multiclass_label" in flags:
        recommendations.append("exclude from supervised risk-label training; keep as unlabeled ROI")
    if "binary_excluded_intermediate_or_missing" in flags:
        recommendations.append("exclude from binary task but keep for multiclass if labeled")
    if "discordant_malignancy_votes" in flags:
        recommendations.append("use sensitivity analysis or lower sample weight")
    if "low_annotation_consistency" in flags:
        recommendations.append("review or use as lower-confidence training sample")
    if "roi_diameter_gt_60mm" in flags:
        recommendations.append("manual review before modeling")
    return "; ".join(recommendations)


def build_quality_summary(demographics, features, roi_rows, annotation_issues, roi_issues):
    total_patients = len(demographics)
    total_annotations = len(features)
    total_rois = len(roi_rows)

    age_missing = sum(1 for row in demographics if safe_float(row.get("PatientAge_years")) is None)
    sex_missing = sum(1 for row in demographics if patient_sex_category(row.get("PatientSex")) == "Unknown")
    missing_spacing = sum(1 for row in annotation_issues if "missing_spacing" in row.get("issue_flags", ""))
    diameter_le_zero = sum(1 for row in annotation_issues if "diameter_le_zero" in row.get("issue_flags", ""))
    diameter_gt_60 = sum(1 for row in annotation_issues if "diameter_gt_60mm" in row.get("issue_flags", ""))
    missing_malignancy_annotation = sum(1 for row in annotation_issues if "missing_malignancy_score" in row.get("issue_flags", ""))
    missing_multiclass = sum(1 for row in roi_rows if clean_value(row.get("multiclass_risk_label")) == "")
    low_consistency = sum(1 for row in roi_rows if clean_value(row.get("overall_consistency")) == "low")
    discordant_labels = sum(1 for row in roi_rows if clean_value(row.get("label_confidence")) == "discordant")

    rows = [
        issue_row("patient_age_missing", age_missing, total_patients, "high", "Use age imputation only for model covariates; report missingness."),
        issue_row("patient_sex_missing", sex_missing, total_patients, "high", "Use Unknown category; do not infer sex/gender from similarity."),
        issue_row("annotation_missing_spacing", missing_spacing, total_annotations, "medium", "Recover from DICOM geometry where possible; otherwise exclude from size/location summaries."),
        issue_row("annotation_diameter_le_zero", diameter_le_zero, total_annotations, "medium", "Exclude from diameter-based analysis."),
        issue_row("annotation_diameter_gt_60mm", diameter_gt_60, total_annotations, "medium", "Manual review before modeling."),
        issue_row("annotation_missing_malignancy_score", missing_malignancy_annotation, total_annotations, "medium", "Exclude from supervised label construction."),
        issue_row("roi_missing_multiclass_label", missing_multiclass, total_rois, "medium", "Keep as unlabeled ROI or exclude from supervised risk training."),
        issue_row("roi_low_annotation_consistency", low_consistency, total_rois, "medium", "Review or use lower-confidence handling."),
        issue_row("roi_discordant_malignancy_votes", discordant_labels, total_rois, "medium", "Sensitivity analysis or lower training weight."),
    ]
    return rows


def issue_row(issue, count, denominator, severity, handling):
    return {
        "issue": issue,
        "affected_count": count,
        "denominator": denominator,
        "affected_fraction": fraction(count, denominator),
        "severity": severity,
        "recommended_handling": handling,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Write LIDC data-quality issue and imputation planning tables.")
    parser.add_argument("--demographics", type=Path, default=DEMOGRAPHICS_TABLE, help="Process1 patient demographics table.")
    parser.add_argument("--features", type=Path, default=FEATURE_TABLE, help="Process1 nodule feature table.")
    parser.add_argument("--roi-labels", type=Path, default=ROI_LABEL_TABLE, help="Process4 labeled ROI manifest.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    demographics = read_csv_rows(args.demographics)
    features = read_csv_rows(args.features)
    roi_rows = read_csv_rows(args.roi_labels)
    roi_3d_outliers = read_csv_rows(ROI_3D_OUTLIER_TABLE, missing_ok=True)

    model_ready, age_mean, age_median = build_demographics_model_ready(demographics)
    imputation_plan = build_imputation_plan(demographics, age_mean, age_median)
    annotation_issues = build_annotation_outlier_rows(features)
    roi_issues = build_roi_issue_rows(roi_rows, roi_3d_outliers)
    quality_summary = build_quality_summary(demographics, features, roi_rows, annotation_issues, roi_issues)

    write_csv(DEMOGRAPHICS_MODEL_READY_TABLE, model_ready)
    write_csv(DEMOGRAPHICS_IMPUTATION_PLAN_TABLE, imputation_plan)
    write_csv(ANNOTATION_OUTLIER_TABLE, annotation_issues)
    write_csv(ROI_QUALITY_ISSUE_TABLE, roi_issues)
    write_csv(QUALITY_SUMMARY_TABLE, quality_summary)

    log("LIDC data-quality and imputation report complete")
    log("  Quality summary: {}".format(QUALITY_SUMMARY_TABLE))
    log("  Demographics model-ready table: {}".format(DEMOGRAPHICS_MODEL_READY_TABLE))
    log("  Imputation plan: {}".format(DEMOGRAPHICS_IMPUTATION_PLAN_TABLE))
    log("  Annotation outliers: {}".format(ANNOTATION_OUTLIER_TABLE))
    log("  ROI quality issues: {}".format(ROI_QUALITY_ISSUE_TABLE))


if __name__ == "__main__":
    main()
