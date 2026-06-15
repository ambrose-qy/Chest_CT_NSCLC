"""
LIDC-IDRI process3: extract agreed nodule ROI manifests.

This step uses the outputs from process1 and process2:

    data/processed/tables/lidc_reader_annotations_clustered.csv
    data/processed/tables/lidc_annotation_consistency_by_lesion_cluster.csv

Goal:

    Extract ROI definitions for lung nodules >= 3 mm in diameter that were
    approved/marked by at least 3 radiologists.

In this script, "approved by at least 3 radiologists" means at least 3 unique
reader annotations in a process2 lesion cluster have an estimated maximum
diameter >= 3 mm, with no repeated-reader conflict inside the cluster. Pass
--strict-reader-size to require every contributing reader annotation in the
selected cluster to be >= 3 mm.

The script writes ROI manifests rather than pixel crops. This keeps the step
independent from pydicom/numpy and gives clean inputs for any later image-crop
or mask-building pipeline.

Run from the repository root:

    conda run -n torch-gpu python data_analysis/LIDC_process3.py

Useful options:

    conda run -n torch-gpu python data_analysis/LIDC_process3.py --margin-mm 5
    conda run -n torch-gpu python data_analysis/LIDC_process3.py --strict-reader-size
"""

from __future__ import print_function

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"

CLUSTERED_TABLE = TABLE_DIR / "lidc_reader_annotations_clustered.csv"
CONSISTENCY_TABLE = TABLE_DIR / "lidc_annotation_consistency_by_lesion_cluster.csv"

ELIGIBLE_CLUSTERS_TABLE = TABLE_DIR / "lidc_roi_eligible_lesion_clusters.csv"
CONSENSUS_ROI_TABLE = TABLE_DIR / "lidc_roi_consensus_manifest.csv"
READER_ROI_TABLE = TABLE_DIR / "lidc_roi_reader_annotation_manifest.csv"
ROI_CRITERIA_TABLE = TABLE_DIR / "lidc_roi_extraction_criteria.csv"
ROI_SUMMARY_TABLE = TABLE_DIR / "lidc_roi_extraction_summary.csv"

MORPHOLOGY_COLS = [
    "subtlety",
    "internal_structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy",
]


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


def safe_bool(value):
    text = clean_value(value).lower()
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n"):
        return False
    return None


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


def percentile(values, q):
    values = sorted([value for value in values if value is not None])
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return values[low]
    return values[low] * (high - pos) + values[high] * (pos - low)


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


def parse_pixel_spacing(value):
    parts = [
        float(text)
        for text in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", clean_value(value))
    ]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def field_float(row, *names):
    for name in names:
        if name in row:
            value = safe_float(row.get(name))
            if value is not None:
                return value
    return None


def clamp(value, low, high):
    if value is None:
        return ""
    if high is None:
        return max(low, value)
    return min(max(low, value), high)


def roi_id_from_cluster(cluster_id):
    text = clean_value(cluster_id)
    if "__cluster_" in text:
        series_uid, cluster_number = text.split("__cluster_", 1)
        suffix = cluster_number.zfill(3)
        series_tail = series_uid.split(".")[-1]
        return "ROI_{}_{}".format(series_tail, suffix)
    return "ROI_" + re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def group_rows_by_cluster(clustered_rows):
    groups = defaultdict(list)
    for row in clustered_rows:
        cluster_id = clean_value(row.get("lesion_cluster_id"))
        if cluster_id:
            groups[cluster_id].append(row)
    return groups


def size_approving_readers(reader_rows, min_diameter_mm):
    approving = set()
    for row in reader_rows:
        reader_id = safe_int(row.get("reader_id"))
        diameter = safe_float(row.get("max_diameter_mm"))
        if reader_id is not None and diameter is not None and diameter >= min_diameter_mm:
            approving.add(reader_id)
    return approving


def cluster_is_eligible(consistency_row, reader_rows, min_diameter_mm, min_readers, strict_reader_size):
    reader_count = safe_int(consistency_row.get("reader_count")) or 0
    conflict_count = safe_int(consistency_row.get("reader_conflict_count")) or 0
    median_diameter = safe_float(consistency_row.get("median_max_diameter_mm"))
    approving_readers = size_approving_readers(reader_rows, min_diameter_mm)

    if reader_count < min_readers:
        return False, "reader_count_below_threshold"
    if conflict_count > 0:
        return False, "reader_conflict"
    if len(approving_readers) < min_readers:
        return False, "size_approving_reader_count_below_threshold"

    if strict_reader_size:
        diameters = [safe_float(row.get("max_diameter_mm")) for row in reader_rows]
        if not diameters or any(value is None or value < min_diameter_mm for value in diameters):
            return False, "reader_annotation_diameter_below_threshold"

    if median_diameter is None or median_diameter < min_diameter_mm:
        return False, "cluster_median_diameter_below_threshold"

    return True, "eligible"


def build_consensus_roi(cluster_id, consistency_row, reader_rows, margin_mm, min_diameter_mm):
    roi_id = roi_id_from_cluster(cluster_id)

    x_mins = [safe_float(row.get("x_min")) for row in reader_rows]
    x_maxs = [safe_float(row.get("x_max")) for row in reader_rows]
    y_mins = [safe_float(row.get("y_min")) for row in reader_rows]
    y_maxs = [safe_float(row.get("y_max")) for row in reader_rows]
    z_mins = [safe_float(row.get("z_min")) for row in reader_rows]
    z_maxs = [safe_float(row.get("z_max")) for row in reader_rows]

    x_centers = [safe_float(row.get("x_center_px")) for row in reader_rows]
    y_centers = [safe_float(row.get("y_center_px")) for row in reader_rows]
    z_centers = [safe_float(row.get("z_center_mm")) for row in reader_rows]

    x_diameters = [safe_float(row.get("x_diameter_mm")) for row in reader_rows]
    y_diameters = [safe_float(row.get("y_diameter_mm")) for row in reader_rows]
    z_diameters = [safe_float(row.get("z_diameter_mm")) for row in reader_rows]
    max_diameters = [safe_float(row.get("max_diameter_mm")) for row in reader_rows]

    first = reader_rows[0]
    rows = safe_float(first.get("Rows"))
    columns = safe_float(first.get("Columns"))
    y_spacing = safe_float(first.get("y_spacing_mm"))
    x_spacing = safe_float(first.get("x_spacing_mm"))
    if y_spacing is None or x_spacing is None:
        parsed_y, parsed_x = parse_pixel_spacing(first.get("PixelSpacing"))
        y_spacing = y_spacing if y_spacing is not None else parsed_y
        x_spacing = x_spacing if x_spacing is not None else parsed_x

    margin_x_px = margin_mm / x_spacing if x_spacing else 0.0
    margin_y_px = margin_mm / y_spacing if y_spacing else 0.0

    x_min_raw = min([value for value in x_mins if value is not None]) if any(value is not None for value in x_mins) else None
    x_max_raw = max([value for value in x_maxs if value is not None]) if any(value is not None for value in x_maxs) else None
    y_min_raw = min([value for value in y_mins if value is not None]) if any(value is not None for value in y_mins) else None
    y_max_raw = max([value for value in y_maxs if value is not None]) if any(value is not None for value in y_maxs) else None
    z_min_raw = min([value for value in z_mins if value is not None]) if any(value is not None for value in z_mins) else None
    z_max_raw = max([value for value in z_maxs if value is not None]) if any(value is not None for value in z_maxs) else None

    x_min_roi = math.floor(x_min_raw - margin_x_px) if x_min_raw is not None else None
    x_max_roi = math.ceil(x_max_raw + margin_x_px) if x_max_raw is not None else None
    y_min_roi = math.floor(y_min_raw - margin_y_px) if y_min_raw is not None else None
    y_max_roi = math.ceil(y_max_raw + margin_y_px) if y_max_raw is not None else None

    x_min_roi = clamp(x_min_roi, 0, columns - 1 if columns is not None else None)
    x_max_roi = clamp(x_max_roi, 0, columns - 1 if columns is not None else None)
    y_min_roi = clamp(y_min_roi, 0, rows - 1 if rows is not None else None)
    y_max_roi = clamp(y_max_roi, 0, rows - 1 if rows is not None else None)

    z_min_roi = z_min_raw - margin_mm if z_min_raw is not None else ""
    z_max_roi = z_max_raw + margin_mm if z_max_raw is not None else ""

    reader_ids = sorted(set(safe_int(row.get("reader_id")) for row in reader_rows if safe_int(row.get("reader_id")) is not None))
    diameter_approving_reader_ids = sorted(size_approving_readers(reader_rows, min_diameter_mm))

    consensus = {
        "roi_id": roi_id,
        "lesion_cluster_id": cluster_id,
        "SeriesInstanceUID": consistency_row.get("SeriesInstanceUID") or first.get("SeriesInstanceUID", ""),
        "StudyInstanceUID": first.get("StudyInstanceUID", ""),
        "patient_folder": consistency_row.get("patient_folder") or first.get("patient_folder", ""),
        "PatientID": first.get("PatientID", ""),
        "reader_count": safe_int(consistency_row.get("reader_count")) or len(reader_ids),
        "reader_ids": ",".join(str(value) for value in reader_ids),
        "diameter_approving_reader_count": len(diameter_approving_reader_ids),
        "diameter_approving_reader_ids": ",".join(str(value) for value in diameter_approving_reader_ids),
        "annotation_count": len(reader_rows),
        "reader_agreement_category": consistency_row.get("reader_agreement_category", ""),
        "overall_consistency": consistency_row.get("overall_consistency", ""),
        "centroid_consistent": consistency_row.get("centroid_consistent", ""),
        "size_consistent": consistency_row.get("size_consistent", ""),
        "malignancy_consistent": consistency_row.get("malignancy_consistent", ""),
        "morphology_consistent": consistency_row.get("morphology_consistent", ""),
        "median_max_diameter_mm": safe_float(consistency_row.get("median_max_diameter_mm")),
        "mean_reader_max_diameter_mm": mean(max_diameters),
        "min_reader_max_diameter_mm": min([value for value in max_diameters if value is not None]) if any(value is not None for value in max_diameters) else "",
        "max_reader_max_diameter_mm": max([value for value in max_diameters if value is not None]) if any(value is not None for value in max_diameters) else "",
        "median_x_diameter_mm": median(x_diameters),
        "median_y_diameter_mm": median(y_diameters),
        "median_z_diameter_mm": median(z_diameters),
        "x_center_px_consensus": median(x_centers),
        "y_center_px_consensus": median(y_centers),
        "z_center_mm_consensus": median(z_centers),
        "x_min_px_union": x_min_raw,
        "x_max_px_union": x_max_raw,
        "y_min_px_union": y_min_raw,
        "y_max_px_union": y_max_raw,
        "z_min_mm_union": z_min_raw,
        "z_max_mm_union": z_max_raw,
        "x_min_px_roi": x_min_roi,
        "x_max_px_roi": x_max_roi,
        "y_min_px_roi": y_min_roi,
        "y_max_px_roi": y_max_roi,
        "z_min_mm_roi": z_min_roi,
        "z_max_mm_roi": z_max_roi,
        "roi_margin_mm": margin_mm,
        "Rows": rows if rows is not None else "",
        "Columns": columns if columns is not None else "",
        "x_spacing_mm": x_spacing if x_spacing is not None else "",
        "y_spacing_mm": y_spacing if y_spacing is not None else "",
        "SliceThickness": first.get("SliceThickness", ""),
        "PixelSpacing": first.get("PixelSpacing", ""),
        "image_side": first.get("image_side", ""),
        "image_ap_position": first.get("image_ap_position", ""),
        "z_location_tertile": first.get("z_location_tertile", ""),
        "majority_malignancy": consistency_row.get("majority_malignancy", ""),
        "majority_internal_structure": consistency_row.get("majority_internal_structure", ""),
        "majority_calcification": consistency_row.get("majority_calcification", ""),
        "majority_sphericity": consistency_row.get("majority_sphericity", ""),
        "majority_margin": consistency_row.get("majority_margin", ""),
        "majority_lobulation": consistency_row.get("majority_lobulation", ""),
        "majority_spiculation": consistency_row.get("majority_spiculation", ""),
        "majority_texture": consistency_row.get("majority_texture", ""),
    }
    return consensus


def build_reader_roi_rows(consensus_row, reader_rows):
    output_rows = []
    for row in reader_rows:
        out = {
            "roi_id": consensus_row.get("roi_id", ""),
            "lesion_cluster_id": consensus_row.get("lesion_cluster_id", ""),
            "SeriesInstanceUID": consensus_row.get("SeriesInstanceUID", ""),
            "patient_folder": consensus_row.get("patient_folder", ""),
            "reader_id": row.get("reader_id", ""),
            "nodule_id": row.get("nodule_id", ""),
            "xml_file": row.get("xml_file", ""),
            "xml_name": row.get("xml_name", ""),
            "roi_count": row.get("roi_count", ""),
            "edge_point_count": row.get("edge_point_count", ""),
            "max_diameter_mm": row.get("max_diameter_mm", ""),
            "size_category": row.get("size_category", ""),
            "x_min": row.get("x_min", ""),
            "x_max": row.get("x_max", ""),
            "y_min": row.get("y_min", ""),
            "y_max": row.get("y_max", ""),
            "z_min": row.get("z_min", ""),
            "z_max": row.get("z_max", ""),
            "x_center_px": row.get("x_center_px", ""),
            "y_center_px": row.get("y_center_px", ""),
            "z_center_mm": row.get("z_center_mm", ""),
        }
        for col in MORPHOLOGY_COLS:
            out[col] = row.get(col, "")
        output_rows.append(out)
    return output_rows


def extract_rois(clustered_rows, consistency_rows, min_diameter_mm, min_readers, strict_reader_size, margin_mm):
    clustered_by_cluster = group_rows_by_cluster(clustered_rows)

    eligible_clusters = []
    consensus_rows = []
    reader_manifest_rows = []
    exclusion_counts = Counter()

    for consistency_row in consistency_rows:
        cluster_id = clean_value(consistency_row.get("lesion_cluster_id"))
        reader_rows = clustered_by_cluster.get(cluster_id, [])
        eligible, reason = cluster_is_eligible(
            consistency_row,
            reader_rows,
            min_diameter_mm=min_diameter_mm,
            min_readers=min_readers,
            strict_reader_size=strict_reader_size,
        )

        if not eligible:
            exclusion_counts[reason] += 1
            continue

        consensus_row = build_consensus_roi(
            cluster_id,
            consistency_row,
            reader_rows,
            margin_mm=margin_mm,
            min_diameter_mm=min_diameter_mm,
        )
        eligible_clusters.append(consistency_row)
        consensus_rows.append(consensus_row)
        reader_manifest_rows.extend(build_reader_roi_rows(consensus_row, reader_rows))

    write_csv(ELIGIBLE_CLUSTERS_TABLE, eligible_clusters)
    write_csv(CONSENSUS_ROI_TABLE, consensus_rows)
    write_csv(READER_ROI_TABLE, reader_manifest_rows)

    return eligible_clusters, consensus_rows, reader_manifest_rows, exclusion_counts


def write_criteria(min_diameter_mm, min_readers, strict_reader_size, margin_mm):
    rows = [
        {
            "criterion": "Input tables",
            "definition": "Uses process2 lesion clusters and process1/process2 reader-level XML annotation features.",
        },
        {
            "criterion": "Minimum nodule size",
            "definition": "At least the minimum number of approving reader annotations must each have maximum diameter >= {} mm. The cluster median diameter must also be >= {} mm.".format(min_diameter_mm, min_diameter_mm),
        },
        {
            "criterion": "Radiologist approval",
            "definition": "A lesion must be represented by at least {} unique radiologists in the matched cluster.".format(min_readers),
        },
        {
            "criterion": "Unanimous within selected readers",
            "definition": "Clusters with repeated-reader conflicts are excluded, so each approving reader contributes at most one annotation.",
        },
        {
            "criterion": "Strict reader-size rule",
            "definition": "Enabled: every reader annotation must be >= threshold. Disabled: cluster median diameter defines the >=3 mm rule.",
            "value": str(bool(strict_reader_size)),
        },
        {
            "criterion": "Consensus ROI",
            "definition": "The exported ROI is the union of all selected reader bounding boxes plus a margin in x/y/z.",
        },
        {
            "criterion": "ROI margin",
            "definition": "{} mm is added around the reader-union box and clamped to image rows/columns in x/y.".format(margin_mm),
        },
    ]
    write_csv(ROI_CRITERIA_TABLE, rows)


def write_summary(clustered_rows, consistency_rows, eligible_clusters, consensus_rows, reader_manifest_rows, exclusion_counts):
    diameters = [safe_float(row.get("median_max_diameter_mm")) for row in consensus_rows]
    reader_counts = Counter(safe_int(row.get("reader_count")) for row in consensus_rows)
    consistency_counts = Counter(row.get("overall_consistency") or "missing" for row in consensus_rows)

    rows = [{
        "total_reader_annotations_input": len(clustered_rows),
        "total_lesion_clusters_input": len(consistency_rows),
        "eligible_lesion_clusters": len(eligible_clusters),
        "eligible_reader_roi_annotations": len(reader_manifest_rows),
        "three_reader_roi_count": reader_counts.get(3, 0),
        "four_reader_roi_count": reader_counts.get(4, 0),
        "high_consistency_roi_count": consistency_counts.get("high", 0),
        "moderate_consistency_roi_count": consistency_counts.get("moderate", 0),
        "low_consistency_roi_count": consistency_counts.get("low", 0),
        "median_roi_diameter_mm": median(diameters) if median(diameters) is not None else "",
        "p25_roi_diameter_mm": percentile(diameters, 0.25) if percentile(diameters, 0.25) is not None else "",
        "p75_roi_diameter_mm": percentile(diameters, 0.75) if percentile(diameters, 0.75) is not None else "",
        "excluded_reader_count_below_threshold": exclusion_counts.get("reader_count_below_threshold", 0),
        "excluded_reader_conflict": exclusion_counts.get("reader_conflict", 0),
        "excluded_cluster_median_diameter_below_threshold": exclusion_counts.get("cluster_median_diameter_below_threshold", 0),
        "excluded_size_approving_reader_count_below_threshold": exclusion_counts.get("size_approving_reader_count_below_threshold", 0),
        "excluded_reader_annotation_diameter_below_threshold": exclusion_counts.get("reader_annotation_diameter_below_threshold", 0),
    }]
    write_csv(ROI_SUMMARY_TABLE, rows)


def print_summary(eligible_clusters, consensus_rows, reader_manifest_rows, exclusion_counts):
    reader_counts = Counter(safe_int(row.get("reader_count")) for row in consensus_rows)
    consistency_counts = Counter(row.get("overall_consistency") or "missing" for row in consensus_rows)

    log("")
    log("LIDC ROI extraction manifest complete")
    log("  Eligible lesion ROIs: {}".format(len(consensus_rows)))
    log("  Reader ROI annotations: {}".format(len(reader_manifest_rows)))
    log("  Reader-count distribution: {}".format(dict(reader_counts)))
    log("  Consistency distribution: {}".format(dict(consistency_counts)))
    log("  Exclusions: {}".format(dict(exclusion_counts)))
    log("  Eligible clusters: {}".format(ELIGIBLE_CLUSTERS_TABLE))
    log("  Consensus ROI manifest: {}".format(CONSENSUS_ROI_TABLE))
    log("  Reader ROI manifest: {}".format(READER_ROI_TABLE))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract agreed LIDC-IDRI nodule ROI manifests.")
    parser.add_argument(
        "--min-diameter-mm",
        type=float,
        default=3.0,
        help="Minimum cluster median maximum diameter for ROI inclusion.",
    )
    parser.add_argument(
        "--min-readers",
        type=int,
        default=3,
        help="Minimum number of unique radiologists approving/marking the lesion.",
    )
    parser.add_argument(
        "--strict-reader-size",
        action="store_true",
        help="Require every contributing reader annotation to be >= min-diameter-mm.",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=5.0,
        help="Physical margin added to the union ROI bounding box.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not CLUSTERED_TABLE.exists():
        raise FileNotFoundError("Missing clustered annotations. Run LIDC_process2.py first: {}".format(CLUSTERED_TABLE))
    if not CONSISTENCY_TABLE.exists():
        raise FileNotFoundError("Missing consistency table. Run LIDC_process2.py first: {}".format(CONSISTENCY_TABLE))

    log("Project root: {}".format(PROJECT_ROOT))
    log("Clustered annotation table: {}".format(CLUSTERED_TABLE))
    log("Consistency table: {}".format(CONSISTENCY_TABLE))

    clustered_rows = read_csv_rows(CLUSTERED_TABLE)
    consistency_rows = read_csv_rows(CONSISTENCY_TABLE)
    log("Loaded reader annotations: {}".format(len(clustered_rows)))
    log("Loaded lesion clusters: {}".format(len(consistency_rows)))

    eligible_clusters, consensus_rows, reader_manifest_rows, exclusion_counts = extract_rois(
        clustered_rows,
        consistency_rows,
        min_diameter_mm=args.min_diameter_mm,
        min_readers=args.min_readers,
        strict_reader_size=args.strict_reader_size,
        margin_mm=args.margin_mm,
    )

    write_criteria(
        min_diameter_mm=args.min_diameter_mm,
        min_readers=args.min_readers,
        strict_reader_size=args.strict_reader_size,
        margin_mm=args.margin_mm,
    )
    write_summary(clustered_rows, consistency_rows, eligible_clusters, consensus_rows, reader_manifest_rows, exclusion_counts)
    print_summary(eligible_clusters, consensus_rows, reader_manifest_rows, exclusion_counts)


if __name__ == "__main__":
    main()
