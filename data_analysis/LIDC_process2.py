"""
LIDC-IDRI multi-reader annotation consistency analysis.

This is the second LIDC exercise. It uses the process1 output table:

    data/processed/tables/lidc_nodule_size_location_morphology.csv

Process1 parses the raw XML annotation files and joins them to CT geometry.
Process2 focuses on the four-reader annotation problem:

1. Treat every XML nodule mark as one reader annotation.
2. Match annotations from different radiologists that likely describe the
   same physical lesion.
3. Cluster matched annotations into lesion-level groups.
4. Establish explicit criteria for annotation consistency:
   - reader agreement count
   - centroid/location consistency
   - size consistency
   - malignancy consistency
   - morphology consistency

Run from the repository root:

    conda run -n torch-gpu python data_analysis/LIDC_process2.py

Useful options:

    conda run -n torch-gpu python data_analysis/LIDC_process2.py --force
    conda run -n torch-gpu python data_analysis/LIDC_process2.py --max-series 20

Outputs are written under:

    data/processed/tables/
    data/processed/figures/   (only when matplotlib is installed)
"""

from __future__ import print_function

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"
FIGURE_DIR = PROJECT_ROOT / "data" / "processed" / "figures"

FEATURE_TABLE = TABLE_DIR / "lidc_nodule_size_location_morphology.csv"
CLUSTERED_TABLE = TABLE_DIR / "lidc_reader_annotations_clustered.csv"
CONSISTENCY_TABLE = TABLE_DIR / "lidc_annotation_consistency_by_lesion_cluster.csv"

MORPHOLOGY_COLS = [
    "subtlety",
    "internal_structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]

CONSISTENCY_MORPHOLOGY_COLS = MORPHOLOGY_COLS + ["malignancy"]


def log(message):
    print(message)
    sys.stdout.flush()


def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


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


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def median(values):
    values = sorted([v for v in values if v is not None])
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def percentile(values, q):
    values = sorted([v for v in values if v is not None])
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


def score_range(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return max(values) - min(values)


def coefficient_of_variation(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    avg = mean(values)
    if avg is None or avg == 0:
        return None
    variance = sum((v - avg) ** 2 for v in values) / float(len(values))
    return math.sqrt(variance) / avg


def field_float(row, *names):
    for name in names:
        if name in row:
            value = safe_float(row.get(name))
            if value is not None:
                return value
    return None


def annotation_geometry(row):
    """Return geometry used for matching reader annotations."""
    x_min = field_float(row, "x_min_mm")
    x_max = field_float(row, "x_max_mm")
    y_min = field_float(row, "y_min_mm")
    y_max = field_float(row, "y_max_mm")

    x_center = field_float(row, "x_center_mm", "x_center_mm_image")
    y_center = field_float(row, "y_center_mm", "y_center_mm_image")
    z_center = field_float(row, "z_center_mm")

    z_min = field_float(row, "z_min")
    z_max = field_float(row, "z_max")
    diameter = field_float(row, "max_diameter_mm")

    if x_center is None and x_min is not None and x_max is not None:
        x_center = (x_min + x_max) / 2.0
    if y_center is None and y_min is not None and y_max is not None:
        y_center = (y_min + y_max) / 2.0
    if z_center is None and z_min is not None and z_max is not None:
        z_center = (z_min + z_max) / 2.0

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "x_center": x_center,
        "y_center": y_center,
        "z_center": z_center,
        "diameter": diameter,
    }


def clusterable(row):
    geom = annotation_geometry(row)
    required = [
        geom["x_min"],
        geom["x_max"],
        geom["y_min"],
        geom["y_max"],
        geom["z_min"],
        geom["z_max"],
        geom["x_center"],
        geom["y_center"],
        geom["z_center"],
    ]
    return (
        clean_value(row.get("SeriesInstanceUID")) != ""
        and safe_int(row.get("reader_id")) is not None
        and all(value is not None for value in required)
    )


def interval_overlap(min_a, max_a, min_b, max_b):
    if None in (min_a, max_a, min_b, max_b):
        return 0.0
    return max(0.0, min(max_a, max_b) - max(min_a, min_b))


def bbox_iou_3d(row_a, row_b):
    geom_a = annotation_geometry(row_a)
    geom_b = annotation_geometry(row_b)

    x_overlap = interval_overlap(geom_a["x_min"], geom_a["x_max"], geom_b["x_min"], geom_b["x_max"])
    y_overlap = interval_overlap(geom_a["y_min"], geom_a["y_max"], geom_b["y_min"], geom_b["y_max"])
    z_overlap = interval_overlap(geom_a["z_min"], geom_a["z_max"], geom_b["z_min"], geom_b["z_max"])

    intersection = x_overlap * y_overlap * z_overlap

    volume_a = (
        max(0.0, (geom_a["x_max"] or 0.0) - (geom_a["x_min"] or 0.0))
        * max(0.0, (geom_a["y_max"] or 0.0) - (geom_a["y_min"] or 0.0))
        * max(0.0, (geom_a["z_max"] or 0.0) - (geom_a["z_min"] or 0.0))
    )
    volume_b = (
        max(0.0, (geom_b["x_max"] or 0.0) - (geom_b["x_min"] or 0.0))
        * max(0.0, (geom_b["y_max"] or 0.0) - (geom_b["y_min"] or 0.0))
        * max(0.0, (geom_b["z_max"] or 0.0) - (geom_b["z_min"] or 0.0))
    )
    union = volume_a + volume_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def centroid_distance_mm(row_a, row_b):
    geom_a = annotation_geometry(row_a)
    geom_b = annotation_geometry(row_b)
    needed = [
        geom_a["x_center"],
        geom_a["y_center"],
        geom_a["z_center"],
        geom_b["x_center"],
        geom_b["y_center"],
        geom_b["z_center"],
    ]
    if any(value is None for value in needed):
        return None
    return math.sqrt(
        (geom_a["x_center"] - geom_b["x_center"]) ** 2
        + (geom_a["y_center"] - geom_b["y_center"]) ** 2
        + (geom_a["z_center"] - geom_b["z_center"]) ** 2
    )


def same_lesion_candidate(row_a, row_b, min_iou=0.10, min_distance_mm=10.0):
    reader_a = safe_int(row_a.get("reader_id"))
    reader_b = safe_int(row_b.get("reader_id"))
    if reader_a is None or reader_b is None or reader_a == reader_b:
        return False

    distance = centroid_distance_mm(row_a, row_b)
    geom_a = annotation_geometry(row_a)
    geom_b = annotation_geometry(row_b)
    mean_diameter = mean([geom_a["diameter"], geom_b["diameter"]])
    distance_threshold = max(min_distance_mm, 0.5 * mean_diameter) if mean_diameter is not None else min_distance_mm
    iou = bbox_iou_3d(row_a, row_b)

    return (iou >= min_iou) or (distance is not None and distance <= distance_threshold)


def connected_components_for_series(rows, min_iou=0.10, min_distance_mm=10.0):
    indices = [row["_row_index"] for row in rows]
    parent = {idx: idx for idx in indices}

    def find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for row_a, row_b in combinations(rows, 2):
        if same_lesion_candidate(row_a, row_b, min_iou=min_iou, min_distance_mm=min_distance_mm):
            union(row_a["_row_index"], row_b["_row_index"])

    roots = {idx: find(idx) for idx in indices}
    root_to_number = {
        root: number
        for number, root in enumerate(sorted(set(roots.values())), start=1)
    }
    return {idx: root_to_number[root] for idx, root in roots.items()}


def cluster_annotations(features, force=False, max_series=None, min_iou=0.10, min_distance_mm=10.0):
    if CLUSTERED_TABLE.exists() and not force:
        log("Loading existing clustered reader annotation table.")
        return read_csv_rows(CLUSTERED_TABLE)

    log("Clustering reader annotations into probable lesion groups.")
    indexed_features = []
    for idx, row in enumerate(features):
        out = dict(row)
        out["_row_index"] = idx
        indexed_features.append(out)

    by_series = defaultdict(list)
    for row in indexed_features:
        if clusterable(row):
            by_series[row.get("SeriesInstanceUID")].append(row)

    series_uids = sorted(by_series.keys())
    if max_series is not None:
        series_uids = series_uids[:max_series]

    assignments = {}
    total_series = len(series_uids)
    for series_number, series_uid in enumerate(series_uids, start=1):
        if series_number % 100 == 0 or series_number == total_series:
            log("  Clustered {}/{} series".format(series_number, total_series))

        component_numbers = connected_components_for_series(
            by_series[series_uid],
            min_iou=min_iou,
            min_distance_mm=min_distance_mm,
        )
        for row_index, component_number in component_numbers.items():
            assignments[row_index] = {
                "lesion_cluster_number": component_number,
                "lesion_cluster_id": "{}__cluster_{:03d}".format(series_uid, component_number),
            }

    clustered = []
    for idx, row in enumerate(features):
        out = dict(row)
        assignment = assignments.get(idx, {})
        out["lesion_cluster_number"] = assignment.get("lesion_cluster_number", "")
        out["lesion_cluster_id"] = assignment.get("lesion_cluster_id", "")
        out["clustered_by_process2"] = "yes" if assignment else "no"
        clustered.append(out)

    fieldnames = []
    for row in clustered:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(CLUSTERED_TABLE, clustered, fieldnames=fieldnames)
    return clustered


def max_pairwise_centroid_distance(rows):
    distances = []
    for row_a, row_b in combinations(rows, 2):
        distance = centroid_distance_mm(row_a, row_b)
        if distance is not None:
            distances.append(distance)
    if not distances:
        return None
    return max(distances)


def reader_conflict_count(rows):
    counts = Counter(safe_int(row.get("reader_id")) for row in rows if safe_int(row.get("reader_id")) is not None)
    return sum(count - 1 for count in counts.values() if count > 1)


def majority_value(rows, column):
    values = []
    for row in rows:
        value = safe_int(row.get(column))
        if value is not None:
            values.append(value)
    if not values:
        return ""
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def summarize_cluster(cluster_id, rows):
    readers = sorted(set(safe_int(row.get("reader_id")) for row in rows if safe_int(row.get("reader_id")) is not None))
    diameters = [field_float(row, "max_diameter_mm") for row in rows]
    median_diameter = median(diameters)
    diameter_range = score_range(diameters)
    diameter_cv = coefficient_of_variation(diameters)
    max_centroid_distance = max_pairwise_centroid_distance(rows)
    centroid_threshold = max(10.0, 0.5 * median_diameter) if median_diameter is not None else 10.0

    centroid_consistent = max_centroid_distance is None or max_centroid_distance <= centroid_threshold
    size_consistent = (
        diameter_range is None
        or diameter_range <= 5.0
        or (diameter_cv is not None and diameter_cv <= 0.25)
    )

    morphology_ranges = {}
    for col in CONSISTENCY_MORPHOLOGY_COLS:
        morphology_ranges[col + "_range"] = score_range([safe_float(row.get(col)) for row in rows])

    malignancy_range = morphology_ranges.get("malignancy_range")
    malignancy_consistent = malignancy_range is None or malignancy_range <= 1.0
    morphology_consistent = True
    for col in MORPHOLOGY_COLS:
        value = morphology_ranges.get(col + "_range")
        if value is not None and value > 1.0:
            morphology_consistent = False
            break

    conflicts = reader_conflict_count(rows)
    reader_count = len(readers)
    annotation_count = len(rows)

    if reader_count == 4:
        reader_agreement_category = "4-reader consensus"
    elif reader_count == 3:
        reader_agreement_category = "3-reader majority"
    elif reader_count == 2:
        reader_agreement_category = "2-reader partial"
    else:
        reader_agreement_category = "single-reader only"

    high_consistency = (
        reader_count >= 3
        and conflicts == 0
        and centroid_consistent
        and size_consistent
        and malignancy_consistent
        and morphology_consistent
    )
    moderate_consistency = (
        reader_count >= 2
        and conflicts == 0
        and centroid_consistent
        and size_consistent
    )

    if high_consistency:
        overall_consistency = "high"
    elif moderate_consistency:
        overall_consistency = "moderate"
    else:
        overall_consistency = "low"

    patient_folder = ""
    series_uid = ""
    for row in rows:
        if not patient_folder and clean_value(row.get("patient_folder")):
            patient_folder = row.get("patient_folder")
        if not series_uid and clean_value(row.get("SeriesInstanceUID")):
            series_uid = row.get("SeriesInstanceUID")

    result = {
        "lesion_cluster_id": cluster_id,
        "SeriesInstanceUID": series_uid,
        "patient_folder": patient_folder,
        "annotation_count": annotation_count,
        "reader_count": reader_count,
        "reader_ids": ",".join(str(reader) for reader in readers),
        "reader_conflict_count": conflicts,
        "reader_agreement_category": reader_agreement_category,
        "median_max_diameter_mm": median_diameter if median_diameter is not None else "",
        "mean_max_diameter_mm": mean(diameters) if mean(diameters) is not None else "",
        "diameter_range_mm": diameter_range if diameter_range is not None else "",
        "diameter_cv": diameter_cv if diameter_cv is not None else "",
        "max_centroid_distance_mm": max_centroid_distance if max_centroid_distance is not None else "",
        "centroid_threshold_mm": centroid_threshold,
        "centroid_consistent": centroid_consistent,
        "size_consistent": size_consistent,
        "malignancy_consistent": malignancy_consistent,
        "morphology_consistent": morphology_consistent,
        "overall_consistency": overall_consistency,
        "majority_malignancy": majority_value(rows, "malignancy"),
        "majority_sphericity": majority_value(rows, "sphericity"),
        "majority_margin": majority_value(rows, "margin"),
        "majority_lobulation": majority_value(rows, "lobulation"),
        "majority_spiculation": majority_value(rows, "spiculation"),
        "majority_texture": majority_value(rows, "texture"),
    }
    result.update({
        key: (value if value is not None else "")
        for key, value in morphology_ranges.items()
    })
    return result


def summarize_consistency(clustered, force=False):
    if CONSISTENCY_TABLE.exists() and not force:
        log("Loading existing lesion-cluster consistency table.")
        return read_csv_rows(CONSISTENCY_TABLE)

    log("Summarizing consistency criteria for each lesion cluster.")
    by_cluster = defaultdict(list)
    for row in clustered:
        cluster_id = clean_value(row.get("lesion_cluster_id"))
        if cluster_id:
            by_cluster[cluster_id].append(row)

    rows = []
    for cluster_id in sorted(by_cluster.keys()):
        rows.append(summarize_cluster(cluster_id, by_cluster[cluster_id]))

    write_csv(CONSISTENCY_TABLE, rows)
    return rows


def build_criteria_table():
    criteria_rows = [
        {
            "criterion": "Lesion matching",
            "definition": "Two annotations can be linked only if they are from different readers and either 3D bounding-box IoU >= 0.10 or centroid distance <= max(10 mm, half the mean pair diameter).",
        },
        {
            "criterion": "Reader agreement",
            "definition": "The number of unique radiologists represented in a lesion cluster: 1, 2, 3, or 4.",
        },
        {
            "criterion": "Reader conflict",
            "definition": "A conflict occurs if one cluster contains more than one annotation from the same reader.",
        },
        {
            "criterion": "Centroid consistency",
            "definition": "Maximum pairwise centroid distance within a cluster <= max(10 mm, half the cluster median diameter).",
        },
        {
            "criterion": "Size consistency",
            "definition": "Maximum-diameter range <= 5 mm or diameter coefficient of variation <= 0.25.",
        },
        {
            "criterion": "Malignancy consistency",
            "definition": "Reader malignancy score range <= 1 point when malignancy scores are available.",
        },
        {
            "criterion": "Morphology consistency",
            "definition": "Ordinal morphology score ranges are <= 1 point for subtlety, internal structure, calcification, sphericity, margin, lobulation, spiculation, and texture.",
        },
        {
            "criterion": "Overall consistency",
            "definition": "High: >=3 readers, no reader conflict, centroid consistency, size consistency, malignancy consistency, and morphology consistency. Moderate: >=2 readers, no conflict, centroid consistency, and size consistency. Otherwise low.",
        },
    ]
    write_csv(TABLE_DIR / "lidc_annotation_consistency_criteria.csv", criteria_rows)


def build_distribution_outputs(consistency_rows):
    agreement_counts = Counter(row.get("reader_agreement_category") or "missing" for row in consistency_rows)
    write_csv(
        TABLE_DIR / "lidc_reader_agreement_distribution.csv",
        [
            {"reader_agreement_category": key, "lesion_cluster_count": agreement_counts[key]}
            for key in sorted(agreement_counts.keys())
        ],
    )

    overall_counts = Counter(row.get("overall_consistency") or "missing" for row in consistency_rows)
    write_csv(
        TABLE_DIR / "lidc_overall_consistency_distribution.csv",
        [
            {"overall_consistency": key, "lesion_cluster_count": overall_counts[key]}
            for key in sorted(overall_counts.keys())
        ],
    )

    criterion_cols = [
        "centroid_consistent",
        "size_consistent",
        "malignancy_consistent",
        "morphology_consistent",
    ]
    criterion_rows = []
    total = len(consistency_rows)
    for col in criterion_cols:
        passed = sum(1 for row in consistency_rows if clean_value(row.get(col)).lower() == "true")
        criterion_rows.append({
            "criterion": col,
            "pass_count": passed,
            "cluster_count": total,
            "pass_fraction": passed / float(total) if total else "",
        })
    write_csv(TABLE_DIR / "lidc_consistency_criterion_pass_rates.csv", criterion_rows)

    four_reader = [
        row for row in consistency_rows
        if safe_int(row.get("reader_count")) == 4
    ]
    write_csv(TABLE_DIR / "lidc_four_reader_annotation_consistency.csv", four_reader)

    diameters = [safe_float(row.get("median_max_diameter_mm")) for row in consistency_rows]
    centroid_distances = [safe_float(row.get("max_centroid_distance_mm")) for row in consistency_rows]
    summary = [{
        "lesion_cluster_count": len(consistency_rows),
        "four_reader_cluster_count": len(four_reader),
        "three_reader_cluster_count": sum(1 for row in consistency_rows if safe_int(row.get("reader_count")) == 3),
        "two_reader_cluster_count": sum(1 for row in consistency_rows if safe_int(row.get("reader_count")) == 2),
        "single_reader_cluster_count": sum(1 for row in consistency_rows if safe_int(row.get("reader_count")) == 1),
        "high_consistency_cluster_count": overall_counts.get("high", 0),
        "moderate_consistency_cluster_count": overall_counts.get("moderate", 0),
        "low_consistency_cluster_count": overall_counts.get("low", 0),
        "median_cluster_diameter_mm": median(diameters) if median(diameters) is not None else "",
        "median_max_centroid_distance_mm": median(centroid_distances) if median(centroid_distances) is not None else "",
    }]
    write_csv(TABLE_DIR / "lidc_multireader_consistency_summary.csv", summary)


def maybe_plot(consistency_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log("matplotlib is not installed; skipping consistency PNG figures.")
        return

    agreement_counts = Counter(row.get("reader_agreement_category") or "missing" for row in consistency_rows)
    labels = list(sorted(agreement_counts.keys()))
    plt.figure(figsize=(8, 5))
    plt.bar(labels, [agreement_counts[label] for label in labels])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Lesion clusters")
    plt.title("LIDC Reader Agreement Per Lesion Cluster")
    plt.tight_layout()
    plt.savefig(str(FIGURE_DIR / "lidc_reader_agreement_distribution.png"), dpi=300)
    plt.close()

    overall_counts = Counter(row.get("overall_consistency") or "missing" for row in consistency_rows)
    labels = list(sorted(overall_counts.keys()))
    plt.figure(figsize=(6, 5))
    plt.bar(labels, [overall_counts[label] for label in labels])
    plt.ylabel("Lesion clusters")
    plt.title("LIDC Overall Annotation Consistency")
    plt.tight_layout()
    plt.savefig(str(FIGURE_DIR / "lidc_overall_consistency_distribution.png"), dpi=300)
    plt.close()

    distances = [safe_float(row.get("max_centroid_distance_mm")) for row in consistency_rows]
    distances = [min(value, 60.0) for value in distances if value is not None]
    if distances:
        plt.figure(figsize=(8, 5))
        plt.hist(distances, bins=40)
        plt.xlabel("Maximum pairwise centroid distance (mm, clipped at 60)")
        plt.ylabel("Lesion clusters")
        plt.title("LIDC Reader Centroid Disagreement")
        plt.tight_layout()
        plt.savefig(str(FIGURE_DIR / "lidc_centroid_disagreement_distribution.png"), dpi=300)
        plt.close()


def print_summary(consistency_rows):
    agreement_counts = Counter(row.get("reader_agreement_category") or "missing" for row in consistency_rows)
    overall_counts = Counter(row.get("overall_consistency") or "missing" for row in consistency_rows)
    conflict_count = sum(1 for row in consistency_rows if (safe_int(row.get("reader_conflict_count")) or 0) > 0)

    log("")
    log("LIDC multi-reader annotation consistency complete")
    log("  Lesion clusters: {}".format(len(consistency_rows)))
    log("  Reader agreement: {}".format(dict(agreement_counts)))
    log("  Overall consistency: {}".format(dict(overall_counts)))
    log("  Clusters with repeated-reader conflicts: {}".format(conflict_count))
    log("  Criteria table: {}".format(TABLE_DIR / "lidc_annotation_consistency_criteria.csv"))
    log("  Clustered annotations: {}".format(CLUSTERED_TABLE))
    log("  Consistency table: {}".format(CONSISTENCY_TABLE))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyse LIDC-IDRI four-reader XML annotation consistency.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate process2 clustered and consistency tables.",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Debug option: cluster only the first N series.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.10,
        help="Minimum 3D bounding-box IoU for linking annotations from different readers.",
    )
    parser.add_argument(
        "--min-distance-mm",
        type=float,
        default=10.0,
        help="Minimum centroid-distance threshold used by the lesion matching rule.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_dirs()

    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(
            "Missing {}. Run data_analysis/LIDC_process1.py first.".format(FEATURE_TABLE)
        )

    log("Project root: {}".format(PROJECT_ROOT))
    log("Feature table: {}".format(FEATURE_TABLE))

    features = read_csv_rows(FEATURE_TABLE)
    log("Loaded reader annotations: {}".format(len(features)))

    clustered = cluster_annotations(
        features,
        force=args.force,
        max_series=args.max_series,
        min_iou=args.min_iou,
        min_distance_mm=args.min_distance_mm,
    )
    consistency_rows = summarize_consistency(clustered, force=args.force)

    build_criteria_table()
    build_distribution_outputs(consistency_rows)
    maybe_plot(consistency_rows)
    print_summary(consistency_rows)


if __name__ == "__main__":
    main()
