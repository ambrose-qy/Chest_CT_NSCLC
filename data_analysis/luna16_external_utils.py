"""Shared helpers for LUNA16 external-domain ROI extraction and detection."""

from __future__ import print_function

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from LUNA_process import clean_value, read_luna_volume, safe_float


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"
DEFAULT_LUNA_PROCESSED_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "luna_s0_4",
    PROJECT_ROOT / "data" / "processed" / "luna_s5_9",
]
DEFAULT_LIDC_LABELED = TABLE_DIR / "lidc_roi_labeled_manifest.csv"
DEFAULT_LIDC_BINARY_SPLIT = TABLE_DIR / "lidc_roi_binary_split_manifest.csv"
DEFAULT_LIDC_MULTICLASS_SPLIT = TABLE_DIR / "lidc_roi_multiclass_split_manifest.csv"


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_shape(value):
    parts = [
        int(part.strip())
        for part in clean_value(value).lower().replace("x", ",").split(",")
        if part.strip()
    ]
    if len(parts) != 3 or min(parts) <= 0:
        raise ValueError("Shape must contain positive z,y,x values, for example 64,64,64.")
    return tuple(parts)


def load_luna_scan_index(processed_dirs=None):
    processed_dirs = processed_dirs or DEFAULT_LUNA_PROCESSED_DIRS
    index = {}
    for processed_dir in processed_dirs:
        table_dir = Path(processed_dir) / "tables"
        manifest_paths = sorted(table_dir.glob("*_manifest.csv"))
        for manifest_path in manifest_paths:
            for row in read_csv_rows(manifest_path):
                seriesuid = clean_value(row.get("seriesuid"))
                mhd_path = clean_value(row.get("mhd_path"))
                if seriesuid and mhd_path and Path(mhd_path).exists():
                    index[seriesuid] = {
                        "seriesuid": seriesuid,
                        "anonymised_id": clean_value(row.get("anonymised_id")),
                        "subset": clean_value(row.get("subset")),
                        "mhd_path": mhd_path,
                    }
    return index


def load_luna_annotation_coordinates(processed_dirs=None):
    processed_dirs = processed_dirs or DEFAULT_LUNA_PROCESSED_DIRS
    rows = []
    for processed_dir in processed_dirs:
        table_dir = Path(processed_dir) / "tables"
        for path in sorted(table_dir.glob("*_coord_validation.csv")):
            rows.extend(read_csv_rows(path))
    return rows


def load_lidc_label_index(
    labeled_path=DEFAULT_LIDC_LABELED,
    binary_split_path=DEFAULT_LIDC_BINARY_SPLIT,
    multiclass_split_path=DEFAULT_LIDC_MULTICLASS_SPLIT,
):
    binary_splits = {
        clean_value(row.get("roi_id")): clean_value(row.get("binary_split"))
        for row in read_csv_rows(binary_split_path)
    }
    multiclass_splits = {
        clean_value(row.get("roi_id")): clean_value(row.get("multiclass_split"))
        for row in read_csv_rows(multiclass_split_path)
    }
    by_series = defaultdict(list)
    for row in read_csv_rows(labeled_path):
        out = dict(row)
        roi_id = clean_value(row.get("roi_id"))
        out["binary_split"] = binary_splits.get(roi_id, "")
        out["multiclass_split"] = multiclass_splits.get(roi_id, "")
        by_series[clean_value(row.get("SeriesInstanceUID"))].append(out)
    return by_series


def match_distance_mm(luna_row, lidc_row):
    x = safe_float(luna_row.get("original_voxel_x"))
    y = safe_float(luna_row.get("original_voxel_y"))
    z = safe_float(luna_row.get("coordZ"))
    rx = safe_float(lidc_row.get("x_center_px_consensus"))
    ry = safe_float(lidc_row.get("y_center_px_consensus"))
    rz = safe_float(lidc_row.get("z_center_mm_consensus"))
    spacing_x = safe_float(lidc_row.get("x_spacing_mm"))
    spacing_y = safe_float(lidc_row.get("y_spacing_mm"))
    if None in (x, y, z, rx, ry, rz, spacing_x, spacing_y):
        return None
    return math.sqrt(
        ((x - rx) * spacing_x) ** 2
        + ((y - ry) * spacing_y) ** 2
        + (z - rz) ** 2
    )


def match_luna_to_lidc(
    luna_rows,
    lidc_by_series,
    max_distance_mm=5.0,
):
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:
        raise ImportError("scipy is required for one-to-one LUNA/LIDC matching.") from exc

    luna_by_series = defaultdict(list)
    for row in luna_rows:
        luna_by_series[clean_value(row.get("seriesuid"))].append(row)

    matches = []
    unmatched = []
    for seriesuid, series_luna_rows in sorted(luna_by_series.items()):
        series_lidc_rows = lidc_by_series.get(seriesuid, [])
        if not series_lidc_rows:
            for row in series_luna_rows:
                unmatched.append({
                    "seriesuid": seriesuid,
                    "coordX": row.get("coordX", ""),
                    "coordY": row.get("coordY", ""),
                    "coordZ": row.get("coordZ", ""),
                    "reason": "series_not_present_in_lidc_manifest",
                })
            continue

        costs = np.full(
            (len(series_luna_rows), len(series_lidc_rows)),
            float(max_distance_mm) + 1000.0,
            dtype=np.float64,
        )
        for luna_index, luna_row in enumerate(series_luna_rows):
            for lidc_index, lidc_row in enumerate(series_lidc_rows):
                distance = match_distance_mm(luna_row, lidc_row)
                if distance is not None:
                    costs[luna_index, lidc_index] = distance

        assigned_luna = set()
        luna_indices, lidc_indices = linear_sum_assignment(costs)
        for luna_index, lidc_index in zip(luna_indices.tolist(), lidc_indices.tolist()):
            distance = float(costs[luna_index, lidc_index])
            if distance > float(max_distance_mm):
                continue
            assigned_luna.add(luna_index)
            luna_row = series_luna_rows[luna_index]
            lidc_row = series_lidc_rows[lidc_index]
            out = dict(luna_row)
            out.update({
                "matched_roi_id": lidc_row.get("roi_id", ""),
                "match_distance_mm": distance,
                "patient_id": lidc_row.get("PatientID", ""),
                "median_malignancy_score": lidc_row.get("median_malignancy_score", ""),
                "binary_label": lidc_row.get("binary_label", ""),
                "binary_label_id": lidc_row.get("binary_label_id", ""),
                "binary_split": lidc_row.get("binary_split", ""),
                "multiclass_risk_label": lidc_row.get("multiclass_risk_label", ""),
                "multiclass_risk_label_id": lidc_row.get("multiclass_risk_label_id", ""),
                "multiclass_split": lidc_row.get("multiclass_split", ""),
                "label_confidence": lidc_row.get("label_confidence", ""),
                "reader_agreement_category": lidc_row.get("reader_agreement_category", ""),
                "label_source": "LIDC-IDRI radiologist malignancy scores matched by series and nodule centre",
            })
            matches.append(out)

        for luna_index, row in enumerate(series_luna_rows):
            if luna_index not in assigned_luna:
                nearest = float(np.min(costs[luna_index])) if costs.shape[1] else ""
                unmatched.append({
                    "seriesuid": seriesuid,
                    "coordX": row.get("coordX", ""),
                    "coordY": row.get("coordY", ""),
                    "coordZ": row.get("coordZ", ""),
                    "nearest_distance_mm": nearest,
                    "reason": "no_unique_lidc_roi_within_threshold",
                })
    return matches, unmatched


def continuous_voxel_xyz(world_xyz, info):
    transform = np.asarray(info["transform_matrix"], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(info["origin_xyz"], dtype=np.float64)
    spacing = np.asarray(info["spacing_xyz"], dtype=np.float64)
    return np.linalg.inv(transform).dot(np.asarray(world_xyz, dtype=np.float64) - origin) / spacing


def extract_world_patch(
    volume_hu,
    info,
    center_world_xyz,
    output_shape=(64, 64, 64),
    output_spacing_xyz=(1.0, 1.0, 1.0),
    pad_hu=-1000.0,
):
    try:
        from scipy.ndimage import map_coordinates
    except Exception as exc:
        raise ImportError("scipy is required for physical-space ROI extraction.") from exc

    output_shape = tuple(int(value) for value in output_shape)
    center_voxel_xyz = continuous_voxel_xyz(center_world_xyz, info)
    input_spacing_xyz = np.asarray(info["spacing_xyz"], dtype=np.float64)
    output_spacing_xyz = np.asarray(output_spacing_xyz, dtype=np.float64)

    offset_z = (np.arange(output_shape[0], dtype=np.float64) - (output_shape[0] - 1) / 2.0)
    offset_y = (np.arange(output_shape[1], dtype=np.float64) - (output_shape[1] - 1) / 2.0)
    offset_x = (np.arange(output_shape[2], dtype=np.float64) - (output_shape[2] - 1) / 2.0)
    grid_z, grid_y, grid_x = np.meshgrid(offset_z, offset_y, offset_x, indexing="ij")
    sample_x = center_voxel_xyz[0] + grid_x * output_spacing_xyz[0] / input_spacing_xyz[0]
    sample_y = center_voxel_xyz[1] + grid_y * output_spacing_xyz[1] / input_spacing_xyz[1]
    sample_z = center_voxel_xyz[2] + grid_z * output_spacing_xyz[2] / input_spacing_xyz[2]
    return map_coordinates(
        np.asarray(volume_hu, dtype=np.float32),
        [sample_z, sample_y, sample_x],
        order=1,
        mode="constant",
        cval=float(pad_hu),
        prefilter=False,
    ).astype(np.float32)


def normalise_hu(volume_hu, hu_min=-1000.0, hu_max=400.0):
    clipped = np.clip(np.asarray(volume_hu, dtype=np.float32), hu_min, hu_max)
    return (clipped - float(hu_min)) / float(hu_max - hu_min)


def read_luna_scan(mhd_path):
    volume, info = read_luna_volume(mhd_path)
    return np.asarray(volume, dtype=np.float32), info
