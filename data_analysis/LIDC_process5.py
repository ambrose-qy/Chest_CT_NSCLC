"""
LIDC-IDRI process5: standardised 3D ROI volume dataset.

This step consumes the process3/process4 ROI manifests and writes actual
pixel-level 3D nodule ROI volumes:

    data/processed/lidc_roi_3d/volumes/*.npz

It also writes label and quality-assessment tables:

    data/processed/tables/lidc_roi_3d_volume_manifest.csv
    data/processed/tables/lidc_roi_3d_label_manifest.csv
    data/processed/tables/lidc_roi_3d_preprocessing_qc.csv
    data/processed/tables/lidc_roi_3d_outlier_report.csv
    data/processed/tables/lidc_roi_3d_preprocessing_summary.csv
    data/processed/tables/lidc_roi_3d_preprocessing_criteria.csv

The output volumes are HU-windowed, normalised to [0, 1], and resized to a
fixed z/y/x shape. Run from the repository root:

    conda run -n torch-gpu python data_analysis/LIDC_process5.py

Useful debug run:

    conda run -n torch-gpu python data_analysis/LIDC_process5.py --max-rois 20 --force
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIDC_ROOT = PROJECT_ROOT / "data" / "raw" / "LIDC"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROCESSED_DIR / "tables"
ROI_3D_DIR = PROCESSED_DIR / "lidc_roi_3d"
VOLUME_DIR = ROI_3D_DIR / "volumes"

MULTICLASS_SPLIT_TABLE = TABLE_DIR / "lidc_roi_multiclass_split_manifest.csv"
BINARY_SPLIT_TABLE = TABLE_DIR / "lidc_roi_binary_split_manifest.csv"

VOLUME_MANIFEST_TABLE = TABLE_DIR / "lidc_roi_3d_volume_manifest.csv"
LABEL_MANIFEST_TABLE = TABLE_DIR / "lidc_roi_3d_label_manifest.csv"
PREPROCESSING_QC_TABLE = TABLE_DIR / "lidc_roi_3d_preprocessing_qc.csv"
OUTLIER_REPORT_TABLE = TABLE_DIR / "lidc_roi_3d_outlier_report.csv"
PREPROCESSING_SUMMARY_TABLE = TABLE_DIR / "lidc_roi_3d_preprocessing_summary.csv"
PREPROCESSING_CRITERIA_TABLE = TABLE_DIR / "lidc_roi_3d_preprocessing_criteria.csv"


def log(message):
    print(message)
    sys.stdout.flush()


def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    VOLUME_DIR.mkdir(parents=True, exist_ok=True)


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


def discover_manifest_root(lidc_root=LIDC_ROOT):
    preferred = Path(lidc_root) / "manifest-1600709154662"
    if preferred.exists():
        return preferred
    candidates = sorted(Path(lidc_root).glob("manifest-*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Could not find data/raw/LIDC/manifest-*")


def normalize_metadata_location(file_location):
    rel = clean_value(file_location).replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    return rel


def build_series_dir_map(manifest_root):
    metadata_path = Path(manifest_root) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(str(metadata_path))

    series_dirs = {}
    for row in read_csv_rows(metadata_path):
        series_uid = clean_value(row.get("Series UID"))
        modality = clean_value(row.get("Modality")).upper()
        rel = normalize_metadata_location(row.get("File Location"))
        if not series_uid or not rel or modality != "CT":
            continue
        series_dir = Path(manifest_root) / rel
        if series_uid not in series_dirs or series_dir.exists():
            series_dirs[series_uid] = series_dir
    return series_dirs


def parse_shape(text):
    parts = [part.strip() for part in clean_value(text).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Volume shape must have z,y,x, for example 32,64,64")
    values = tuple(int(part) for part in parts)
    if min(values) <= 0:
        raise ValueError("Volume shape dimensions must be positive.")
    return values


def image_z_from_dataset(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return safe_float(ipp[2])
    return safe_float(getattr(ds, "SliceLocation", ""))


def pixel_spacing_from_dataset(ds):
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is not None and len(spacing) >= 2:
        return safe_float(spacing[0]), safe_float(spacing[1])
    return None, None


def require_pixel_libraries():
    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("numpy is required for LIDC_process5 3D ROI extraction.") from exc
    try:
        import pydicom
    except Exception as exc:
        raise ImportError("pydicom is required for LIDC_process5 3D ROI extraction.") from exc
    return np, pydicom


def load_dicom_series(series_dir, np, pydicom):
    slices = []
    failures = 0

    for path in sorted(Path(series_dir).iterdir()):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), force=True)
            if not hasattr(ds, "PixelData"):
                continue
            z = image_z_from_dataset(ds)
            instance_number = safe_int(getattr(ds, "InstanceNumber", ""))
            image = ds.pixel_array.astype(np.float32)
            slope = safe_float(getattr(ds, "RescaleSlope", 1.0))
            intercept = safe_float(getattr(ds, "RescaleIntercept", 0.0))
            slope = 1.0 if slope is None else slope
            intercept = 0.0 if intercept is None else intercept
            image = image * slope + intercept
            y_spacing, x_spacing = pixel_spacing_from_dataset(ds)
            slices.append({
                "path": str(path),
                "z": z,
                "instance_number": instance_number,
                "image": image,
                "y_spacing": y_spacing,
                "x_spacing": x_spacing,
                "slice_thickness": safe_float(getattr(ds, "SliceThickness", "")),
            })
        except Exception:
            failures += 1

    if not slices:
        raise ValueError("No readable pixel slices in {}".format(series_dir))

    slices.sort(key=lambda item: (
        item["z"] is None,
        item["z"] if item["z"] is not None else item["instance_number"] or 0,
    ))

    stack = np.stack([item["image"] for item in slices], axis=0).astype(np.float32)
    z_positions = np.array([
        item["z"] if item["z"] is not None else float(index)
        for index, item in enumerate(slices)
    ], dtype=np.float32)

    y_spacing = next((item["y_spacing"] for item in slices if item["y_spacing"] is not None), None)
    x_spacing = next((item["x_spacing"] for item in slices if item["x_spacing"] is not None), None)
    z_diffs = np.diff(np.sort(z_positions))
    z_diffs = z_diffs[np.isfinite(z_diffs)]
    z_diffs = np.abs(z_diffs[z_diffs != 0])
    if z_diffs.size:
        z_spacing = float(np.median(z_diffs))
    else:
        z_spacing = next((item["slice_thickness"] for item in slices if item["slice_thickness"] is not None), 1.0)

    return {
        "volume": stack,
        "z_positions": z_positions,
        "y_spacing": y_spacing,
        "x_spacing": x_spacing,
        "z_spacing": z_spacing,
        "readable_slice_count": len(slices),
        "failed_slice_count": failures,
    }


def clamp_int(value, low, high):
    return max(low, min(int(value), high))


def crop_roi_volume(row, series_data, np):
    volume = series_data["volume"]
    z_positions = series_data["z_positions"]
    depth, height, width = volume.shape

    x_min = safe_float(row.get("x_min_px_roi"))
    x_max = safe_float(row.get("x_max_px_roi"))
    y_min = safe_float(row.get("y_min_px_roi"))
    y_max = safe_float(row.get("y_max_px_roi"))
    z_min = safe_float(row.get("z_min_mm_roi"))
    z_max = safe_float(row.get("z_max_mm_roi"))
    z_center = safe_float(row.get("z_center_mm_consensus"))

    if None in (x_min, x_max, y_min, y_max):
        raise ValueError("Missing ROI x/y bounds")

    x0 = clamp_int(math.floor(x_min), 0, width - 1)
    x1 = clamp_int(math.ceil(x_max), 0, width - 1)
    y0 = clamp_int(math.floor(y_min), 0, height - 1)
    y1 = clamp_int(math.ceil(y_max), 0, height - 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid ROI x/y bounds after clamping")

    if z_min is not None and z_max is not None:
        low = min(z_min, z_max)
        high = max(z_min, z_max)
        z_indices = np.where((z_positions >= low) & (z_positions <= high))[0]
    else:
        z_indices = np.array([], dtype=np.int64)

    z_selection = "roi_z_range"
    if z_indices.size == 0:
        target = z_center if z_center is not None else float(np.median(z_positions))
        nearest = int(np.argmin(np.abs(z_positions - target)))
        z_indices = np.array([nearest], dtype=np.int64)
        z_selection = "nearest_center_slice"

    z0 = int(z_indices.min())
    z1 = int(z_indices.max())
    crop = volume[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].astype(np.float32)

    return crop, {
        "input_depth": depth,
        "input_height": height,
        "input_width": width,
        "crop_z0": z0,
        "crop_z1": z1,
        "crop_y0": y0,
        "crop_y1": y1,
        "crop_x0": x0,
        "crop_x1": x1,
        "crop_depth": crop.shape[0],
        "crop_height": crop.shape[1],
        "crop_width": crop.shape[2],
        "z_selection": z_selection,
    }


def resize_axis(data, target_size, axis, np):
    data = np.moveaxis(data, axis, 0)
    old_size = data.shape[0]
    if old_size == target_size:
        return np.moveaxis(data, 0, axis)
    if old_size == 1:
        resized = np.repeat(data, target_size, axis=0)
        return np.moveaxis(resized, 0, axis)

    old_positions = np.linspace(0.0, old_size - 1.0, num=target_size)
    left = np.floor(old_positions).astype(np.int64)
    right = np.minimum(left + 1, old_size - 1)
    weight = (old_positions - left).astype(np.float32)

    flat = data.reshape(old_size, -1)
    resized_flat = (1.0 - weight[:, None]) * flat[left] + weight[:, None] * flat[right]
    resized = resized_flat.reshape((target_size,) + data.shape[1:])
    return np.moveaxis(resized, 0, axis)


def resize_volume(volume, target_shape, np):
    resized = volume.astype(np.float32, copy=False)
    for axis, target_size in enumerate(target_shape):
        resized = resize_axis(resized, target_size, axis, np)
    return resized.astype(np.float32, copy=False)


def finite_stats(values, np):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "min": "",
            "max": "",
            "mean": "",
            "std": "",
            "p01": "",
            "p99": "",
            "nonfinite_count": int(values.size),
        }
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p99": float(np.percentile(finite, 99)),
        "nonfinite_count": int(values.size - finite.size),
    }


def preprocess_crop(crop, target_shape, hu_min, hu_max, np):
    raw_stats = finite_stats(crop, np)
    nonfinite_count = raw_stats["nonfinite_count"]
    crop = np.nan_to_num(crop, nan=hu_min, posinf=hu_max, neginf=hu_min)

    clipped = np.clip(crop, hu_min, hu_max)
    clipped_voxels = int(np.sum((crop < hu_min) | (crop > hu_max)))
    normalised = (clipped - hu_min) / float(hu_max - hu_min)
    resized = resize_volume(normalised, target_shape, np)

    processed_stats = finite_stats(resized, np)
    return resized.astype(np.float32, copy=False), raw_stats, processed_stats, {
        "nonfinite_replaced_voxels": nonfinite_count,
        "hu_window_clipped_voxels": clipped_voxels,
        "hu_window_clipped_fraction": clipped_voxels / float(crop.size) if crop.size else "",
    }


def volume_filename(row):
    roi_id = clean_value(row.get("roi_id")) or "unknown_roi"
    patient = clean_value(row.get("PatientID")) or clean_value(row.get("patient_folder")) or "unknown_patient"
    safe = "{}__{}".format(patient, roi_id)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe)
    return safe + ".npz"


def load_roi_rows(multiclass_split_path, binary_split_path, include_missing_labels=False):
    rows = read_csv_rows(multiclass_split_path)
    binary_rows = read_csv_rows(binary_split_path) if Path(binary_split_path).exists() else []
    binary_by_roi = {clean_value(row.get("roi_id")): row for row in binary_rows if clean_value(row.get("roi_id"))}

    merged = []
    for row in rows:
        roi_id = clean_value(row.get("roi_id"))
        out = dict(row)
        binary_row = binary_by_roi.get(roi_id, {})
        if binary_row:
            out["binary_split"] = binary_row.get("binary_split", out.get("binary_split", ""))
            out["binary_label"] = binary_row.get("binary_label", out.get("binary_label", ""))
            out["binary_label_id"] = binary_row.get("binary_label_id", out.get("binary_label_id", ""))

        has_multiclass = clean_value(out.get("multiclass_risk_label")) != ""
        if include_missing_labels or has_multiclass:
            merged.append(out)

    return merged


def label_row_from_manifest(row):
    return {
        "roi_id": row.get("roi_id", ""),
        "volume_path": row.get("volume_path", ""),
        "PatientID": row.get("PatientID", ""),
        "patient_folder": row.get("patient_folder", ""),
        "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
        "multiclass_split": row.get("multiclass_split", ""),
        "multiclass_risk_label": row.get("multiclass_risk_label", ""),
        "multiclass_risk_label_id": row.get("multiclass_risk_label_id", ""),
        "binary_split": row.get("binary_split", ""),
        "binary_label": row.get("binary_label", ""),
        "binary_label_id": row.get("binary_label_id", ""),
        "binary_label_status": row.get("binary_label_status", ""),
        "median_malignancy_score": row.get("median_malignancy_score", ""),
        "mean_malignancy_score": row.get("mean_malignancy_score", ""),
        "reader_malignancy_scores": row.get("reader_malignancy_scores", ""),
        "label_confidence": row.get("label_confidence", ""),
    }


def outlier_flags(row, crop_meta, raw_stats, handling):
    flags = []
    diameter = safe_float(row.get("median_max_diameter_mm"))
    if diameter is not None and diameter > 60.0:
        flags.append("diameter_gt_60mm")
    if safe_int(row.get("reader_count")) is not None and safe_int(row.get("reader_count")) < 3:
        flags.append("reader_count_lt_3")
    if clean_value(row.get("overall_consistency")) == "low":
        flags.append("low_annotation_consistency")
    if crop_meta["crop_depth"] <= 1:
        flags.append("single_slice_z_crop")
    if crop_meta["crop_height"] <= 2 or crop_meta["crop_width"] <= 2:
        flags.append("tiny_xy_crop")
    if raw_stats["min"] != "" and raw_stats["min"] < -1200:
        flags.append("raw_hu_below_expected_range")
    if raw_stats["max"] != "" and raw_stats["max"] > 2000:
        flags.append("raw_hu_above_expected_range")
    if handling["nonfinite_replaced_voxels"]:
        flags.append("nonfinite_voxels_replaced")
    return flags


def extract_standardised_volumes(
    rows,
    series_dir_by_uid,
    output_dir,
    target_shape,
    hu_min,
    hu_max,
    force=False,
):
    np, pydicom = require_pixel_libraries()

    grouped = defaultdict(list)
    for row in rows:
        grouped[clean_value(row.get("SeriesInstanceUID"))].append(row)

    volume_manifest = []
    label_manifest = []
    qc_rows = []
    outlier_rows = []
    failure_counts = Counter()
    processed_count = 0
    total_rois = len(rows)

    for series_index, series_uid in enumerate(sorted(grouped.keys()), start=1):
        series_rows = grouped[series_uid]
        if series_index % 50 == 0 or series_index == len(grouped):
            log("  Processed series {}/{}; ROI volumes so far {}".format(
                series_index,
                len(grouped),
                processed_count,
            ))

        series_dir = series_dir_by_uid.get(series_uid)
        if series_dir is None or not Path(series_dir).exists():
            failure_counts["missing_series_dir"] += len(series_rows)
            for row in series_rows:
                outlier_rows.append(failure_row(row, "missing_series_dir"))
            continue

        try:
            series_data = load_dicom_series(series_dir, np, pydicom)
        except Exception as exc:
            failure_counts["series_pixel_load_failed"] += len(series_rows)
            for row in series_rows:
                outlier_rows.append(failure_row(row, "series_pixel_load_failed", str(exc)))
            continue

        for row in series_rows:
            roi_id = clean_value(row.get("roi_id"))
            volume_path = Path(output_dir) / volume_filename(row)
            if volume_path.exists() and not force:
                manifest_row = build_manifest_row(row, volume_path, target_shape, "reused_existing_file")
                volume_manifest.append(manifest_row)
                label_manifest.append(label_row_from_manifest(manifest_row))
                continue

            try:
                crop, crop_meta = crop_roi_volume(row, series_data, np)
                standard_volume, raw_stats, processed_stats, handling = preprocess_crop(
                    crop,
                    target_shape=target_shape,
                    hu_min=hu_min,
                    hu_max=hu_max,
                    np=np,
                )
                flags = outlier_flags(row, crop_meta, raw_stats, handling)

                np.savez_compressed(
                    str(volume_path),
                    volume=standard_volume,
                    roi_id=np.array([roi_id]),
                    multiclass_risk_label_id=np.array([safe_int(row.get("multiclass_risk_label_id")) or -1], dtype=np.int16),
                    binary_label_id=np.array([safe_int(row.get("binary_label_id")) if safe_int(row.get("binary_label_id")) is not None else -1], dtype=np.int16),
                )

                manifest_row = build_manifest_row(row, volume_path, target_shape, "processed")
                manifest_row.update({
                    "series_dir": str(series_dir),
                    "readable_slice_count": series_data["readable_slice_count"],
                    "failed_slice_count": series_data["failed_slice_count"],
                    "z_spacing_mm_estimated": series_data["z_spacing"],
                    "y_spacing_mm": series_data["y_spacing"] if series_data["y_spacing"] is not None else "",
                    "x_spacing_mm": series_data["x_spacing"] if series_data["x_spacing"] is not None else "",
                })
                manifest_row.update(crop_meta)
                volume_manifest.append(manifest_row)
                label_manifest.append(label_row_from_manifest(manifest_row))

                qc_row = build_qc_row(
                    row,
                    volume_path,
                    target_shape,
                    crop_meta,
                    raw_stats,
                    processed_stats,
                    handling,
                    flags,
                )
                qc_rows.append(qc_row)
                if flags:
                    outlier_rows.append(outlier_row(row, volume_path, flags, handling))

                processed_count += 1
            except Exception as exc:
                failure_counts["roi_processing_failed"] += 1
                outlier_rows.append(failure_row(row, "roi_processing_failed", str(exc)))

    log("  Input ROIs: {}".format(total_rois))
    log("  New/updated ROI volumes: {}".format(processed_count))
    log("  Failures: {}".format(dict(failure_counts)))
    return volume_manifest, label_manifest, qc_rows, outlier_rows, failure_counts


def build_manifest_row(row, volume_path, target_shape, status):
    out = dict(row)
    out.update({
        "volume_path": str(volume_path),
        "volume_format": "npz",
        "volume_array_key": "volume",
        "preprocessing_status": status,
        "standard_depth": target_shape[0],
        "standard_height": target_shape[1],
        "standard_width": target_shape[2],
    })
    return out


def build_qc_row(row, volume_path, target_shape, crop_meta, raw_stats, processed_stats, handling, flags):
    return {
        "roi_id": row.get("roi_id", ""),
        "volume_path": str(volume_path),
        "PatientID": row.get("PatientID", ""),
        "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
        "multiclass_split": row.get("multiclass_split", ""),
        "multiclass_risk_label": row.get("multiclass_risk_label", ""),
        "binary_split": row.get("binary_split", ""),
        "binary_label": row.get("binary_label", ""),
        "reader_count": row.get("reader_count", ""),
        "overall_consistency": row.get("overall_consistency", ""),
        "median_max_diameter_mm": row.get("median_max_diameter_mm", ""),
        "crop_depth_before": crop_meta["crop_depth"],
        "crop_height_before": crop_meta["crop_height"],
        "crop_width_before": crop_meta["crop_width"],
        "standard_depth_after": target_shape[0],
        "standard_height_after": target_shape[1],
        "standard_width_after": target_shape[2],
        "z_selection": crop_meta["z_selection"],
        "raw_hu_min_before": raw_stats["min"],
        "raw_hu_max_before": raw_stats["max"],
        "raw_hu_mean_before": raw_stats["mean"],
        "raw_hu_std_before": raw_stats["std"],
        "raw_hu_p01_before": raw_stats["p01"],
        "raw_hu_p99_before": raw_stats["p99"],
        "normalised_min_after": processed_stats["min"],
        "normalised_max_after": processed_stats["max"],
        "normalised_mean_after": processed_stats["mean"],
        "normalised_std_after": processed_stats["std"],
        "nonfinite_replaced_voxels": handling["nonfinite_replaced_voxels"],
        "hu_window_clipped_voxels": handling["hu_window_clipped_voxels"],
        "hu_window_clipped_fraction": handling["hu_window_clipped_fraction"],
        "outlier_flags": ";".join(flags),
    }


def outlier_row(row, volume_path, flags, handling):
    return {
        "roi_id": row.get("roi_id", ""),
        "volume_path": str(volume_path),
        "PatientID": row.get("PatientID", ""),
        "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
        "outlier_status": "handled",
        "outlier_flags": ";".join(flags),
        "handling": "ROI bounds clamped; HU values clipped to window; volume resized to fixed shape.",
        "nonfinite_replaced_voxels": handling.get("nonfinite_replaced_voxels", ""),
        "hu_window_clipped_fraction": handling.get("hu_window_clipped_fraction", ""),
    }


def failure_row(row, reason, detail=""):
    return {
        "roi_id": row.get("roi_id", ""),
        "volume_path": "",
        "PatientID": row.get("PatientID", ""),
        "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
        "outlier_status": "excluded",
        "outlier_flags": reason,
        "handling": "Excluded from 3D ROI volume dataset.",
        "failure_detail": detail,
    }


def write_criteria(target_shape, hu_min, hu_max, include_missing_labels):
    rows = [
        {
            "criterion": "Input ROI source",
            "definition": "Uses process4 multiclass split manifest, augmented with binary split assignments when available.",
        },
        {
            "criterion": "ROI volume extraction",
            "definition": "Crops the process3 consensus x/y ROI box and z ROI range from the raw DICOM CT series.",
        },
        {
            "criterion": "HU conversion",
            "definition": "Applies DICOM RescaleSlope and RescaleIntercept to convert pixels to Hounsfield units.",
        },
        {
            "criterion": "Intensity standardisation",
            "definition": "HU values are clipped to [{}, {}] and normalised to [0, 1].".format(hu_min, hu_max),
        },
        {
            "criterion": "Shape standardisation",
            "definition": "Each ROI is resized to fixed z/y/x shape {}.".format("x".join(str(v) for v in target_shape)),
        },
        {
            "criterion": "Label file",
            "definition": "Each volume has binary and multiclass label columns copied from process4.",
        },
        {
            "criterion": "Missing labels",
            "definition": "Included in extraction: {}.".format(bool(include_missing_labels)),
        },
        {
            "criterion": "Outlier handling",
            "definition": "Bounds are clamped to image limits, non-finite voxels are replaced, HU outliers are clipped, and failed pixel extractions are excluded with reasons recorded.",
        },
    ]
    write_csv(PREPROCESSING_CRITERIA_TABLE, rows)


def counter_rows(counter, prefix):
    return [
        {"section": prefix, "name": key, "value": counter[key]}
        for key in sorted(counter.keys())
    ]


def write_summary(input_rows, volume_manifest, qc_rows, outlier_rows, failure_counts, target_shape, hu_min, hu_max):
    split_counts = Counter(clean_value(row.get("multiclass_split")) or "missing" for row in volume_manifest)
    multi_counts = Counter(clean_value(row.get("multiclass_risk_label")) or "missing" for row in volume_manifest)
    binary_counts = Counter(clean_value(row.get("binary_label")) or "excluded_intermediate" for row in volume_manifest)
    status_counts = Counter(clean_value(row.get("preprocessing_status")) or "missing" for row in volume_manifest)
    outlier_counts = Counter()
    for row in outlier_rows:
        flags = clean_value(row.get("outlier_flags"))
        for flag in flags.split(";"):
            if flag:
                outlier_counts[flag] += 1

    rows = [
        {"section": "dataset", "name": "input_roi_rows", "value": len(input_rows)},
        {"section": "dataset", "name": "volume_manifest_rows", "value": len(volume_manifest)},
        {"section": "dataset", "name": "qc_rows", "value": len(qc_rows)},
        {"section": "dataset", "name": "outlier_or_failure_rows", "value": len(outlier_rows)},
        {"section": "preprocessing", "name": "standard_shape_zyx", "value": "x".join(str(value) for value in target_shape)},
        {"section": "preprocessing", "name": "hu_min", "value": hu_min},
        {"section": "preprocessing", "name": "hu_max", "value": hu_max},
    ]
    rows.extend(counter_rows(status_counts, "status"))
    rows.extend(counter_rows(split_counts, "multiclass_split"))
    rows.extend(counter_rows(multi_counts, "multiclass_label"))
    rows.extend(counter_rows(binary_counts, "binary_label"))
    rows.extend(counter_rows(outlier_counts, "outlier_flag"))
    rows.extend(counter_rows(failure_counts, "failure"))
    write_csv(PREPROCESSING_SUMMARY_TABLE, rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build standardised 3D LIDC-IDRI ROI volumes and label files.")
    parser.add_argument("--multiclass-split", type=Path, default=MULTICLASS_SPLIT_TABLE, help="Process4 multiclass split manifest.")
    parser.add_argument("--binary-split", type=Path, default=BINARY_SPLIT_TABLE, help="Process4 binary split manifest.")
    parser.add_argument("--lidc-root", type=Path, default=LIDC_ROOT, help="Root containing raw LIDC data.")
    parser.add_argument("--output-dir", type=Path, default=VOLUME_DIR, help="Directory for compressed 3D ROI volumes.")
    parser.add_argument("--shape", default="32,64,64", help="Standard output volume shape as z,y,x.")
    parser.add_argument("--hu-min", type=float, default=-1000.0, help="Lower HU window bound.")
    parser.add_argument("--hu-max", type=float, default=400.0, help="Upper HU window bound.")
    parser.add_argument("--max-rois", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--include-missing-labels", action="store_true", help="Include rows without multiclass labels if present.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing volume files.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_dirs()

    target_shape = parse_shape(args.shape)
    if args.hu_max <= args.hu_min:
        raise ValueError("--hu-max must be greater than --hu-min")

    manifest_root = discover_manifest_root(args.lidc_root)
    series_dir_by_uid = build_series_dir_map(manifest_root)
    roi_rows = load_roi_rows(
        args.multiclass_split,
        args.binary_split,
        include_missing_labels=args.include_missing_labels,
    )
    if args.max_rois is not None:
        roi_rows = roi_rows[:args.max_rois]

    log("Building standardised 3D LIDC ROI volumes")
    log("  Input ROI rows: {}".format(len(roi_rows)))
    log("  Standard shape z/y/x: {}".format(target_shape))
    log("  HU window: [{}, {}]".format(args.hu_min, args.hu_max))

    volume_manifest, label_manifest, qc_rows, outlier_rows, failure_counts = extract_standardised_volumes(
        rows=roi_rows,
        series_dir_by_uid=series_dir_by_uid,
        output_dir=args.output_dir,
        target_shape=target_shape,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        force=args.force,
    )

    write_csv(VOLUME_MANIFEST_TABLE, volume_manifest)
    write_csv(LABEL_MANIFEST_TABLE, label_manifest)
    write_csv(PREPROCESSING_QC_TABLE, qc_rows)
    write_csv(OUTLIER_REPORT_TABLE, outlier_rows)
    write_criteria(target_shape, args.hu_min, args.hu_max, args.include_missing_labels)
    write_summary(roi_rows, volume_manifest, qc_rows, outlier_rows, failure_counts, target_shape, args.hu_min, args.hu_max)

    metadata = {
        "volume_manifest": str(VOLUME_MANIFEST_TABLE),
        "label_manifest": str(LABEL_MANIFEST_TABLE),
        "preprocessing_qc": str(PREPROCESSING_QC_TABLE),
        "outlier_report": str(OUTLIER_REPORT_TABLE),
        "summary": str(PREPROCESSING_SUMMARY_TABLE),
        "criteria": str(PREPROCESSING_CRITERIA_TABLE),
        "volume_dir": str(args.output_dir),
        "standard_shape_zyx": target_shape,
        "hu_window": [args.hu_min, args.hu_max],
    }
    with (ROI_3D_DIR / "lidc_roi_3d_dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    log("")
    log("LIDC 3D ROI dataset complete")
    log("  Volume manifest: {}".format(VOLUME_MANIFEST_TABLE))
    log("  Label manifest: {}".format(LABEL_MANIFEST_TABLE))
    log("  QC table: {}".format(PREPROCESSING_QC_TABLE))
    log("  Outlier report: {}".format(OUTLIER_REPORT_TABLE))
    log("  Volume directory: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
