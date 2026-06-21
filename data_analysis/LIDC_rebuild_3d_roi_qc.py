"""
Rebuild lightweight QC rows for already exported LIDC 3D ROI .npz volumes.

Use this when ``lidc_roi_3d_preprocessing_qc.csv`` exists but has no rows. The
script does not re-export ROIs; it inspects the existing ``volume`` arrays and
writes a practical QC table plus summary.

Run:

    C:\\Users\\Ambro\\.conda\\envs\\torch-gpu\\python.exe data_analysis\\LIDC_rebuild_3d_roi_qc.py
"""

from __future__ import print_function

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"
DEFAULT_MANIFEST = TABLE_DIR / "lidc_roi_3d_volume_manifest.csv"
DEFAULT_QC = TABLE_DIR / "lidc_roi_3d_preprocessing_qc.csv"
DEFAULT_SUMMARY = TABLE_DIR / "lidc_roi_3d_preprocessing_qc_summary.csv"


def clean_value(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("none", "nan"):
        return ""
    return text


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


def stats_for_volume(path):
    if not path.exists():
        return None, "missing_volume_file"
    try:
        with np.load(str(path)) as npz:
            if "volume" not in npz:
                return None, "missing_volume_key"
            volume = npz["volume"].astype(np.float32)
    except Exception as exc:
        return None, "volume_load_failed:{}".format(exc)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return {
            "shape": "x".join(str(value) for value in volume.shape),
            "voxel_count": int(volume.size),
            "finite_voxel_count": 0,
            "nonfinite_voxel_count": int(volume.size),
            "min": "",
            "max": "",
            "mean": "",
            "std": "",
            "p01": "",
            "p99": "",
            "zero_fraction": "",
            "status": "failed",
        }, "no_finite_voxels"
    return {
        "shape": "x".join(str(value) for value in volume.shape),
        "voxel_count": int(volume.size),
        "finite_voxel_count": int(finite.size),
        "nonfinite_voxel_count": int(volume.size - finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p99": float(np.percentile(finite, 99)),
        "zero_fraction": float((finite == 0.0).mean()),
        "status": "ok",
    }, ""


def qc_row(row):
    volume_path = Path(clean_value(row.get("volume_path")))
    stats, error = stats_for_volume(volume_path)
    out = {
        "roi_id": row.get("roi_id", ""),
        "PatientID": row.get("PatientID", ""),
        "patient_folder": row.get("patient_folder", ""),
        "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
        "volume_path": str(volume_path),
        "multiclass_split": row.get("multiclass_split", ""),
        "multiclass_risk_label": row.get("multiclass_risk_label", ""),
        "binary_split": row.get("binary_split", ""),
        "binary_label": row.get("binary_label", ""),
        "median_max_diameter_mm": row.get("median_max_diameter_mm", ""),
        "qc_error": error,
    }
    if stats:
        for key, value in stats.items():
            out["volume_{}".format(key)] = value
    else:
        out["volume_status"] = "failed"
    return out


def summary_rows(qc_rows):
    rows = []
    status_counts = Counter(clean_value(row.get("volume_status")) or "missing" for row in qc_rows)
    shape_counts = Counter(clean_value(row.get("volume_shape")) or "missing" for row in qc_rows)
    error_counts = Counter(clean_value(row.get("qc_error")) or "none" for row in qc_rows)
    rows.append({"section": "dataset", "name": "qc_rows", "value": len(qc_rows)})
    for name, count in sorted(status_counts.items()):
        rows.append({"section": "volume_status", "name": name, "value": count})
    for name, count in sorted(shape_counts.items()):
        rows.append({"section": "volume_shape", "name": name, "value": count})
    for name, count in sorted(error_counts.items()):
        rows.append({"section": "qc_error", "name": name, "value": count})
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild LIDC 3D ROI QC from exported NPZ volumes.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_QC)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metadata-json", type=Path, default=TABLE_DIR / "lidc_roi_3d_preprocessing_qc_rebuild_metadata.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = read_csv_rows(args.manifest)
    qc_rows = [qc_row(row) for row in rows]
    write_csv(args.output, qc_rows)
    write_csv(args.summary, summary_rows(qc_rows))
    write_json(args.metadata_json, {
        "manifest": str(args.manifest),
        "output": str(args.output),
        "summary": str(args.summary),
        "row_count": len(qc_rows),
    })
    print("Rebuilt 3D ROI QC: {}".format(args.output))


if __name__ == "__main__":
    main()
