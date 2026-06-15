"""
Export LUNA16 annotation-centred ROI volumes for external LIDC model validation.

This script consumes the processed LUNA full-volume NPZ files written by
LUNA_process.py / LUNA_process2.py with --save-volumes, plus their coordinate
validation tables. It writes fixed-size nodule-centred ROI cubes that can be
fed to the LIDC 2D/3D Lightning models for external-domain inference.

LUNA16 annotations do not include LIDC malignancy labels. The exported manifest
therefore has nodule coordinates and diameter only, not supervised class labels.
"""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LUNA_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "luna_s0_4",
    PROJECT_ROOT / "data" / "processed" / "luna_s5_9",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "luna_lidc_external_rois"


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
    return int(round(value))


def as_bool(value):
    if isinstance(value, bool):
        return value
    return clean_value(value).lower() in ("1", "true", "yes", "y", "on")


def parse_shape(text):
    parts = [part.strip() for part in clean_value(text).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Shape must be z,y,x, for example 64,64,64")
    shape = tuple(int(part) for part in parts)
    if min(shape) <= 0:
        raise ValueError("Shape dimensions must be positive.")
    return shape


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


def discover_tables(luna_dir):
    tables = Path(luna_dir) / "tables"
    summaries = sorted(tables.glob("*_preprocess_summary.csv"))
    coords = sorted(tables.glob("*_coord_validation.csv"))
    if not summaries or not coords:
        return None, None
    return summaries[0], coords[0]


def fixed_window(center, size, limit):
    center = int(round(center))
    start = center - int(size) // 2
    stop = start + int(size)
    src_start = max(start, 0)
    src_stop = min(stop, int(limit))
    dst_start = src_start - start
    dst_stop = dst_start + max(0, src_stop - src_start)
    return start, stop, src_start, src_stop, dst_start, dst_stop


def crop_volume(volume, center_zyx, shape, pad_value, np):
    z, y, x = [int(round(value)) for value in center_zyx]
    z_start, z_stop, z0, z1, dz0, dz1 = fixed_window(z, shape[0], volume.shape[0])
    y_start, y_stop, y0, y1, dy0, dy1 = fixed_window(y, shape[1], volume.shape[1])
    x_start, x_stop, x0, x1, dx0, dx1 = fixed_window(x, shape[2], volume.shape[2])
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        raise ValueError("Crop centre does not intersect volume.")

    crop = np.full(shape, float(pad_value), dtype=np.float32)
    crop[dz0:dz1, dy0:dy1, dx0:dx1] = volume[z0:z1, y0:y1, x0:x1].astype(np.float32)
    copied = (z1 - z0) * (y1 - y0) * (x1 - x0)
    return crop, {
        "requested_crop_z0": z_start,
        "requested_crop_z1_exclusive": z_stop,
        "requested_crop_y0": y_start,
        "requested_crop_y1_exclusive": y_stop,
        "requested_crop_x0": x_start,
        "requested_crop_x1_exclusive": x_stop,
        "crop_z0": z0,
        "crop_z1": z1 - 1,
        "crop_y0": y0,
        "crop_y1": y1 - 1,
        "crop_x0": x0,
        "crop_x1": x1 - 1,
        "padded_voxels": int(crop.size - copied),
    }


def luna_roi_id(row, index):
    series = clean_value(row.get("seriesuid")) or clean_value(row.get("anonymised_id")) or "unknown"
    return "{}__ann{:04d}".format(series.replace(".", "_"), index)


def build_manifest(luna_dirs, output_dir, shape, force=False, max_rois=None):
    import numpy as np

    output_dir = Path(output_dir)
    volume_dir = output_dir / "volumes"
    volume_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    failures = []
    processed = 0

    for luna_dir in luna_dirs:
        summary_path, coord_path = discover_tables(luna_dir)
        if summary_path is None or coord_path is None:
            continue

        summaries = read_csv_rows(summary_path)
        summary_by_id = {clean_value(row.get("anonymised_id")): row for row in summaries}
        coord_rows = read_csv_rows(coord_path)

        cache = {}
        for index, row in enumerate(coord_rows, start=1):
            if max_rois is not None and processed >= max_rois:
                break
            if not as_bool(row.get("resampled_inside_volume")):
                continue

            anonymised_id = clean_value(row.get("anonymised_id"))
            summary = summary_by_id.get(anonymised_id, {})
            source_npz = clean_value(summary.get("output_npz"))
            if not source_npz or not Path(source_npz).exists():
                failures.append({"seriesuid": row.get("seriesuid", ""), "reason": "missing_saved_luna_volume", "output_npz": source_npz})
                continue

            center_z = safe_int(row.get("resampled_voxel_z"))
            center_y = safe_int(row.get("resampled_voxel_y"))
            center_x = safe_int(row.get("resampled_voxel_x"))
            if None in (center_z, center_y, center_x):
                failures.append({"seriesuid": row.get("seriesuid", ""), "reason": "missing_resampled_center"})
                continue

            roi_id = luna_roi_id(row, index)
            volume_path = volume_dir / "{}.npz".format("".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in roi_id))
            if volume_path.exists() and not force:
                manifest_rows.append(manifest_row(row, summary, roi_id, source_npz, volume_path, shape, {}, "reused_existing_file"))
                processed += 1
                continue

            try:
                if source_npz not in cache:
                    with np.load(source_npz) as npz:
                        cache[source_npz] = npz["image"].astype(np.float32)
                crop, crop_meta = crop_volume(cache[source_npz], (center_z, center_y, center_x), shape, 0.0, np)
                np.savez_compressed(
                    str(volume_path),
                    volume=crop.astype(np.float32),
                    luna_roi_id=np.array([roi_id]),
                    diameter_mm=np.array([safe_float(row.get("diameter_mm")) or -1.0], dtype=np.float32),
                )
                manifest_rows.append(manifest_row(row, summary, roi_id, source_npz, volume_path, shape, crop_meta, "processed"))
                processed += 1
            except Exception as exc:
                failures.append({"seriesuid": row.get("seriesuid", ""), "reason": "roi_export_failed", "detail": str(exc)})

    return manifest_rows, failures


def manifest_row(row, summary, roi_id, source_npz, volume_path, shape, crop_meta, status):
    out = {
        "luna_roi_id": roi_id,
        "seriesuid": row.get("seriesuid", ""),
        "anonymised_id": row.get("anonymised_id", ""),
        "subset": summary.get("subset", ""),
        "source_npz": source_npz,
        "volume_path": str(volume_path),
        "volume_format": "npz",
        "volume_array_key": "volume",
        "preprocessing_status": status,
        "diameter_mm": row.get("diameter_mm", ""),
        "resampled_voxel_x": row.get("resampled_voxel_x", ""),
        "resampled_voxel_y": row.get("resampled_voxel_y", ""),
        "resampled_voxel_z": row.get("resampled_voxel_z", ""),
        "crop_depth": shape[0],
        "crop_height": shape[1],
        "crop_width": shape[2],
        "external_label_available": False,
        "label_note": "LUNA16 annotation list has nodule coordinates and diameter but no LIDC malignancy labels.",
    }
    out.update(crop_meta)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export LUNA16 ROI cubes for external LIDC model validation.")
    parser.add_argument("--luna-processed-dirs", type=Path, nargs="+", default=DEFAULT_LUNA_DIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shape", default="64,64,64", help="ROI crop shape as z,y,x.")
    parser.add_argument("--max-rois", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    shape = parse_shape(args.shape)
    manifest_rows, failures = build_manifest(args.luna_processed_dirs, args.output_dir, shape, force=args.force, max_rois=args.max_rois)

    manifest_path = args.output_dir / "luna_lidc_external_roi_manifest.csv"
    failure_path = args.output_dir / "luna_lidc_external_roi_failures.csv"
    write_csv(manifest_path, manifest_rows)
    write_csv(failure_path, failures)
    with (args.output_dir / "luna_lidc_external_roi_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "manifest": str(manifest_path),
            "failures": str(failure_path),
            "roi_shape_zyx": shape,
            "roi_count": len(manifest_rows),
            "failure_count": len(failures),
            "label_note": "External LUNA16 inference is unsupervised for LIDC malignancy labels.",
        }, f, indent=2, sort_keys=True)

    print("LUNA external ROI manifest: {}".format(manifest_path))
    print("Exported ROI count: {}".format(len(manifest_rows)))
    print("Failures: {}".format(len(failures)))


if __name__ == "__main__":
    main()
