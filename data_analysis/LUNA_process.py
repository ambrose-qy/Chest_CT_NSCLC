"""
LUNA16 subset0-4 preprocessing pipeline.

This script implements the LUNA16 preprocessing pipeline for the unzipped
subset resources in:

    E:/LUNA/subset0 ... E:/LUNA/subset4

and writes derived outputs inside this project:

    data/processed/luna_s0_4/

Pipeline scope only:

* de-identification through SHA1 anonymised scan identifiers
* MetaImage `.mhd/.raw` CT volume loading
* resampling to standard voxel spacing, default 1 x 1 x 1 mm
* HU clipping and normalisation to [0, 1]
* threshold-based lung parenchyma segmentation
* annotation coordinate validation after resampling

LUNA16 is distributed as MetaImage CT volumes rather than original DICOM files.
The de-identification step therefore removes original series UIDs from output
filenames and keeps the mapping in a controlled CSV.

Run from the repository root:

    conda run -n torch-gpu python data_analysis/LUNA_process.py

Useful options:

    conda run -n torch-gpu python data_analysis/LUNA_process.py --max-scans 10 --save-volumes
    conda run -n torch-gpu python data_analysis/LUNA_process.py --save-volumes
    conda run -n torch-gpu python data_analysis/LUNA_process.py --subsets 0 1 2 3 4
    conda run -n torch-gpu python data_analysis/LUNA_process.py --manifest-only
"""

from __future__ import print_function

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LUNA_ROOT = Path("E:/LUNA")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "luna_s0_4"
DEFAULT_OUTPUT_PREFIX = "luna_s0_4"

HU_CLIP_MIN = -1000
HU_CLIP_MAX = 400
LUNG_THRESHOLD_HU = -320
TARGET_SPACING_XYZ = (1.0, 1.0, 1.0)


def log(message):
    print(message)
    sys.stdout.flush()


def require_scientific_stack():
    missing = []
    try:
        import numpy as np  # noqa: F401
    except Exception:
        missing.append("numpy")
    try:
        from scipy import ndimage  # noqa: F401
    except Exception:
        missing.append("scipy")

    if missing:
        raise RuntimeError(
            "Missing required package(s): {}. Install them before running LUNA "
            "preprocessing, for example: python -m pip install numpy scipy matplotlib".format(
                ", ".join(missing)
            )
        )


def optional_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


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


def sha1_anonymise(value, prefix="luna"):
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    return "{}_{}".format(prefix, digest)


def ensure_dirs(output_dir):
    table_dir = output_dir / "tables"
    volume_dir = output_dir / "volumes"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    volume_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, volume_dir, figure_dir


def output_table_path(table_dir, output_prefix, suffix):
    return table_dir / "{}_{}.csv".format(output_prefix, suffix)


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


def parse_number_list(value, dtype=float):
    return [dtype(item) for item in clean_value(value).replace(",", " ").split()]


def parse_mhd_header(mhd_path):
    header = {}
    text = Path(mhd_path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        header[key.strip()] = value.strip()
    return header


def get_numpy_dtype(element_type, byte_order_msb=False):
    import numpy as np

    type_map = {
        "MET_CHAR": "i1",
        "MET_UCHAR": "u1",
        "MET_SHORT": "i2",
        "MET_USHORT": "u2",
        "MET_INT": "i4",
        "MET_UINT": "u4",
        "MET_FLOAT": "f4",
        "MET_DOUBLE": "f8",
    }
    if element_type not in type_map:
        raise ValueError("Unsupported ElementType: {}".format(element_type))

    dtype_code = type_map[element_type]
    if dtype_code.endswith("1"):
        return np.dtype(dtype_code)

    endian = ">" if byte_order_msb else "<"
    return np.dtype(endian + dtype_code)


def discover_subset_mhds(luna_root, subsets):
    records = []
    for subset in subsets:
        subset_name = "subset{}".format(int(subset))
        subset_dir = Path(luna_root) / subset_name
        if not subset_dir.exists():
            log("Subset directory missing, skipped: {}".format(subset_dir))
            continue

        for mhd_path in sorted(subset_dir.glob("*.mhd")):
            header = parse_mhd_header(mhd_path)
            seriesuid = mhd_path.stem
            raw_name = header.get("ElementDataFile", "")
            raw_path = mhd_path.parent / raw_name

            dim_size = parse_number_list(header.get("DimSize", ""), int)
            spacing = parse_number_list(header.get("ElementSpacing", ""), float)
            origin = parse_number_list(header.get("Offset", header.get("Position", "0 0 0")), float)
            transform = parse_number_list(header.get("TransformMatrix", "1 0 0 0 1 0 0 0 1"), float)

            if len(dim_size) != 3 or len(spacing) != 3:
                log("Invalid MHD geometry, skipped: {}".format(mhd_path))
                continue

            row = {
                "seriesuid": seriesuid,
                "anonymised_id": sha1_anonymise(seriesuid),
                "subset": subset_name,
                "mhd_path": str(mhd_path),
                "raw_path": str(raw_path),
                "raw_exists": raw_path.exists(),
                "size_x": dim_size[0],
                "size_y": dim_size[1],
                "size_z": dim_size[2],
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "origin_x": origin[0] if len(origin) > 0 else "",
                "origin_y": origin[1] if len(origin) > 1 else "",
                "origin_z": origin[2] if len(origin) > 2 else "",
                "element_type": header.get("ElementType", ""),
                "compressed_data": header.get("CompressedData", "False"),
                "byte_order_msb": header.get(
                    "ElementByteOrderMSB",
                    header.get("BinaryDataByteOrderMSB", "False"),
                ),
            }
            for i, value in enumerate(transform[:9]):
                row["transform_{}".format(i)] = value
            records.append(row)
    return records


def read_luna_volume(mhd_path):
    import numpy as np

    mhd_path = Path(mhd_path)
    header = parse_mhd_header(mhd_path)
    dim_size = parse_number_list(header["DimSize"], int)
    spacing = parse_number_list(header["ElementSpacing"], float)
    origin = parse_number_list(header.get("Offset", header.get("Position", "0 0 0")), float)
    transform = parse_number_list(header.get("TransformMatrix", "1 0 0 0 1 0 0 0 1"), float)
    raw_path = mhd_path.parent / header["ElementDataFile"]

    byte_order_msb = clean_value(
        header.get("ElementByteOrderMSB", header.get("BinaryDataByteOrderMSB", "False"))
    ).lower() == "true"
    dtype = get_numpy_dtype(header["ElementType"], byte_order_msb)

    raw_bytes = raw_path.read_bytes()
    compressed = clean_value(header.get("CompressedData", "False")).lower() == "true"
    if compressed:
        raw_bytes = zlib.decompress(raw_bytes)

    volume = np.frombuffer(raw_bytes, dtype=dtype)
    expected = dim_size[0] * dim_size[1] * dim_size[2]
    if volume.size != expected:
        raise ValueError("Volume size mismatch for {}: expected {}, got {}".format(mhd_path, expected, volume.size))

    volume = volume.reshape((dim_size[2], dim_size[1], dim_size[0]))
    info = {
        "seriesuid": mhd_path.stem,
        "anonymised_id": sha1_anonymise(mhd_path.stem),
        "mhd_path": str(mhd_path),
        "raw_path": str(raw_path),
        "dim_size_xyz": dim_size,
        "spacing_xyz": spacing,
        "origin_xyz": origin,
        "transform_matrix": transform[:9],
        "element_type": header["ElementType"],
        "header": header,
    }
    return volume, info


def resample_volume_to_spacing(volume_hu_zyx, original_spacing_xyz, target_spacing_xyz, order=1):
    import numpy as np
    from scipy import ndimage

    original_spacing_xyz = np.asarray(original_spacing_xyz, dtype=np.float32)
    target_spacing_xyz = np.asarray(target_spacing_xyz, dtype=np.float32)
    original_spacing_zyx = original_spacing_xyz[::-1]
    target_spacing_zyx = target_spacing_xyz[::-1]
    original_shape_zyx = np.asarray(volume_hu_zyx.shape, dtype=np.float32)

    new_shape_zyx = np.round(original_shape_zyx * (original_spacing_zyx / target_spacing_zyx)).astype(int)
    new_shape_zyx = np.maximum(new_shape_zyx, 1)
    resize_factor_zyx = new_shape_zyx / original_shape_zyx
    new_spacing_zyx = original_spacing_zyx / resize_factor_zyx

    resampled = ndimage.zoom(volume_hu_zyx, zoom=resize_factor_zyx, order=order)
    return resampled, new_spacing_zyx[::-1].astype(np.float32)


def normalise_hu(volume_hu, clip_min=HU_CLIP_MIN, clip_max=HU_CLIP_MAX):
    import numpy as np

    volume = np.clip(volume_hu, clip_min, clip_max).astype(np.float32)
    volume = (volume - clip_min) / float(clip_max - clip_min)
    return volume.astype(np.float32)


def clear_border_2d(mask):
    import numpy as np
    from scipy import ndimage

    labels, _count = ndimage.label(mask)
    if labels.size == 0:
        return mask

    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())
    border_labels.discard(0)
    if not border_labels:
        return mask
    return mask & ~np.isin(labels, list(border_labels))


def keep_largest_components(mask, max_components=2, min_component_size=10000):
    import numpy as np
    from scipy import ndimage

    labels, component_count = ndimage.label(mask)
    if component_count == 0:
        return mask

    component_sizes = ndimage.sum(mask, labels, index=list(range(1, component_count + 1)))
    ranked = []
    for label_id, size in enumerate(component_sizes, start=1):
        if size >= min_component_size:
            ranked.append((label_id, size))
    ranked = sorted(ranked, key=lambda item: item[1], reverse=True)[:max_components]

    if not ranked:
        return mask

    keep_labels = [item[0] for item in ranked]
    return np.isin(labels, keep_labels)


def segment_lung_parenchyma(volume_hu_zyx, threshold_hu=LUNG_THRESHOLD_HU):
    import numpy as np
    from scipy import ndimage

    air_mask = volume_hu_zyx < threshold_hu
    labels, _component_count = ndimage.label(air_mask)

    border_labels = set()
    border_labels.update(np.unique(labels[0, :, :]).tolist())
    border_labels.update(np.unique(labels[-1, :, :]).tolist())
    border_labels.update(np.unique(labels[:, 0, :]).tolist())
    border_labels.update(np.unique(labels[:, -1, :]).tolist())
    border_labels.update(np.unique(labels[:, :, 0]).tolist())
    border_labels.update(np.unique(labels[:, :, -1]).tolist())
    border_labels.discard(0)

    internal_air = air_mask.copy()
    if border_labels:
        internal_air[np.isin(labels, list(border_labels))] = False

    if int(internal_air.sum()) < 10000:
        internal_air = np.zeros_like(air_mask, dtype=bool)
        for z in range(air_mask.shape[0]):
            internal_air[z] = clear_border_2d(air_mask[z])

    lung_mask = keep_largest_components(internal_air, max_components=2, min_component_size=10000)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    lung_mask = ndimage.binary_closing(lung_mask, structure=structure, iterations=2)
    lung_mask = ndimage.binary_opening(lung_mask, structure=structure, iterations=1)

    filled = np.zeros_like(lung_mask, dtype=bool)
    for z in range(lung_mask.shape[0]):
        filled[z] = ndimage.binary_fill_holes(lung_mask[z])

    return filled.astype(np.uint8)


def world_to_voxel(world_coord_xyz, origin_xyz, spacing_xyz, transform_matrix):
    import numpy as np

    world_coord_xyz = np.asarray(world_coord_xyz, dtype=np.float32)
    origin_xyz = np.asarray(origin_xyz, dtype=np.float32)
    spacing_xyz = np.asarray(spacing_xyz, dtype=np.float32)
    transform_matrix = np.asarray(transform_matrix, dtype=np.float32).reshape(3, 3)

    relative = world_coord_xyz - origin_xyz
    voxel_continuous_xyz = np.linalg.inv(transform_matrix).dot(relative) / spacing_xyz
    voxel_xyz = np.round(voxel_continuous_xyz).astype(int)
    voxel_zyx = voxel_xyz[::-1]
    return voxel_xyz, voxel_zyx


def is_inside_volume_zyx(coord_zyx, shape_zyx):
    z, y, x = [int(v) for v in coord_zyx]
    return 0 <= z < shape_zyx[0] and 0 <= y < shape_zyx[1] and 0 <= x < shape_zyx[2]


def save_qc_png(anonymised_id, original_hu, resampled_hu, normalised, lung_mask, output_path):
    plt = optional_matplotlib()
    if plt is None:
        return False

    z_orig = original_hu.shape[0] // 2
    z_res = resampled_hu.shape[0] // 2
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original_hu[z_orig], cmap="gray", vmin=-1000, vmax=400)
    axes[0].set_title("Original HU")
    axes[1].imshow(resampled_hu[z_res], cmap="gray", vmin=-1000, vmax=400)
    axes[1].set_title("Resampled HU")
    axes[2].imshow(normalised[z_res], cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Normalised")
    axes[3].imshow(resampled_hu[z_res], cmap="gray", vmin=-1000, vmax=400)
    axes[3].imshow(lung_mask[z_res], alpha=0.35)
    axes[3].set_title("Lung mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(anonymised_id)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=160)
    plt.close()
    return True


def annotations_for_series(annotations):
    grouped = defaultdict(list)
    for row in annotations:
        seriesuid = clean_value(row.get("seriesuid"))
        if seriesuid:
            grouped[seriesuid].append(row)
    return grouped


def validate_annotation_coordinates(seriesuid, annotations, info, original_shape_zyx, resampled_shape_zyx, new_spacing_xyz):
    import numpy as np

    rows = []
    origin_xyz = info["origin_xyz"]
    original_spacing_xyz = info["spacing_xyz"]
    transform_matrix = info["transform_matrix"]
    anonymised_id = info["anonymised_id"]

    for ann in annotations:
        world_coord = [
            safe_float(ann.get("coordX")),
            safe_float(ann.get("coordY")),
            safe_float(ann.get("coordZ")),
        ]
        if any(value is None for value in world_coord):
            continue

        original_voxel_xyz, original_voxel_zyx = world_to_voxel(
            world_coord,
            origin_xyz,
            original_spacing_xyz,
            transform_matrix,
        )
        resampled_voxel_xyz, resampled_voxel_zyx = world_to_voxel(
            world_coord,
            origin_xyz,
            new_spacing_xyz,
            transform_matrix,
        )

        rows.append({
            "seriesuid": seriesuid,
            "anonymised_id": anonymised_id,
            "coordX": world_coord[0],
            "coordY": world_coord[1],
            "coordZ": world_coord[2],
            "diameter_mm": safe_float(ann.get("diameter_mm")),
            "original_voxel_x": int(original_voxel_xyz[0]),
            "original_voxel_y": int(original_voxel_xyz[1]),
            "original_voxel_z": int(original_voxel_xyz[2]),
            "original_inside_volume": is_inside_volume_zyx(original_voxel_zyx, tuple(original_shape_zyx)),
            "resampled_voxel_x": int(resampled_voxel_xyz[0]),
            "resampled_voxel_y": int(resampled_voxel_xyz[1]),
            "resampled_voxel_z": int(resampled_voxel_xyz[2]),
            "resampled_inside_volume": is_inside_volume_zyx(resampled_voxel_zyx, tuple(resampled_shape_zyx)),
        })
    return rows


def lung_mask_qc_status(lung_fraction):
    if lung_fraction < 0.01:
        return "review_low_lung_fraction"
    if lung_fraction > 0.75:
        return "review_high_lung_fraction"
    return "ok"


def preprocess_single_scan(record, annotations_by_series, output_dirs, params):
    import numpy as np

    table_dir, volume_dir, figure_dir = output_dirs
    volume, info = read_luna_volume(record["mhd_path"])
    original_hu = volume.astype(np.int16, copy=False)
    original_shape = tuple(int(v) for v in original_hu.shape)
    original_spacing_xyz = tuple(float(v) for v in info["spacing_xyz"])

    resampled_hu, new_spacing_xyz = resample_volume_to_spacing(
        original_hu,
        original_spacing_xyz=original_spacing_xyz,
        target_spacing_xyz=params["target_spacing_xyz"],
        order=1,
    )
    resampled_hu = resampled_hu.astype(np.float32)
    normalised = normalise_hu(resampled_hu, params["hu_clip_min"], params["hu_clip_max"])
    lung_mask = segment_lung_parenchyma(resampled_hu, params["lung_threshold_hu"])

    lung_voxels = int(lung_mask.sum())
    total_voxels = int(lung_mask.size)
    lung_fraction = float(lung_voxels / float(total_voxels)) if total_voxels else 0.0
    anonymised_id = info["anonymised_id"]

    output_npz = ""
    if params["save_volumes"]:
        output_npz_path = volume_dir / "{}.npz".format(anonymised_id)
        np.savez_compressed(
            str(output_npz_path),
            image=normalised.astype(np.float16),
            lung_mask=lung_mask.astype(np.uint8),
            spacing_xyz=np.asarray(new_spacing_xyz, dtype=np.float32),
            origin_xyz=np.asarray(info["origin_xyz"], dtype=np.float32),
            transform_matrix=np.asarray(info["transform_matrix"], dtype=np.float32).reshape(3, 3),
        )
        output_npz = str(output_npz_path)

    qc_saved = False
    if params["qc_index"] < params["max_qc_png"]:
        qc_path = figure_dir / "{}_qc.png".format(anonymised_id)
        qc_saved = save_qc_png(anonymised_id, original_hu, resampled_hu, normalised, lung_mask, qc_path)

    coord_rows = validate_annotation_coordinates(
        info["seriesuid"],
        annotations_by_series.get(info["seriesuid"], []),
        info,
        original_shape,
        tuple(int(v) for v in resampled_hu.shape),
        new_spacing_xyz,
    )

    summary = {
        "seriesuid": info["seriesuid"],
        "anonymised_id": anonymised_id,
        "subset": record.get("subset", ""),
        "original_shape_z": original_shape[0],
        "original_shape_y": original_shape[1],
        "original_shape_x": original_shape[2],
        "original_spacing_x": original_spacing_xyz[0],
        "original_spacing_y": original_spacing_xyz[1],
        "original_spacing_z": original_spacing_xyz[2],
        "resampled_shape_z": int(resampled_hu.shape[0]),
        "resampled_shape_y": int(resampled_hu.shape[1]),
        "resampled_shape_x": int(resampled_hu.shape[2]),
        "new_spacing_x": float(new_spacing_xyz[0]),
        "new_spacing_y": float(new_spacing_xyz[1]),
        "new_spacing_z": float(new_spacing_xyz[2]),
        "original_hu_min": float(np.min(original_hu)),
        "original_hu_max": float(np.max(original_hu)),
        "original_hu_mean": float(np.mean(original_hu)),
        "resampled_hu_min": float(np.min(resampled_hu)),
        "resampled_hu_max": float(np.max(resampled_hu)),
        "resampled_hu_mean": float(np.mean(resampled_hu)),
        "normalised_min": float(np.min(normalised)),
        "normalised_max": float(np.max(normalised)),
        "normalised_mean": float(np.mean(normalised)),
        "lung_voxels": lung_voxels,
        "total_voxels": total_voxels,
        "lung_fraction": lung_fraction,
        "lung_mask_qc_status": lung_mask_qc_status(lung_fraction),
        "annotation_coordinate_count": len(coord_rows),
        "saved_full_volume": bool(params["save_volumes"]),
        "output_npz": output_npz,
        "qc_png_saved": bool(qc_saved),
    }

    del volume, original_hu, resampled_hu, normalised, lung_mask
    gc.collect()
    return summary, coord_rows


def select_scan_records(inventory, max_scans=None):
    rows = [row for row in inventory if row.get("raw_exists") is True or clean_value(row.get("raw_exists")).lower() == "true"]
    rows = sorted(rows, key=lambda row: (row.get("subset", ""), row.get("seriesuid", "")))
    if max_scans is not None:
        rows = rows[:int(max_scans)]
    return rows


def write_report(path, params, inventory, preprocessing_rows, coord_rows):
    attempted = len(preprocessing_rows)
    successful = [row for row in preprocessing_rows if not clean_value(row.get("error"))]
    failed = attempted - len(successful)
    lung_fractions = [safe_float(row.get("lung_fraction")) for row in successful]
    lung_fractions = [value for value in lung_fractions if value is not None]
    coords_checked = len(coord_rows)
    coords_inside = [
        row for row in coord_rows
        if clean_value(row.get("resampled_inside_volume")).lower() == "true"
        or row.get("resampled_inside_volume") is True
    ]

    with Path(path).open("w", encoding="utf-8") as f:
        f.write("# LUNA16 {} Preprocessing Report\n\n".format(params["subset_label"]))
        f.write("## Scope\n\n")
        f.write("- Raw source: `{}`\n".format(params["luna_root"]))
        f.write("- Subsets: `{}`\n".format(",".join("subset{}".format(s) for s in params["subsets"])))
        f.write("- Output directory: `{}`\n".format(params["output_dir"]))
        f.write("- Full preprocessed `.npz` saved: `{}`\n\n".format(params["save_volumes"]))
        f.write("## Pipeline\n\n")
        f.write("- De-identification: SHA1 anonymised IDs for output filenames.\n")
        f.write("- Standard spacing: `{}` mm XYZ.\n".format(params["target_spacing_xyz"]))
        f.write("- HU clipping/normalisation: `[{}, {}] -> [0, 1]`.\n".format(params["hu_clip_min"], params["hu_clip_max"]))
        f.write("- Lung segmentation: HU threshold `{}` with connected-component cleanup.\n\n".format(params["lung_threshold_hu"]))
        f.write("## Counts\n\n")
        f.write("- Source scans discovered: `{}`\n".format(len(inventory)))
        f.write("- Attempted scans: `{}`\n".format(attempted))
        f.write("- Successful scans: `{}`\n".format(len(successful)))
        f.write("- Failed scans: `{}`\n".format(failed))
        f.write("- Annotation coordinates checked: `{}`\n".format(coords_checked))
        if coords_checked:
            f.write("- Resampled annotation coordinates inside volume: `{:.2f}%`\n".format(100.0 * len(coords_inside) / float(coords_checked)))
        if lung_fractions:
            f.write("- Mean lung mask fraction: `{:.4f}`\n".format(sum(lung_fractions) / float(len(lung_fractions))))


def preprocess_luna(args):
    output_dir = Path(args.output_dir)
    table_dir, volume_dir, figure_dir = ensure_dirs(output_dir)
    annotations_path = Path(args.luna_root) / "annotations.csv"
    if not annotations_path.exists():
        raise FileNotFoundError("Missing annotations.csv: {}".format(annotations_path))

    inventory = discover_subset_mhds(args.luna_root, args.subsets)

    output_prefix = getattr(args, "output_prefix", DEFAULT_OUTPUT_PREFIX)
    subset_label = "{}-{}".format(min(args.subsets), max(args.subsets)) if args.subsets else "custom"

    write_csv(output_table_path(table_dir, output_prefix, "manifest"), inventory)
    write_csv(
        output_table_path(table_dir, output_prefix, "deid_map"),
        [{"seriesuid": row["seriesuid"], "anonymised_id": row["anonymised_id"], "subset": row["subset"]} for row in inventory],
    )

    annotations = read_csv_rows(annotations_path)
    annotations_by = annotations_for_series(annotations)
    scan_records = select_scan_records(inventory, max_scans=args.max_scans)

    params = {
        "luna_root": str(args.luna_root),
        "output_dir": str(output_dir),
        "output_prefix": output_prefix,
        "subset_label": subset_label,
        "subsets": args.subsets,
        "target_spacing_xyz": tuple(float(v) for v in args.target_spacing),
        "hu_clip_min": args.hu_min,
        "hu_clip_max": args.hu_max,
        "lung_threshold_hu": args.lung_threshold,
        "save_volumes": args.save_volumes,
        "max_qc_png": args.max_qc_png,
        "qc_index": 0,
    }

    write_csv(output_table_path(table_dir, output_prefix, "processing_manifest"), scan_records)
    (table_dir / "{}_params.json".format(output_prefix)).write_text(json.dumps(params, indent=2), encoding="utf-8")

    if args.manifest_only:
        log("Manifest-only run complete. Preprocessing manifest and de-id map were written.")
        return [], []

    require_scientific_stack()

    preprocessing_rows = []
    coordinate_rows = []
    total = len(scan_records)
    log("Processing {} LUNA scans from subsets {}.".format(total, args.subsets))

    for index, record in enumerate(scan_records, start=1):
        log("[{}/{}] {}".format(index, total, record["seriesuid"]))
        try:
            summary, coord_rows = preprocess_single_scan(
                record,
                annotations_by,
                (table_dir, volume_dir, figure_dir),
                params,
            )
            preprocessing_rows.append(summary)
            coordinate_rows.extend(coord_rows)
        except Exception as exc:
            preprocessing_rows.append({
                "seriesuid": record.get("seriesuid", ""),
                "anonymised_id": record.get("anonymised_id", ""),
                "subset": record.get("subset", ""),
                "error": repr(exc),
            })
            log("  Failed: {}".format(repr(exc)))
        params["qc_index"] += 1

    write_csv(output_table_path(table_dir, output_prefix, "preprocess_summary"), preprocessing_rows)
    write_csv(output_table_path(table_dir, output_prefix, "coord_validation"), coordinate_rows)
    write_report(table_dir / "{}_preprocess_report.md".format(output_prefix), params, inventory, preprocessing_rows, coordinate_rows)
    return preprocessing_rows, coordinate_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess unzipped LUNA16 subset0-4 CT volumes.")
    parser.add_argument("--luna-root", type=Path, default=DEFAULT_LUNA_ROOT, help="Root containing subset0 ... subset4 and annotations.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Project-local output directory.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Prefix for generated preprocessing tables.")
    parser.add_argument("--subsets", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Subset numbers to process.")
    parser.add_argument("--max-scans", type=int, default=None, help="Optional limit for runtime testing.")
    parser.add_argument("--target-spacing", type=float, nargs=3, default=list(TARGET_SPACING_XYZ), metavar=("X", "Y", "Z"))
    parser.add_argument("--hu-min", type=int, default=HU_CLIP_MIN, help="HU lower clipping bound.")
    parser.add_argument("--hu-max", type=int, default=HU_CLIP_MAX, help="HU upper clipping bound.")
    parser.add_argument("--lung-threshold", type=int, default=LUNG_THRESHOLD_HU, help="HU threshold for lung air segmentation.")
    parser.add_argument("--save-volumes", action="store_true", help="Save normalised image and lung mask NPZ files. Off by default to protect disk space.")
    parser.add_argument("--max-qc-png", type=int, default=5, help="Maximum QC PNG figures to save if matplotlib is installed.")
    parser.add_argument("--manifest-only", action="store_true", help="Only write preprocessing manifest/de-id map; do not read volume data.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log("LUNA root: {}".format(args.luna_root))
    log("Output dir: {}".format(args.output_dir))
    rows, coord_rows = preprocess_luna(args)
    if args.manifest_only:
        return

    success = [row for row in rows if not clean_value(row.get("error"))]
    log("")
    log("LUNA preprocessing complete")
    log("  attempted scans: {}".format(len(rows)))
    log("  successful scans: {}".format(len(success)))
    log("  annotation coordinates checked: {}".format(len(coord_rows)))
    log("  outputs: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
