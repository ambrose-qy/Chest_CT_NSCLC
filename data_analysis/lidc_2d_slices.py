"""
Build 2D LIDC-IDRI nodule slice manifests for CNN baselines.

This script consumes the process4 train/validation/test split and writes
one row per ROI with:

* the resolved raw DICOM series directory,
* the DICOM slice nearest the nodule's largest XML contour slice, and
* the consensus ROI crop box and malignancy label.

Run from the repository root:

    conda run -n torch-gpu python data_analysis/lidc_2d_slices.py

The output is used by the 2D Lightning model files such as LIDC_2d_resnet.py
and LIDC_2d_densenet.py.
"""

from __future__ import print_function

import argparse
import csv
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIDC_ROOT = PROJECT_ROOT / "data" / "raw" / "LIDC"
TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "tables"

SPLIT_TABLES = {
    "binary": TABLE_DIR / "lidc_roi_binary_split_manifest.csv",
    "multiclass": TABLE_DIR / "lidc_roi_multiclass_split_manifest.csv",
}
READER_ROI_TABLE = TABLE_DIR / "lidc_roi_reader_annotation_manifest.csv"
DEFAULT_OUTPUT_TABLES = {
    "binary": TABLE_DIR / "lidc_roi_binary_2d_slice_manifest.csv",
    "multiclass": TABLE_DIR / "lidc_roi_multiclass_2d_slice_manifest.csv",
}
DEFAULT_OUTPUT_TABLE = DEFAULT_OUTPUT_TABLES["binary"]


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


def image_z_from_dataset(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return safe_float(ipp[2])
    return safe_float(getattr(ds, "SliceLocation", ""))


def read_series_slice_headers(series_dir):
    try:
        import pydicom
    except Exception as exc:
        raise ImportError("pydicom is required to build the 2D slice manifest.") from exc

    headers = []
    for path in sorted(Path(series_dir).iterdir()):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        z = image_z_from_dataset(ds)
        if z is None:
            continue
        instance_number = safe_int(getattr(ds, "InstanceNumber", ""))
        headers.append({
            "dicom_path": str(path),
            "slice_z_mm": z,
            "instance_number": instance_number if instance_number is not None else "",
        })

    headers.sort(key=lambda row: (row["slice_z_mm"], row["instance_number"] or 0))
    return headers


def nearest_slice(headers, target_z):
    if target_z is None or not headers:
        return None
    return min(headers, key=lambda row: abs(row["slice_z_mm"] - target_z))


def local_name(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def get_child_text(parent, target_name):
    if parent is None:
        return ""
    for child in parent:
        if local_name(child.tag) == target_name:
            return child.text or ""
    return ""


def contour_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def bbox_area(points):
    if not points:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (max(xs) - min(xs) + 1.0) * (max(ys) - min(ys) + 1.0)


def roi_contours_for_annotation(reader_row, xml_cache):
    xml_path = Path(clean_value(reader_row.get("xml_file")))
    reader_id = safe_int(reader_row.get("reader_id"))
    nodule_id = clean_value(reader_row.get("nodule_id"))
    if not xml_path.exists() or reader_id is None or not nodule_id:
        return []

    if xml_path not in xml_cache:
        xml_cache[xml_path] = ET.parse(str(xml_path)).getroot()

    root = xml_cache[xml_path]
    sessions = [elem for elem in root.iter() if local_name(elem.tag) == "readingSession"]
    if reader_id < 1 or reader_id > len(sessions):
        return []

    session = sessions[reader_id - 1]
    contours = []
    for nodule in session:
        if local_name(nodule.tag) != "unblindedReadNodule":
            continue
        if clean_value(get_child_text(nodule, "noduleID")) != nodule_id:
            continue

        for roi in nodule:
            if local_name(roi.tag) != "roi":
                continue
            z = safe_float(get_child_text(roi, "imageZposition"))
            points = []
            for edge_map in roi:
                if local_name(edge_map.tag) != "edgeMap":
                    continue
                x = safe_float(get_child_text(edge_map, "xCoord"))
                y = safe_float(get_child_text(edge_map, "yCoord"))
                if x is not None and y is not None:
                    points.append((x, y))
            if z is None or not points:
                continue

            poly_area = contour_area(points)
            box_area = bbox_area(points)
            contours.append({
                "z_mm": z,
                "area_px": max(poly_area, box_area),
                "polygon_area_px": poly_area,
                "bbox_area_px": box_area,
                "edge_point_count": len(points),
                "reader_id": reader_id,
                "nodule_id": nodule_id,
                "xml_file": str(xml_path),
            })
        break

    return contours


def largest_cross_section_for_roi(roi_id, reader_rows_by_roi, xml_cache):
    best = None
    for reader_row in reader_rows_by_roi.get(roi_id, []):
        for contour in roi_contours_for_annotation(reader_row, xml_cache):
            if best is None or contour["area_px"] > best["area_px"]:
                best = contour
    return best


def field_value(row, name):
    value = clean_value(row.get(name))
    return value


def build_slice_manifest(
    split_path=None,
    reader_roi_path=READER_ROI_TABLE,
    output_path=None,
    lidc_root=LIDC_ROOT,
    task="binary",
    max_rows=None,
):
    if task not in ("binary", "multiclass"):
        raise ValueError("task must be binary or multiclass.")

    split_path = Path(split_path) if split_path is not None else SPLIT_TABLES[task]
    output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_TABLES[task]
    split_rows = read_csv_rows(split_path)
    reader_rows = read_csv_rows(reader_roi_path)
    split_col = "binary_split" if task == "binary" else "multiclass_split"
    label_col = "binary_label_id" if task == "binary" else "multiclass_risk_label_id"

    reader_rows_by_roi = defaultdict(list)
    for row in reader_rows:
        roi_id = clean_value(row.get("roi_id"))
        if roi_id:
            reader_rows_by_roi[roi_id].append(row)

    manifest_root = discover_manifest_root(lidc_root)
    series_dir_by_uid = build_series_dir_map(manifest_root)
    xml_cache = {}
    header_cache = {}

    output_rows = []
    failures = defaultdict(int)
    selected_rows = split_rows[:max_rows] if max_rows is not None else split_rows

    for index, row in enumerate(selected_rows, start=1):
        if index % 100 == 0:
            log("  Processed {}/{} {} ROIs".format(index, len(selected_rows), task))

        roi_id = clean_value(row.get("roi_id"))
        series_uid = clean_value(row.get("SeriesInstanceUID"))
        split_name = clean_value(row.get(split_col))
        label_id = safe_int(row.get(label_col))

        if not roi_id or not series_uid or not split_name or label_id is None:
            failures["missing_required_fields"] += 1
            continue

        series_dir = series_dir_by_uid.get(series_uid)
        if series_dir is None or not Path(series_dir).exists():
            failures["missing_series_dir"] += 1
            continue

        if series_uid not in header_cache:
            header_cache[series_uid] = read_series_slice_headers(series_dir)
        headers = header_cache[series_uid]
        if not headers:
            failures["missing_slice_headers"] += 1
            continue

        largest = largest_cross_section_for_roi(roi_id, reader_rows_by_roi, xml_cache)
        target_z = largest["z_mm"] if largest is not None else safe_float(row.get("z_center_mm_consensus"))
        z_source = "largest_xml_contour" if largest is not None else "consensus_z_center"
        chosen = nearest_slice(headers, target_z)
        if chosen is None:
            failures["missing_target_slice"] += 1
            continue

        output = {
            "roi_id": roi_id,
            "SeriesInstanceUID": series_uid,
            "StudyInstanceUID": field_value(row, "StudyInstanceUID"),
            "patient_folder": field_value(row, "patient_folder"),
            "PatientID": field_value(row, "PatientID"),
            "binary_split": field_value(row, "binary_split"),
            "binary_label": field_value(row, "binary_label"),
            "binary_label_id": field_value(row, "binary_label_id"),
            "multiclass_split": field_value(row, "multiclass_split"),
            "multiclass_risk_label": field_value(row, "multiclass_risk_label"),
            "multiclass_risk_label_id": field_value(row, "multiclass_risk_label_id"),
            "reader_count": field_value(row, "reader_count"),
            "median_malignancy_score": field_value(row, "median_malignancy_score"),
            "label_confidence": field_value(row, "label_confidence"),
            "median_max_diameter_mm": field_value(row, "median_max_diameter_mm"),
            "x_center_px_consensus": field_value(row, "x_center_px_consensus"),
            "y_center_px_consensus": field_value(row, "y_center_px_consensus"),
            "z_center_mm_consensus": field_value(row, "z_center_mm_consensus"),
            "x_min_px_roi": field_value(row, "x_min_px_roi"),
            "x_max_px_roi": field_value(row, "x_max_px_roi"),
            "y_min_px_roi": field_value(row, "y_min_px_roi"),
            "y_max_px_roi": field_value(row, "y_max_px_roi"),
            "Rows": field_value(row, "Rows"),
            "Columns": field_value(row, "Columns"),
            "x_spacing_mm": field_value(row, "x_spacing_mm"),
            "y_spacing_mm": field_value(row, "y_spacing_mm"),
            "SliceThickness": field_value(row, "SliceThickness"),
            "series_dir": str(series_dir),
            "dicom_path": chosen["dicom_path"],
            "target_z_mm": target_z if target_z is not None else "",
            "selected_slice_z_mm": chosen["slice_z_mm"],
            "selected_instance_number": chosen["instance_number"],
            "target_slice_abs_error_mm": abs(chosen["slice_z_mm"] - target_z) if target_z is not None else "",
            "target_z_source": z_source,
            "largest_contour_area_px": largest["area_px"] if largest is not None else "",
            "largest_contour_reader_id": largest["reader_id"] if largest is not None else "",
            "largest_contour_nodule_id": largest["nodule_id"] if largest is not None else "",
            "largest_contour_edge_point_count": largest["edge_point_count"] if largest is not None else "",
        }
        output_rows.append(output)

    write_csv(output_path, output_rows)
    log("")
    log("2D LIDC {} slice manifest complete".format(task))
    log("  input {} ROIs: {}".format(task, len(selected_rows)))
    log("  output rows: {}".format(len(output_rows)))
    log("  failures: {}".format(dict(failures)))
    log("  output: {}".format(output_path))
    return output_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build 2D DICOM slice manifest for LIDC-IDRI CNN baselines.")
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"], help="Classification task.")
    parser.add_argument("--split", type=Path, default=None, help="Process4 split manifest. Defaults to the task-specific table.")
    parser.add_argument("--binary-split", type=Path, default=None, help="Deprecated alias for --split.")
    parser.add_argument("--reader-roi", type=Path, default=READER_ROI_TABLE, help="Process3 reader ROI manifest.")
    parser.add_argument("--output", type=Path, default=None, help="Output 2D slice manifest CSV.")
    parser.add_argument("--lidc-root", type=Path, default=LIDC_ROOT, help="Root containing raw LIDC manifest data.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional debug limit.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build_slice_manifest(
        split_path=args.split or args.binary_split,
        reader_roi_path=args.reader_roi,
        output_path=args.output,
        lidc_root=args.lidc_root,
        task=args.task,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
