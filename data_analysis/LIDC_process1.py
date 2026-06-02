"""
LIDC-IDRI raw-data exploration.

This script analyses the local LIDC-IDRI release from TCIA-style raw files:

* manifest metadata.csv for subject/series/file-location indexing
* CT DICOM headers for patient age, patient sex, spacing, and image geometry
* LIDC XML reader annotations for nodule size, coarse location, and morphology

Dataset background used for this workflow:

* The LIDC-IDRI collection contains thoracic CT scans and XML annotations from
  up to four radiologists.
* XML annotations distinguish nodules >= 3 mm, nodules < 3 mm, and non-nodules
  >= 3 mm. The ordinal morphology scores are recorded for annotated nodules
  with the LIDC "characteristics" block.
* Morphology scores include subtlety, internal structure, calcification,
  sphericity, margin, lobulation, spiculation, texture, and malignancy.

Run from the repository root:

    conda run -n torch-gpu python data_analysis/LIDC_process1.py

Useful options:

    conda run -n torch-gpu python data_analysis/LIDC_process1.py --force
    conda run -n torch-gpu python data_analysis/LIDC_process1.py --skip-slice-ranges

Outputs are written under:

    data/processed/tables/
    data/processed/figures/   (only when matplotlib is installed)
"""

from __future__ import print_function

import argparse
import csv
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIDC_ROOT = PROJECT_ROOT / "data" / "raw" / "LIDC"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROCESSED_DIR / "tables"
FIGURE_DIR = PROCESSED_DIR / "figures"

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

NUMERIC_ANNOTATION_COLS = MORPHOLOGY_COLS + [
    "reader_id",
    "roi_count",
    "edge_point_count",
    "z_min",
    "z_max",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
]

LONG_VR = set(["OB", "OD", "OF", "OL", "OV", "OW", "SQ", "UC", "UR", "UT", "UN"])

DICOM_TAGS = {
    (0x0002, 0x0010): "TransferSyntaxUID",
    (0x0008, 0x0016): "SOPClassUID",
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0008, 0x0060): "Modality",
    (0x0008, 0x0070): "Manufacturer",
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0010, 0x0040): "PatientSex",
    (0x0010, 0x1010): "PatientAge",
    (0x0018, 0x0050): "SliceThickness",
    (0x0018, 0x1210): "ConvolutionKernel",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0020, 0x0013): "InstanceNumber",
    (0x0020, 0x0032): "ImagePositionPatient",
    (0x0020, 0x0037): "ImageOrientationPatient",
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
    (0x0028, 0x0030): "PixelSpacing",
}

NEEDED_DICOM_FIELDS = set(DICOM_TAGS.values())


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def log(message):
    print(message)
    sys.stdout.flush()


def as_text(value):
    if value is None:
        return ""
    return str(value)


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


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    if fieldnames is None:
        seen = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def discover_manifest_root():
    preferred = LIDC_ROOT / "manifest-1600709154662"
    if preferred.exists():
        return preferred
    candidates = sorted([p for p in LIDC_ROOT.glob("manifest-*") if p.is_dir()])
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Could not find data/raw/LIDC/manifest-*")


def discover_xml_root(manifest_root):
    xml_only = LIDC_ROOT / "tcia-lidc-xml"
    if xml_only.exists():
        return xml_only
    dicom_root = manifest_root / "LIDC-IDRI"
    if dicom_root.exists():
        return dicom_root
    return LIDC_ROOT


def metadata_series_dir(manifest_root, file_location):
    rel = clean_value(file_location).replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    if rel == "":
        return None
    return manifest_root / rel


def parse_patient_age_to_years(value):
    text = clean_value(value).upper()
    if text == "":
        return None

    match = re.match(r"^(\d+(?:\.\d+)?)([YMWD]?)$", text)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2) or "Y"
    if unit == "Y":
        return amount
    if unit == "M":
        return amount / 12.0
    if unit == "W":
        return amount / 52.1775
    if unit == "D":
        return amount / 365.25
    return None


def clean_patient_sex(value):
    text = clean_value(value).upper()
    if text in ("M", "F", "O"):
        return text
    return None


def split_numeric_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            number = safe_float(item)
            if number is not None:
                result.append(number)
        return result
    text = clean_value(value)
    if text == "":
        return []
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]


def parse_pixel_spacing(value):
    parts = split_numeric_list(value)
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def nodule_size_category(diameter_mm):
    if diameter_mm is None:
        return ""
    if diameter_mm < 3:
        return "<3 mm"
    if diameter_mm < 6:
        return "3-<6 mm"
    if diameter_mm < 10:
        return "6-<10 mm"
    if diameter_mm < 30:
        return "10-<30 mm"
    return ">=30 mm"


def local_name(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def find_first_text(root, target_name):
    for elem in root.iter():
        if local_name(elem.tag) == target_name:
            return elem.text
    return None


def get_child_text(parent, target_name):
    for child in parent:
        if local_name(child.tag) == target_name:
            return child.text
    return None


def decode_dicom_string(raw):
    return raw.decode("latin-1", errors="ignore").replace("\x00", "").strip()


def parse_dicom_value(vr, raw):
    if raw is None:
        return ""

    if vr in ("US",):
        if len(raw) >= 2:
            return struct.unpack("<H", raw[:2])[0]
        return ""

    if vr in ("SS",):
        if len(raw) >= 2:
            return struct.unpack("<h", raw[:2])[0]
        return ""

    if vr in ("UL",):
        if len(raw) >= 4:
            return struct.unpack("<L", raw[:4])[0]
        return ""

    if vr in ("SL",):
        if len(raw) >= 4:
            return struct.unpack("<l", raw[:4])[0]
        return ""

    return decode_dicom_string(raw)


def read_dicom_header(file_path):
    """
    Minimal DICOM metadata reader.

    pydicom is preferable when installed, but this lightweight parser is enough
    for standard TCIA LIDC headers and keeps the script runnable in a bare
    Python environment. Pixel data is never read.
    """
    file_path = Path(file_path)
    fields = {}

    try:
        with file_path.open("rb") as f:
            prefix = f.read(132)
            if len(prefix) >= 132 and prefix[128:132] == b"DICM":
                pos = 132
            else:
                pos = 0
                f.seek(0)

            transfer_syntax = "1.2.840.10008.1.2.1"
            explicit_vr = True
            little_endian = True

            while True:
                f.seek(pos)
                tag_raw = f.read(4)
                if len(tag_raw) < 4:
                    break
                group, element = struct.unpack("<HH", tag_raw)

                if explicit_vr:
                    vr_raw = f.read(2)
                    if len(vr_raw) < 2:
                        break
                    vr = vr_raw.decode("ascii", errors="ignore")
                    if vr in LONG_VR:
                        f.read(2)
                        length_raw = f.read(4)
                        if len(length_raw) < 4:
                            break
                        length = struct.unpack("<L", length_raw)[0]
                    else:
                        length_raw = f.read(2)
                        if len(length_raw) < 2:
                            break
                        length = struct.unpack("<H", length_raw)[0]
                else:
                    vr = ""
                    length_raw = f.read(4)
                    if len(length_raw) < 4:
                        break
                    length = struct.unpack("<L", length_raw)[0]

                value_pos = f.tell()
                next_pos = value_pos + length if length != 0xFFFFFFFF else value_pos

                if (group, element) == (0x7FE0, 0x0010):
                    break

                if length != 0xFFFFFFFF and length <= 1024 * 1024:
                    raw = f.read(length)
                    name = DICOM_TAGS.get((group, element))
                    if name:
                        value = parse_dicom_value(vr, raw)
                        fields[name] = value
                        if name == "TransferSyntaxUID":
                            transfer_syntax = clean_value(value)
                            explicit_vr = transfer_syntax != "1.2.840.10008.1.2"
                            little_endian = transfer_syntax != "1.2.840.10008.1.2.2"
                else:
                    if group != 0x0002 and fields:
                        break

                if length == 0xFFFFFFFF:
                    break

                pos = next_pos

                if not little_endian:
                    # Big-endian transfer syntax is uncommon in this data. The
                    # tags needed here are usually read before this matters.
                    pass

                if len(NEEDED_DICOM_FIELDS.intersection(fields.keys())) >= len(NEEDED_DICOM_FIELDS) - 1:
                    break

    except Exception:
        return {}

    return fields


def read_first_dicom_in_series(series_dir):
    if series_dir is None or not Path(series_dir).exists():
        return {}

    for path in sorted(Path(series_dir).iterdir()):
        if not path.is_file():
            continue
        header = read_dicom_header(path)
        if header:
            header["dicom_path"] = str(path)
            return header
    return {}


def read_slice_z_range(series_dir):
    if series_dir is None or not Path(series_dir).exists():
        return None, None, 0

    z_values = []
    read_count = 0
    for path in sorted(Path(series_dir).iterdir()):
        if not path.is_file():
            continue
        header = read_dicom_header(path)
        if not header:
            continue
        read_count += 1
        ipp = split_numeric_list(header.get("ImagePositionPatient"))
        if len(ipp) >= 3:
            z_values.append(ipp[2])
            continue
        z = safe_float(header.get("SliceLocation"))
        if z is not None:
            z_values.append(z)

    if not z_values:
        return None, None, read_count
    return min(z_values), max(z_values), read_count


def load_or_build_series_tables(manifest_root, force=False, scan_slice_ranges=True):
    metadata_path = manifest_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(str(metadata_path))

    lidc_metadata = read_csv_rows(metadata_path)
    write_csv(TABLE_DIR / "lidc_metadata_copy.csv", lidc_metadata)

    series_path = TABLE_DIR / "lidc_series_inventory.csv"
    main_path = TABLE_DIR / "lidc_main_ct_series.csv"

    if series_path.exists() and main_path.exists() and not force:
        log("Loading existing CT series tables.")
        return read_csv_rows(series_path), read_csv_rows(main_path), lidc_metadata

    log("Building CT series inventory from manifest metadata and DICOM headers.")
    ct_rows = [row for row in lidc_metadata if clean_value(row.get("Modality")).upper() == "CT"]

    series_records = []
    for idx, row in enumerate(ct_rows, start=1):
        if idx % 100 == 0:
            log("  DICOM header scan: {}/{} CT series".format(idx, len(ct_rows)))

        series_dir = metadata_series_dir(manifest_root, row.get("File Location"))
        header = read_first_dicom_in_series(series_dir)
        y_spacing, x_spacing = parse_pixel_spacing(header.get("PixelSpacing"))
        image_position = split_numeric_list(header.get("ImagePositionPatient"))
        image_orientation = split_numeric_list(header.get("ImageOrientationPatient"))

        z_min = z_max = None
        slice_header_count = 0
        if scan_slice_ranges:
            z_min, z_max, slice_header_count = read_slice_z_range(series_dir)

        record = {
            "patient_folder": row.get("Subject ID", ""),
            "PatientID": header.get("PatientID") or row.get("Subject ID", ""),
            "StudyInstanceUID": header.get("StudyInstanceUID") or row.get("Study UID", ""),
            "SeriesInstanceUID": header.get("SeriesInstanceUID") or row.get("Series UID", ""),
            "Modality": header.get("Modality") or row.get("Modality", ""),
            "Rows": header.get("Rows", ""),
            "Columns": header.get("Columns", ""),
            "SliceThickness": header.get("SliceThickness", ""),
            "PixelSpacing": header.get("PixelSpacing", ""),
            "y_spacing_mm": y_spacing if y_spacing is not None else "",
            "x_spacing_mm": x_spacing if x_spacing is not None else "",
            "Manufacturer": header.get("Manufacturer", row.get("Manufacturer", "")),
            "ConvolutionKernel": header.get("ConvolutionKernel", ""),
            "SOPClassUID": row.get("SOP Class UID", ""),
            "num_slices": row.get("Number of Images", ""),
            "FileSize": row.get("File Size", ""),
            "metadata_file_location": row.get("File Location", ""),
            "series_dir": str(series_dir) if series_dir is not None else "",
            "PatientAge_raw": header.get("PatientAge", ""),
            "PatientAge_years": parse_patient_age_to_years(header.get("PatientAge")),
            "PatientSex_raw": header.get("PatientSex", ""),
            "PatientSex": clean_patient_sex(header.get("PatientSex")) or "",
            "ImagePositionPatient": header.get("ImagePositionPatient", ""),
            "ImageOrientationPatient": header.get("ImageOrientationPatient", ""),
            "first_slice_x0_mm": image_position[0] if len(image_position) >= 1 else "",
            "first_slice_y0_mm": image_position[1] if len(image_position) >= 2 else "",
            "first_slice_z0_mm": image_position[2] if len(image_position) >= 3 else "",
            "slice_z_min_mm": z_min if z_min is not None else "",
            "slice_z_max_mm": z_max if z_max is not None else "",
            "slice_header_count": slice_header_count,
            "orientation_r1": image_orientation[0] if len(image_orientation) >= 6 else "",
            "orientation_r2": image_orientation[1] if len(image_orientation) >= 6 else "",
            "orientation_r3": image_orientation[2] if len(image_orientation) >= 6 else "",
            "orientation_c1": image_orientation[3] if len(image_orientation) >= 6 else "",
            "orientation_c2": image_orientation[4] if len(image_orientation) >= 6 else "",
            "orientation_c3": image_orientation[5] if len(image_orientation) >= 6 else "",
        }
        series_records.append(record)

    write_csv(series_path, series_records)

    best_by_patient = {}
    for row in series_records:
        patient = row.get("patient_folder")
        image_count = safe_int(row.get("num_slices")) or 0
        current = best_by_patient.get(patient)
        current_count = safe_int(current.get("num_slices")) if current else -1
        if current is None or image_count > (current_count or -1):
            best_by_patient[patient] = row

    main_records = [best_by_patient[key] for key in sorted(best_by_patient.keys())]
    write_csv(main_path, main_records)
    return series_records, main_records, lidc_metadata


def build_demographics(main_series, force=False):
    path = TABLE_DIR / "lidc_patient_demographics.csv"
    if path.exists() and not force:
        log("Loading existing patient demographics table.")
        demographics = read_csv_rows(path)
    else:
        demographics = []
        for row in main_series:
            age_years = safe_float(row.get("PatientAge_years"))
            sex = clean_patient_sex(row.get("PatientSex"))
            demographics.append({
                "patient_folder": row.get("patient_folder", ""),
                "PatientID": row.get("PatientID", ""),
                "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
                "PatientAge_raw": row.get("PatientAge_raw", ""),
                "PatientAge_years": age_years if age_years is not None else "",
                "PatientAge_source": "DICOM PatientAge" if age_years is not None else "",
                "PatientSex_raw": row.get("PatientSex_raw", ""),
                "PatientSex": sex or "",
                "PatientSex_source": "DICOM PatientSex" if sex else "",
                "ct_series_count": "",
                "max_image_count": row.get("num_slices", ""),
                "demographics_series_dir": row.get("series_dir", ""),
            })
        write_csv(path, demographics)

    total = len(demographics)
    age_available = sum(1 for row in demographics if safe_float(row.get("PatientAge_years")) is not None)
    sex_available = sum(1 for row in demographics if clean_patient_sex(row.get("PatientSex")))
    missingness = [
        {
            "field": "PatientAge_years",
            "available_count": age_available,
            "missing_count": total - age_available,
            "available_fraction": age_available / float(total) if total else "",
        },
        {
            "field": "PatientSex",
            "available_count": sex_available,
            "missing_count": total - sex_available,
            "available_fraction": sex_available / float(total) if total else "",
        },
    ]
    write_csv(TABLE_DIR / "lidc_patient_demographics_missingness.csv", missingness)

    sex_counts = Counter(clean_patient_sex(row.get("PatientSex")) or "missing" for row in demographics)
    write_csv(
        TABLE_DIR / "lidc_patient_sex_distribution.csv",
        [{"PatientSex": key, "count": sex_counts[key]} for key in sorted(sex_counts.keys())],
    )

    ages = [safe_float(row.get("PatientAge_years")) for row in demographics]
    ages = [age for age in ages if age is not None]
    age_summary = [{
        "patient_count": total,
        "available_age_count": len(ages),
        "mean_age_years": mean(ages) if ages else "",
        "median_age_years": median(ages) if ages else "",
        "p25_age_years": percentile(ages, 0.25) if ages else "",
        "p75_age_years": percentile(ages, 0.75) if ages else "",
        "min_age_years": min(ages) if ages else "",
        "max_age_years": max(ages) if ages else "",
    }]
    write_csv(TABLE_DIR / "lidc_patient_age_summary.csv", age_summary)
    return demographics


def parse_lidc_xml(xml_path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    series_uid = (
        find_first_text(root, "SeriesInstanceUid")
        or find_first_text(root, "SeriesInstanceUID")
        or find_first_text(root, "seriesInstanceUid")
        or ""
    )
    study_uid = (
        find_first_text(root, "StudyInstanceUID")
        or find_first_text(root, "StudyInstanceUid")
        or find_first_text(root, "studyInstanceUid")
        or ""
    )

    rows = []
    reading_sessions = [elem for elem in root.iter() if local_name(elem.tag) == "readingSession"]

    for reader_index, session in enumerate(reading_sessions, start=1):
        nodules = [elem for elem in session if local_name(elem.tag) == "unblindedReadNodule"]

        for nodule in nodules:
            characteristics = None
            for child in nodule:
                if local_name(child.tag) == "characteristics":
                    characteristics = child
                    break

            morphology = {}
            source_to_column = {
                "subtlety": "subtlety",
                "internalStructure": "internal_structure",
                "calcification": "calcification",
                "sphericity": "sphericity",
                "margin": "margin",
                "lobulation": "lobulation",
                "spiculation": "spiculation",
                "texture": "texture",
                "malignancy": "malignancy",
            }
            for source, column in source_to_column.items():
                morphology[column] = get_child_text(characteristics, source) if characteristics is not None else ""

            roi_count = 0
            edge_point_count = 0
            z_positions = []
            x_points = []
            y_points = []

            for roi in nodule.iter():
                if local_name(roi.tag) != "roi":
                    continue
                roi_count += 1
                z = safe_float(get_child_text(roi, "imageZposition"))
                if z is not None:
                    z_positions.append(z)

                for edge_map in roi:
                    if local_name(edge_map.tag) != "edgeMap":
                        continue
                    x = safe_float(get_child_text(edge_map, "xCoord"))
                    y = safe_float(get_child_text(edge_map, "yCoord"))
                    if x is None or y is None:
                        continue
                    x_points.append(x)
                    y_points.append(y)
                    edge_point_count += 1

            row = {
                "xml_file": str(xml_path),
                "xml_name": xml_path.name,
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "reader_id": reader_index,
                "nodule_id": get_child_text(nodule, "noduleID") or "",
                "roi_count": roi_count,
                "edge_point_count": edge_point_count,
                "z_min": min(z_positions) if z_positions else "",
                "z_max": max(z_positions) if z_positions else "",
                "x_min": min(x_points) if x_points else "",
                "x_max": max(x_points) if x_points else "",
                "y_min": min(y_points) if y_points else "",
                "y_max": max(y_points) if y_points else "",
            }
            row.update(morphology)
            rows.append(row)

    return rows


def load_or_build_xml_annotations(xml_root, force=False):
    path = TABLE_DIR / "lidc_xml_nodules_raw.csv"
    if path.exists() and not force:
        log("Loading existing XML annotation table.")
        return read_csv_rows(path)

    log("Parsing LIDC XML annotations from {}.".format(xml_root))
    xml_files = sorted(Path(xml_root).rglob("*.xml"))
    all_rows = []
    failures = []
    for idx, xml_path in enumerate(xml_files, start=1):
        if idx % 100 == 0:
            log("  XML parse: {}/{} files".format(idx, len(xml_files)))
        try:
            all_rows.extend(parse_lidc_xml(xml_path))
        except Exception as exc:
            failures.append({"xml_file": str(xml_path), "error": str(exc)})

    write_csv(path, all_rows)
    if failures:
        write_csv(TABLE_DIR / "lidc_xml_parse_failures.csv", failures)
    return all_rows


def z_tertile_for_annotation(z_center, series_row):
    if z_center is None:
        return ""

    z_min = safe_float(series_row.get("slice_z_min_mm")) if series_row else None
    z_max = safe_float(series_row.get("slice_z_max_mm")) if series_row else None

    if z_min is None or z_max is None or z_min == z_max:
        return ""

    lower = min(z_min, z_max)
    upper = max(z_min, z_max)
    rel = (z_center - lower) / (upper - lower)
    if rel < 1.0 / 3.0:
        return "inferior_third"
    if rel < 2.0 / 3.0:
        return "middle_third"
    return "superior_third"


def patient_coordinate(annotation, series_row):
    if not series_row:
        return None, None, None

    x_center_px = safe_float(annotation.get("x_center_px"))
    y_center_px = safe_float(annotation.get("y_center_px"))
    z_center = safe_float(annotation.get("z_center_mm"))
    y_spacing = safe_float(series_row.get("y_spacing_mm"))
    x_spacing = safe_float(series_row.get("x_spacing_mm"))
    ipp = [
        safe_float(series_row.get("first_slice_x0_mm")),
        safe_float(series_row.get("first_slice_y0_mm")),
        safe_float(series_row.get("first_slice_z0_mm")),
    ]
    row_dir = [
        safe_float(series_row.get("orientation_r1")),
        safe_float(series_row.get("orientation_r2")),
        safe_float(series_row.get("orientation_r3")),
    ]
    col_dir = [
        safe_float(series_row.get("orientation_c1")),
        safe_float(series_row.get("orientation_c2")),
        safe_float(series_row.get("orientation_c3")),
    ]

    needed = [x_center_px, y_center_px, z_center, y_spacing, x_spacing] + ipp + row_dir + col_dir
    if any(value is None for value in needed):
        return None, None, z_center

    base = [ipp[0], ipp[1], z_center]
    coord = [
        base[i] + y_center_px * y_spacing * row_dir[i] + x_center_px * x_spacing * col_dir[i]
        for i in range(3)
    ]
    coord[2] = z_center
    return coord[0], coord[1], coord[2]


def enrich_annotations(annotations, series_records, force=False):
    path = TABLE_DIR / "lidc_nodule_size_location_morphology.csv"
    if path.exists() and not force:
        log("Loading existing nodule size/location/morphology table.")
        return read_csv_rows(path)

    series_by_uid = {row.get("SeriesInstanceUID"): row for row in series_records if row.get("SeriesInstanceUID")}
    enriched = []

    for row in annotations:
        out = dict(row)
        series = series_by_uid.get(row.get("SeriesInstanceUID"))

        for col in NUMERIC_ANNOTATION_COLS:
            value = safe_float(out.get(col))
            out[col] = value if value is not None else ""

        x_min = safe_float(out.get("x_min"))
        x_max = safe_float(out.get("x_max"))
        y_min = safe_float(out.get("y_min"))
        y_max = safe_float(out.get("y_max"))
        z_min = safe_float(out.get("z_min"))
        z_max = safe_float(out.get("z_max"))

        rows = safe_float(series.get("Rows")) if series else None
        columns = safe_float(series.get("Columns")) if series else None
        y_spacing = safe_float(series.get("y_spacing_mm")) if series else None
        x_spacing = safe_float(series.get("x_spacing_mm")) if series else None
        slice_thickness = safe_float(series.get("SliceThickness")) if series else None

        x_center = (x_min + x_max) / 2.0 if x_min is not None and x_max is not None else None
        y_center = (y_min + y_max) / 2.0 if y_min is not None and y_max is not None else None
        z_center = (z_min + z_max) / 2.0 if z_min is not None and z_max is not None else None

        x_diameter = (x_max - x_min + 1.0) * x_spacing if None not in (x_min, x_max, x_spacing) else None
        y_diameter = (y_max - y_min + 1.0) * y_spacing if None not in (y_min, y_max, y_spacing) else None
        z_diameter = (abs(z_max - z_min) + (slice_thickness or 0.0)) if None not in (z_min, z_max) else None
        max_diameter = max([v for v in (x_diameter, y_diameter, z_diameter) if v is not None]) if any(v is not None for v in (x_diameter, y_diameter, z_diameter)) else None

        out.update({
            "patient_folder": series.get("patient_folder", "") if series else "",
            "PatientID": series.get("PatientID", "") if series else "",
            "Rows": rows if rows is not None else "",
            "Columns": columns if columns is not None else "",
            "SliceThickness": slice_thickness if slice_thickness is not None else "",
            "PixelSpacing": series.get("PixelSpacing", "") if series else "",
            "y_spacing_mm": y_spacing if y_spacing is not None else "",
            "x_spacing_mm": x_spacing if x_spacing is not None else "",
            "x_center_px": x_center if x_center is not None else "",
            "y_center_px": y_center if y_center is not None else "",
            "z_center_mm": z_center if z_center is not None else "",
            "x_min_mm": x_min * x_spacing if None not in (x_min, x_spacing) else "",
            "x_max_mm": x_max * x_spacing if None not in (x_max, x_spacing) else "",
            "y_min_mm": y_min * y_spacing if None not in (y_min, y_spacing) else "",
            "y_max_mm": y_max * y_spacing if None not in (y_max, y_spacing) else "",
            "x_center_mm_image": x_center * x_spacing if None not in (x_center, x_spacing) else "",
            "y_center_mm_image": y_center * y_spacing if None not in (y_center, y_spacing) else "",
            "x_diameter_mm": x_diameter if x_diameter is not None else "",
            "y_diameter_mm": y_diameter if y_diameter is not None else "",
            "z_diameter_mm": z_diameter if z_diameter is not None else "",
            "max_diameter_mm": max_diameter if max_diameter is not None else "",
            "size_category": nodule_size_category(max_diameter),
            "image_side": "",
            "image_ap_position": "",
            "z_location_tertile": z_tertile_for_annotation(z_center, series),
        })

        if x_center is not None and columns:
            out["image_side"] = "image_left" if x_center < columns / 2.0 else "image_right"
        if y_center is not None and rows:
            out["image_ap_position"] = "anterior_half" if y_center < rows / 2.0 else "posterior_half"

        patient_x, patient_y, patient_z = patient_coordinate(out, series)
        out["patient_x_mm"] = patient_x if patient_x is not None else ""
        out["patient_y_mm"] = patient_y if patient_y is not None else ""
        out["patient_z_mm"] = patient_z if patient_z is not None else ""
        if patient_x is not None:
            out["patient_side"] = "patient_left" if patient_x >= 0 else "patient_right"
        else:
            out["patient_side"] = ""

        enriched.append(out)

    write_csv(path, enriched)
    return enriched


def distribution_rows(counter, key_name, count_name):
    return [{key_name: key, count_name: counter[key]} for key in sorted(counter.keys(), key=lambda x: str(x))]


def build_annotation_summaries(features):
    size_counts = Counter(row.get("size_category") or "missing" for row in features)
    write_csv(
        TABLE_DIR / "lidc_nodule_size_category_distribution.csv",
        distribution_rows(size_counts, "size_category", "annotation_count"),
    )

    location_counter = Counter()
    for row in features:
        key = (
            row.get("image_side") or "missing",
            row.get("patient_side") or "missing",
            row.get("image_ap_position") or "missing",
            row.get("z_location_tertile") or "missing",
        )
        location_counter[key] += 1
    location_rows = [
        {
            "image_side": key[0],
            "patient_side": key[1],
            "image_ap_position": key[2],
            "z_location_tertile": key[3],
            "annotation_count": count,
        }
        for key, count in sorted(location_counter.items(), key=lambda item: item[1], reverse=True)
    ]
    write_csv(TABLE_DIR / "lidc_nodule_location_distribution.csv", location_rows)

    morphology_rows = []
    for col in MORPHOLOGY_COLS:
        counter = Counter()
        for row in features:
            value = row.get(col)
            number = safe_int(value)
            label = str(number) if number is not None else "missing"
            counter[label] += 1
        for label in sorted(counter.keys(), key=lambda x: (x == "missing", x)):
            morphology_rows.append({"feature": col, "score": label, "annotation_count": counter[label]})
    write_csv(TABLE_DIR / "lidc_nodule_morphology_score_distributions.csv", morphology_rows)

    malignancy_counts = Counter(str(safe_int(row.get("malignancy"))) if safe_int(row.get("malignancy")) is not None else "missing" for row in features)
    write_csv(
        TABLE_DIR / "lidc_malignancy_score_distribution.csv",
        distribution_rows(malignancy_counts, "malignancy", "annotation_count"),
    )

    reader_counts = Counter(clean_value(row.get("reader_id")) or "missing" for row in features)
    write_csv(
        TABLE_DIR / "lidc_reader_annotation_counts.csv",
        distribution_rows(reader_counts, "reader_id", "nodule_annotation_count"),
    )

    diameters = [safe_float(row.get("max_diameter_mm")) for row in features]
    diameters = [d for d in diameters if d is not None]
    summary = [{
        "annotation_count": len(features),
        "diameter_available_count": len(diameters),
        "mean_max_diameter_mm": mean(diameters) if diameters else "",
        "median_max_diameter_mm": median(diameters) if diameters else "",
        "p25_max_diameter_mm": percentile(diameters, 0.25) if diameters else "",
        "p75_max_diameter_mm": percentile(diameters, 0.75) if diameters else "",
        "min_max_diameter_mm": min(diameters) if diameters else "",
        "max_max_diameter_mm": max(diameters) if diameters else "",
    }]
    write_csv(TABLE_DIR / "lidc_nodule_size_summary.csv", summary)


def build_match_summary(annotations, series_records):
    xml_series = set(row.get("SeriesInstanceUID") for row in annotations if row.get("SeriesInstanceUID"))
    dicom_series = set(row.get("SeriesInstanceUID") for row in series_records if row.get("SeriesInstanceUID"))
    matched = xml_series.intersection(dicom_series)
    rows = [{
        "xml_series_count": len(xml_series),
        "main_dicom_ct_series_count": len(dicom_series),
        "matched_series_count": len(matched),
        "unmatched_xml_series_count": len(xml_series - dicom_series),
        "match_fraction": len(matched) / float(len(xml_series)) if xml_series else "",
    }]
    write_csv(TABLE_DIR / "lidc_xml_dicom_match_summary.csv", rows)


def build_qc_report(demographics, features):
    total_patients = len(demographics)
    age_available = sum(1 for row in demographics if safe_float(row.get("PatientAge_years")) is not None)
    sex_available = sum(1 for row in demographics if clean_patient_sex(row.get("PatientSex")))

    total_annotations = len(features)
    missing_spacing = sum(1 for row in features if safe_float(row.get("x_spacing_mm")) is None or safe_float(row.get("y_spacing_mm")) is None)
    bad_diameter = sum(1 for row in features if (safe_float(row.get("max_diameter_mm")) or 0) <= 0)
    very_large = sum(1 for row in features if (safe_float(row.get("max_diameter_mm")) or 0) > 60)
    morphology_available = sum(1 for row in features if safe_float(row.get("malignancy")) is not None)

    rows = [
        {
            "check": "patient_age_available_fraction",
            "value": age_available / float(total_patients) if total_patients else "",
            "risk_level": "high" if total_patients and age_available / float(total_patients) < 0.5 else "low",
            "recommendation": "Handle age missingness explicitly before age-stratified analysis.",
        },
        {
            "check": "patient_sex_available_fraction",
            "value": sex_available / float(total_patients) if total_patients else "",
            "risk_level": "high" if total_patients and sex_available / float(total_patients) < 0.5 else "low",
            "recommendation": "Avoid sex-stratified complete-case analysis without reporting missingness.",
        },
        {
            "check": "annotations_missing_spacing_fraction",
            "value": missing_spacing / float(total_annotations) if total_annotations else "",
            "risk_level": "medium" if missing_spacing else "low",
            "recommendation": "Recover CT geometry or exclude annotations without spacing from size/location summaries.",
        },
        {
            "check": "diameter_le_zero_count",
            "value": bad_diameter,
            "risk_level": "medium" if bad_diameter else "low",
            "recommendation": "Review or exclude annotations with non-positive estimated diameter.",
        },
        {
            "check": "diameter_gt_60mm_count",
            "value": very_large,
            "risk_level": "medium" if very_large else "low",
            "recommendation": "Review very large lesions before treating them as pulmonary nodules.",
        },
        {
            "check": "morphology_available_fraction",
            "value": morphology_available / float(total_annotations) if total_annotations else "",
            "risk_level": "medium",
            "recommendation": "Morphology scores are mainly available for nodules with a characteristics block; analyse missing morphology separately.",
        },
    ]
    write_csv(TABLE_DIR / "lidc_processed_quality_control_report.csv", rows)


def maybe_plot(demographics, features):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log("matplotlib is not installed; skipping PNG figures.")
        return

    ages = [safe_float(row.get("PatientAge_years")) for row in demographics]
    ages = [age for age in ages if age is not None]
    if ages:
        plt.figure(figsize=(7, 5))
        plt.hist(ages, bins=20)
        plt.xlabel("Patient age (years)")
        plt.ylabel("Number of patients")
        plt.title("LIDC Patient Age Distribution")
        plt.tight_layout()
        plt.savefig(str(FIGURE_DIR / "lidc_patient_age_distribution.png"), dpi=300)
        plt.close()

    sex_counts = Counter(clean_patient_sex(row.get("PatientSex")) or "missing" for row in demographics)
    if sex_counts:
        labels = list(sorted(sex_counts.keys()))
        plt.figure(figsize=(6, 5))
        plt.bar(labels, [sex_counts[label] for label in labels])
        plt.xlabel("Patient sex")
        plt.ylabel("Number of patients")
        plt.title("LIDC Patient Sex Distribution")
        plt.tight_layout()
        plt.savefig(str(FIGURE_DIR / "lidc_patient_sex_distribution.png"), dpi=300)
        plt.close()

    diameters = [safe_float(row.get("max_diameter_mm")) for row in features]
    diameters = [min(d, 60.0) for d in diameters if d is not None]
    if diameters:
        plt.figure(figsize=(8, 5))
        plt.hist(diameters, bins=40)
        plt.xlabel("Approximate maximum nodule diameter (mm, clipped at 60)")
        plt.ylabel("Reader annotations")
        plt.title("LIDC Nodule Size Distribution")
        plt.tight_layout()
        plt.savefig(str(FIGURE_DIR / "lidc_nodule_size_distribution.png"), dpi=300)
        plt.close()

    for column, filename, title in [
        ("image_side", "lidc_nodule_image_side_distribution.png", "Nodule Image Side Distribution"),
        ("patient_side", "lidc_nodule_patient_side_distribution.png", "Nodule Patient Side Distribution"),
        ("z_location_tertile", "lidc_nodule_z_location_distribution.png", "Nodule Superior/Inferior Distribution"),
    ]:
        counts = Counter(row.get(column) or "missing" for row in features)
        labels = list(sorted(counts.keys()))
        plt.figure(figsize=(7, 5))
        plt.bar(labels, [counts[label] for label in labels])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Reader annotations")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(str(FIGURE_DIR / filename), dpi=300)
        plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, col in zip(axes.ravel(), MORPHOLOGY_COLS):
        counts = Counter(str(safe_int(row.get(col))) for row in features if safe_int(row.get(col)) is not None)
        labels = sorted(counts.keys(), key=lambda x: int(x))
        ax.bar(labels, [counts[label] for label in labels])
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("Score")
        ax.set_ylabel("Reader annotations")
    plt.tight_layout()
    plt.savefig(str(FIGURE_DIR / "lidc_nodule_morphology_distributions.png"), dpi=300)
    plt.close()


def print_summary(demographics, features):
    ages = [safe_float(row.get("PatientAge_years")) for row in demographics]
    ages = [age for age in ages if age is not None]
    sex_counts = Counter(clean_patient_sex(row.get("PatientSex")) or "missing" for row in demographics)
    diameters = [safe_float(row.get("max_diameter_mm")) for row in features]
    diameters = [d for d in diameters if d is not None]
    size_counts = Counter(row.get("size_category") or "missing" for row in features)

    log("")
    log("LIDC-IDRI exploration complete")
    log("  Patients: {}".format(len(demographics)))
    log("  Patient age available: {}/{}".format(len(ages), len(demographics)))
    if ages:
        log("  Age median (IQR): {:.1f} ({:.1f}-{:.1f}) years".format(
            median(ages), percentile(ages, 0.25), percentile(ages, 0.75)
        ))
    log("  Patient sex distribution: {}".format(dict(sex_counts)))
    log("  Reader nodule annotations: {}".format(len(features)))
    if diameters:
        log("  Nodule max diameter median (IQR): {:.2f} ({:.2f}-{:.2f}) mm".format(
            median(diameters), percentile(diameters, 0.25), percentile(diameters, 0.75)
        ))
    log("  Size categories: {}".format(dict(size_counts)))
    log("  Tables: {}".format(TABLE_DIR))
    log("  Figures: {}".format(FIGURE_DIR))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Explore local raw LIDC-IDRI DICOM/XML data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate derived tables even if outputs already exist.",
    )
    parser.add_argument(
        "--skip-slice-ranges",
        action="store_true",
        help="Do not scan every CT slice for per-series z ranges. Faster, but z tertile locations may be missing.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_dirs()

    manifest_root = discover_manifest_root()
    xml_root = discover_xml_root(manifest_root)

    log("Project root: {}".format(PROJECT_ROOT))
    log("LIDC root: {}".format(LIDC_ROOT))
    log("Manifest root: {}".format(manifest_root))
    log("XML root: {}".format(xml_root))

    series_records, main_series, _metadata = load_or_build_series_tables(
        manifest_root,
        force=args.force,
        scan_slice_ranges=not args.skip_slice_ranges,
    )
    demographics = build_demographics(main_series, force=args.force)
    annotations = load_or_build_xml_annotations(xml_root, force=args.force)
    features = enrich_annotations(annotations, series_records, force=args.force)

    build_match_summary(annotations, series_records)
    build_annotation_summaries(features)
    build_qc_report(demographics, features)
    maybe_plot(demographics, features)
    print_summary(demographics, features)


if __name__ == "__main__":
    main()
