"""Build label-matched LUNA16 ROI cubes for fair LIDC test-domain validation.

LUNA16 is derived from LIDC-IDRI and does not publish malignancy labels.
This script matches LUNA annotations to the local LIDC radiologist-consensus
ROI manifest, transfers only the existing LIDC labels/splits, and re-extracts
the image cube from the LUNA MetaImage source. The default export includes
only nodules assigned to the patient-level LIDC test split.
"""

from __future__ import print_function

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from luna16_external_utils import (
    DEFAULT_LUNA_PROCESSED_DIRS,
    clean_value,
    extract_world_patch,
    load_lidc_label_index,
    load_luna_annotation_coordinates,
    load_luna_scan_index,
    match_luna_to_lidc,
    normalise_hu,
    parse_shape,
    read_luna_scan,
    safe_float,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "luna16_labeled_external_rois"


def selected_for_split(row, split):
    if split == "all":
        return True
    return (
        clean_value(row.get("binary_split")) == split
        or clean_value(row.get("multiclass_split")) == split
    )


def safe_name(value):
    return "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in str(value)
    )


def export_rois(matches, scan_index, output_dir, shape, split, force=False, max_rois=None):
    output_dir = Path(output_dir)
    volume_dir = output_dir / "volumes"
    volume_dir.mkdir(parents=True, exist_ok=True)
    selected = [row for row in matches if selected_for_split(row, split)]
    if max_rois is not None:
        selected = selected[: int(max_rois)]

    by_series = defaultdict(list)
    for row in selected:
        by_series[clean_value(row.get("seriesuid"))].append(row)

    manifest_rows = []
    failures = []
    for series_number, (seriesuid, rows) in enumerate(sorted(by_series.items()), start=1):
        scan = scan_index.get(seriesuid)
        if not scan:
            for row in rows:
                failures.append({
                    "seriesuid": seriesuid,
                    "matched_roi_id": row.get("matched_roi_id", ""),
                    "reason": "missing_luna_mhd_path",
                })
            continue

        print(
            "[{}/{}] {} ({} ROI)".format(
                series_number,
                len(by_series),
                seriesuid,
                len(rows),
            ),
            flush=True,
        )
        try:
            volume_hu, info = read_luna_scan(scan["mhd_path"])
        except Exception as exc:
            for row in rows:
                failures.append({
                    "seriesuid": seriesuid,
                    "matched_roi_id": row.get("matched_roi_id", ""),
                    "reason": "scan_read_failed",
                    "detail": str(exc),
                })
            continue

        for row in rows:
            roi_id = clean_value(row.get("matched_roi_id"))
            output_path = volume_dir / "{}__luna.npz".format(safe_name(roi_id))
            try:
                if not output_path.exists() or force:
                    center_world = [
                        safe_float(row.get("coordX")),
                        safe_float(row.get("coordY")),
                        safe_float(row.get("coordZ")),
                    ]
                    if any(value is None for value in center_world):
                        raise ValueError("Missing LUNA world coordinate.")
                    patch_hu = extract_world_patch(
                        volume_hu,
                        info,
                        center_world_xyz=center_world,
                        output_shape=shape,
                    )
                    patch = normalise_hu(patch_hu)
                    np.savez_compressed(
                        str(output_path),
                        volume=patch.astype(np.float32),
                        roi_id=np.asarray([roi_id]),
                        seriesuid=np.asarray([seriesuid]),
                        binary_label_id=np.asarray([
                            int(float(row["binary_label_id"]))
                            if clean_value(row.get("binary_label_id"))
                            else -1
                        ], dtype=np.int16),
                        multiclass_risk_label_id=np.asarray([
                            int(float(row["multiclass_risk_label_id"]))
                            if clean_value(row.get("multiclass_risk_label_id"))
                            else -1
                        ], dtype=np.int16),
                    )

                out = dict(row)
                out.update({
                    "subset": scan.get("subset", ""),
                    "mhd_path": scan.get("mhd_path", ""),
                    "volume_path": str(output_path),
                    "volume_array_key": "volume",
                    "crop_shape_zyx": "x".join(str(value) for value in shape),
                    "crop_spacing_xyz_mm": "1x1x1",
                    "hu_window": "-1000,400",
                    "normalization": "min_max_0_1",
                    "external_domain": "LUNA16 MetaImage preprocessing",
                    "external_independence_note": (
                        "Same underlying LIDC-IDRI scans; evaluates preprocessing/domain "
                        "robustness, not an independent clinical population."
                    ),
                })
                manifest_rows.append(out)
            except Exception as exc:
                failures.append({
                    "seriesuid": seriesuid,
                    "matched_roi_id": roi_id,
                    "reason": "roi_export_failed",
                    "detail": str(exc),
                })
    return manifest_rows, failures


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build LUNA16 image-domain ROIs matched to LIDC test labels."
    )
    parser.add_argument(
        "--luna-processed-dirs",
        type=Path,
        nargs="+",
        default=DEFAULT_LUNA_PROCESSED_DIRS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shape", default="64,64,64")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-match-distance-mm", type=float, default=5.0)
    parser.add_argument("--max-rois", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    shape = parse_shape(args.shape)
    scan_index = load_luna_scan_index(args.luna_processed_dirs)
    luna_rows = load_luna_annotation_coordinates(args.luna_processed_dirs)
    lidc_index = load_lidc_label_index()
    matches, unmatched = match_luna_to_lidc(
        luna_rows,
        lidc_index,
        max_distance_mm=args.max_match_distance_mm,
    )
    manifest_rows, failures = export_rois(
        matches,
        scan_index,
        args.output_dir,
        shape,
        split=args.split,
        force=args.force,
        max_rois=args.max_rois,
    )

    manifest_path = args.output_dir / "luna16_labeled_external_roi_manifest.csv"
    match_path = args.output_dir / "luna16_lidc_coordinate_matches.csv"
    unmatched_path = args.output_dir / "luna16_lidc_unmatched.csv"
    failure_path = args.output_dir / "luna16_labeled_external_roi_failures.csv"
    write_csv(manifest_path, manifest_rows)
    write_csv(match_path, matches)
    write_csv(unmatched_path, unmatched)
    write_csv(failure_path, failures)

    distances = [
        float(row["match_distance_mm"])
        for row in matches
        if clean_value(row.get("match_distance_mm"))
    ]
    summary = {
        "luna_annotation_count": len(luna_rows),
        "matched_annotation_count": len(matches),
        "unmatched_annotation_count": len(unmatched),
        "exported_roi_count": len(manifest_rows),
        "export_failure_count": len(failures),
        "split_filter": args.split,
        "max_match_distance_mm": args.max_match_distance_mm,
        "median_match_distance_mm": float(np.median(distances)) if distances else None,
        "p95_match_distance_mm": float(np.percentile(distances, 95)) if distances else None,
        "binary_label_counts": dict(Counter(
            clean_value(row.get("binary_label")) or "not_binary_labeled"
            for row in manifest_rows
        )),
        "multiclass_label_counts": dict(Counter(
            clean_value(row.get("multiclass_risk_label")) or "missing"
            for row in manifest_rows
        )),
        "manifest": str(manifest_path),
        "match_table": str(match_path),
        "external_independence_note": (
            "LUNA16 is derived from LIDC-IDRI. This is a cross-preprocessing-domain "
            "test on the patient-level LIDC test split, not a new-patient clinical cohort."
        ),
    }
    with (args.output_dir / "luna16_labeled_external_roi_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
