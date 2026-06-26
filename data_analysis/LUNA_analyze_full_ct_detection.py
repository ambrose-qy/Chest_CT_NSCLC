"""Analyse annotation-level misses and false positives in complete-CT detection."""

from __future__ import print_function

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from luna16_external_utils import read_csv_rows, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = Path("E:/LUNA/annotations.csv")
DEFAULT_DETECTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_reports"
    / "luna16_full_ct_detection_subset9"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyse complete-CT detection outcomes by LUNA16 annotation."
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--detection-dir", type=Path, default=DEFAULT_DETECTION_DIR)
    parser.add_argument("--nodule-threshold", type=float, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    return parser.parse_args(argv)


def diameter_group(diameter_mm):
    if diameter_mm < 6.0:
        return "small_3_to_6_mm"
    if diameter_mm < 10.0:
        return "medium_6_to_10_mm"
    return "large_10_mm_or_more"


def nearest_candidate(annotation_xyz, candidates):
    if not candidates:
        return None, None
    center = np.asarray(annotation_xyz, dtype=np.float64)
    distances = [
        float(np.linalg.norm(
            center
            - np.asarray(
                [row["world_x"], row["world_y"], row["world_z"]],
                dtype=np.float64,
            )
        ))
        for row in candidates
    ]
    index = int(np.argmin(distances))
    return candidates[index], distances[index]


def main(argv=None):
    args = parse_args(argv)
    with (args.detection_dir / "luna16_full_ct_proposal_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    threshold = (
        float(args.nodule_threshold)
        if args.nodule_threshold is not None
        else float(summary["nodule_threshold"])
    )
    max_detections = (
        int(args.max_detections)
        if args.max_detections is not None
        else int(summary["max_detections"])
    )

    proposals_by_series = defaultdict(list)
    for row in read_csv_rows(
        args.detection_dir / "luna16_full_ct_scored_proposals.csv"
    ):
        converted = dict(row)
        for key in (
            "world_x",
            "world_y",
            "world_z",
            "proposal_score",
            "nodule_probability",
            "detection_score",
        ):
            converted[key] = float(converted[key])
        converted["matched_annotation"] = (
            str(converted.get("matched_annotation", "")).lower() == "true"
        )
        if converted.get("nearest_annotation_index", "") != "":
            converted["nearest_annotation_index"] = int(
                converted["nearest_annotation_index"]
            )
        proposals_by_series[row["seriesuid"]].append(converted)

    annotations_by_series = defaultdict(list)
    for row in read_csv_rows(args.annotations):
        annotations_by_series[row["seriesuid"]].append({
            "world_xyz": (
                float(row["coordX"]),
                float(row["coordY"]),
                float(row["coordZ"]),
            ),
            "diameter_mm": float(row["diameter_mm"]),
        })

    scan_uids = {
        row["seriesuid"]
        for row in read_csv_rows(
            args.detection_dir / "luna16_full_ct_scan_summary.csv"
        )
    }
    outcomes = []
    for seriesuid in sorted(scan_uids):
        proposals = proposals_by_series.get(seriesuid, [])
        retained = [
            row
            for row in proposals
            if row["nodule_probability"] >= threshold
        ]
        retained.sort(key=lambda row: row["detection_score"], reverse=True)
        retained = retained[:max_detections]
        proposal_matched_indices = {
            row["nearest_annotation_index"]
            for row in proposals
            if row["matched_annotation"]
            and row.get("nearest_annotation_index", "") != ""
        }
        detection_matched_indices = {
            row["nearest_annotation_index"]
            for row in retained
            if row["matched_annotation"]
            and row.get("nearest_annotation_index", "") != ""
        }
        for annotation_index, annotation in enumerate(
            annotations_by_series.get(seriesuid, [])
        ):
            diameter = annotation["diameter_mm"]
            match_radius = max(diameter / 2.0, 3.0)
            nearest_proposal, proposal_distance = nearest_candidate(
                annotation["world_xyz"],
                proposals,
            )
            nearest_detection, detection_distance = nearest_candidate(
                annotation["world_xyz"],
                retained,
            )
            proposal_matched = annotation_index in proposal_matched_indices
            detection_matched = annotation_index in detection_matched_indices
            outcomes.append({
                "seriesuid": seriesuid,
                "annotation_index": annotation_index,
                "coord_x": annotation["world_xyz"][0],
                "coord_y": annotation["world_xyz"][1],
                "coord_z": annotation["world_xyz"][2],
                "diameter_mm": diameter,
                "diameter_group": diameter_group(diameter),
                "match_radius_mm": match_radius,
                "proposal_matched": proposal_matched,
                "nearest_proposal_distance_mm": proposal_distance,
                "nearest_proposal_source": (
                    nearest_proposal["proposal_source"]
                    if nearest_proposal is not None else ""
                ),
                "detection_matched": detection_matched,
                "nearest_detection_distance_mm": detection_distance,
                "nearest_detection_probability": (
                    nearest_detection["nodule_probability"]
                    if nearest_detection is not None else ""
                ),
                "failure_stage": (
                    ""
                    if detection_matched
                    else "candidate_filter"
                    if proposal_matched
                    else "proposal_generation"
                ),
            })

    group_rows = []
    grouped = defaultdict(list)
    for row in outcomes:
        grouped[row["diameter_group"]].append(row)
    for group, rows in sorted(grouped.items()):
        group_rows.append({
            "diameter_group": group,
            "annotation_count": len(rows),
            "proposal_matched_count": sum(row["proposal_matched"] for row in rows),
            "proposal_sensitivity": (
                sum(row["proposal_matched"] for row in rows) / float(len(rows))
            ),
            "detection_matched_count": sum(row["detection_matched"] for row in rows),
            "detection_sensitivity": (
                sum(row["detection_matched"] for row in rows) / float(len(rows))
            ),
        })

    failures = Counter(
        row["failure_stage"] for row in outcomes if row["failure_stage"]
    )
    analysis = {
        "annotation_count": len(outcomes),
        "nodule_threshold": threshold,
        "max_detections": max_detections,
        "proposal_miss_count": failures.get("proposal_generation", 0),
        "candidate_filter_miss_count": failures.get("candidate_filter", 0),
        "detected_count": sum(row["detection_matched"] for row in outcomes),
        "diameter_groups": group_rows,
    }
    write_csv(
        args.detection_dir / "luna16_full_ct_annotation_outcomes.csv",
        outcomes,
    )
    write_csv(
        args.detection_dir / "luna16_full_ct_diameter_subgroups.csv",
        group_rows,
    )
    with (args.detection_dir / "luna16_full_ct_failure_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(analysis, handle, indent=2, sort_keys=True)
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
