"""Merge independently evaluated LUNA16 complete-CT scan partitions."""

from __future__ import print_function

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from LUNA_evaluate_full_ct_proposals import plot_operating_points
from luna16_external_utils import write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_reports"
    / "luna16_full_ct_detection_subset9"
)


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge LUNA16 complete-CT evaluation partitions."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected-threshold", type=float, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summaries = []
    scored_proposals = []
    scan_rows = []
    for input_dir in args.inputs:
        with (input_dir / "luna16_full_ct_proposal_summary.json").open(
            "r", encoding="utf-8"
        ) as handle:
            summaries.append(json.load(handle))
        scored_proposals.extend(
            read_csv_rows(input_dir / "luna16_full_ct_scored_proposals.csv")
        )
        scan_rows.extend(
            read_csv_rows(input_dir / "luna16_full_ct_scan_summary.csv")
        )

    scan_count = sum(int(summary["scan_count"]) for summary in summaries)
    annotation_count = sum(int(summary["annotation_count"]) for summary in summaries)
    proposal_matched = sum(
        int(summary["proposal_matched_annotation_count"]) for summary in summaries
    )
    thresholds = sorted({
        float(point["nodule_threshold"])
        for summary in summaries
        for point in summary["operating_points"]
    })
    operating_points = []
    for threshold in thresholds:
        matched_total = 0.0
        detection_total = 0.0
        false_positive_total = 0.0
        for summary in summaries:
            point = next(
                item
                for item in summary["operating_points"]
                if float(item["nodule_threshold"]) == threshold
            )
            part_scans = int(summary["scan_count"])
            part_annotations = int(summary["annotation_count"])
            matched_total += float(point["sensitivity"]) * part_annotations
            detection_total += (
                float(point["average_detections_per_scan"]) * part_scans
            )
            false_positive_total += (
                float(point["average_false_positives_per_scan"]) * part_scans
            )
        operating_points.append({
            "nodule_threshold": threshold,
            "sensitivity": (
                matched_total / float(annotation_count)
                if annotation_count else None
            ),
            "average_detections_per_scan": (
                detection_total / float(scan_count) if scan_count else None
            ),
            "average_false_positives_per_scan": (
                false_positive_total / float(scan_count) if scan_count else None
            ),
        })

    template = summaries[0]
    selected_threshold = (
        float(args.selected_threshold)
        if args.selected_threshold is not None
        else float(template["nodule_threshold"])
    )
    proposals_by_series = defaultdict(list)
    for row in scored_proposals:
        proposals_by_series[row["seriesuid"]].append(row)
    candidate_matches = []
    max_detections = int(template["max_detections"])
    for rows in proposals_by_series.values():
        retained = [
            row for row in rows
            if float(row["nodule_probability"]) >= selected_threshold
        ]
        retained.sort(
            key=lambda row: float(row["detection_score"]),
            reverse=True,
        )
        candidate_matches.extend(retained[:max_detections])
    detection_count = len(candidate_matches)
    false_positive_count = sum(
        str(row["matched_annotation"]).lower() != "true"
        for row in candidate_matches
    )
    matched_annotations = {
        (row["seriesuid"], int(row["nearest_annotation_index"]))
        for row in candidate_matches
        if str(row["matched_annotation"]).lower() == "true"
        and row.get("nearest_annotation_index", "") != ""
    }
    detection_matched = len(matched_annotations)
    merged = {
        "scan_count": scan_count,
        "annotation_count": annotation_count,
        "proposal_matched_annotation_count": proposal_matched,
        "proposal_sensitivity": (
            proposal_matched / float(annotation_count) if annotation_count else None
        ),
        "matched_annotation_count": detection_matched,
        "detection_sensitivity": (
            detection_matched / float(annotation_count) if annotation_count else None
        ),
        "detection_count": detection_count,
        "average_detections_per_scan": (
            detection_count / float(scan_count) if scan_count else None
        ),
        "average_false_positive_detections_per_scan": (
            false_positive_count / float(scan_count) if scan_count else None
        ),
        "subsets": template["subsets"],
        "response_percentile": template["response_percentile"],
        "max_proposals": template["max_proposals"],
        "min_distance_mm": template.get("min_distance_mm"),
        "max_detections": template["max_detections"],
        "nodule_threshold": selected_threshold,
        "candidate_checkpoint": template["candidate_checkpoint"],
        "candidate_model_config": template["candidate_model_config"],
        "operating_points": operating_points,
        "partition_count": len(summaries),
        "partition_directories": [str(path) for path in args.inputs],
        "interpretation": template["interpretation"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "luna16_full_ct_scored_proposals.csv",
        scored_proposals,
    )
    write_csv(
        args.output_dir / "luna16_full_ct_candidate_matches.csv",
        candidate_matches,
    )
    write_csv(
        args.output_dir / "luna16_full_ct_scan_summary.csv",
        scan_rows,
    )
    with (args.output_dir / "luna16_full_ct_proposal_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
    plot_operating_points(
        operating_points,
        args.output_dir / "luna16_full_ct_detection_operating_points.png",
    )
    print(json.dumps(merged, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
