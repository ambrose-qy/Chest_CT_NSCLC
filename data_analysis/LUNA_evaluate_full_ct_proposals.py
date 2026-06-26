"""Evaluate complete-CT candidate proposal recall on LUNA16 scans."""

from __future__ import print_function

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from lidc_full_ct_utils import (
    match_candidates_to_annotations,
    propose_nodule_candidates,
    read_ct_image,
    resample_image,
    sitk_to_numpy,
)
from luna16_candidate_detector import (
    candidate_probabilities_from_volume,
    load_candidate_detector,
)
from luna16_external_utils import load_luna_scan_index, read_csv_rows, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = Path("E:/LUNA/annotations.csv")
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_reports"
    / "luna16_full_ct_proposals"
)
DEFAULT_CANDIDATE_CHECKPOINT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "luna16_candidate_detector"
    / "best_candidate_detector.pt"
)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "luna16_detection.yaml"


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_evaluation_defaults(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("full_ct_evaluation", {})


def annotation_index(path):
    grouped = defaultdict(list)
    for row in read_csv_rows(path):
        grouped[row["seriesuid"]].append({
            "world_xyz": [
                float(row["coordX"]),
                float(row["coordY"]),
                float(row["coordZ"]),
            ],
            "diameter_mm": float(row["diameter_mm"]),
        })
    return grouped


def plot_operating_points(operating_points, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sorted(
        operating_points,
        key=lambda row: row["average_false_positives_per_scan"],
    )
    fig, axis = plt.subplots(figsize=(6.5, 5))
    axis.plot(
        [row["average_false_positives_per_scan"] for row in points],
        [row["sensitivity"] for row in points],
        marker="o",
        linewidth=2,
    )
    for row in points:
        axis.annotate(
            "{:.2f}".format(row["nodule_threshold"]),
            (
                row["average_false_positives_per_scan"],
                row["sensitivity"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Average false positives per scan")
    axis.set_ylabel("Detection sensitivity")
    axis.set_title("LUNA16 complete-CT detection operating points")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=240)
    plt.close(fig)


def parse_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known, _ = pre_parser.parse_known_args(argv)
    defaults = load_evaluation_defaults(known.config)
    parser = argparse.ArgumentParser(
        description="Evaluate LoG full-CT candidate proposals on LUNA16.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(defaults.get("annotations", DEFAULT_ANNOTATIONS)),
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=defaults.get("subsets", ["subset9"]),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--scan-start", type=int, default=0)
    parser.add_argument("--scan-end", type=int, default=None)
    parser.add_argument(
        "--response-percentile",
        type=float,
        default=float(defaults.get("response_percentile", 99.7)),
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=int(defaults.get("max_proposals", 1000)),
    )
    parser.add_argument(
        "--min-distance-mm",
        type=float,
        default=float(defaults.get("min_distance_mm", 3.0)),
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=int(defaults.get("max_detections", 64)),
    )
    parser.add_argument(
        "--nodule-threshold",
        type=float,
        default=float(defaults.get("nodule_threshold", 0.8)),
    )
    parser.add_argument(
        "--operating-thresholds",
        nargs="+",
        type=float,
        default=defaults.get(
            "operating_thresholds",
            [0.5, 0.7, 0.8, 0.85, 0.9, 0.95],
        ),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=resolve_project_path(
            defaults.get("candidate_checkpoint", DEFAULT_CANDIDATE_CHECKPOINT)
        ),
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=int(defaults.get("candidate_batch_size", 64)),
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    scans = load_luna_scan_index()
    annotations = annotation_index(args.annotations)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(int(args.torch_threads), 1))
    candidate_model, candidate_config = load_candidate_detector(
        args.candidate_checkpoint,
        device,
    )
    selected = [
        scan for scan in scans.values()
        if scan.get("subset") in set(args.subsets)
    ]
    selected.sort(key=lambda row: row["seriesuid"])
    selected = selected[int(args.scan_start):args.scan_end]
    if args.max_scans is not None:
        selected = selected[: int(args.max_scans)]

    candidate_rows = []
    proposal_rows_all = []
    scan_rows = []
    total_annotations = 0
    total_proposal_matched = 0
    total_matched = 0
    operating_totals = {
        float(threshold): {
            "matched_annotations": 0,
            "detections": 0,
            "false_positives": 0,
        }
        for threshold in args.operating_thresholds
    }
    for scan_number, scan in enumerate(selected, start=1):
        seriesuid = scan["seriesuid"]
        scan_annotations = annotations.get(seriesuid, [])
        print(
            "[{}/{}] {} annotations={}".format(
                scan_number, len(selected), seriesuid, len(scan_annotations)
            ),
            flush=True,
        )
        image, _ = read_ct_image(Path(scan["mhd_path"]))
        image = resample_image(image, spacing_xyz=(1.0, 1.0, 1.0))
        volume_hu = sitk_to_numpy(image)
        candidates, proposal_meta = propose_nodule_candidates(
            volume_hu,
            response_percentile=args.response_percentile,
            min_distance_mm=args.min_distance_mm,
            max_candidates=args.max_proposals,
        )
        proposal_rows, proposal_matched_indices = match_candidates_to_annotations(
            candidates,
            scan_annotations,
            image,
        )
        probabilities = candidate_probabilities_from_volume(
            candidate_model,
            volume_hu,
            candidates,
            device,
            batch_size=args.candidate_batch_size,
        )
        for candidate, probability in zip(candidates, probabilities):
            candidate["nodule_probability"] = float(probability)
            candidate["detection_score"] = float(
                0.85 * probability + 0.15 * float(candidate["proposal_score"])
            )
        for row, candidate in zip(proposal_rows, candidates):
            row.update({
                "seriesuid": seriesuid,
                "subset": scan.get("subset", ""),
                "nodule_probability": candidate["nodule_probability"],
                "detection_score": candidate["detection_score"],
            })
        proposal_rows_all.extend(proposal_rows)
        for threshold, totals in operating_totals.items():
            operating_candidates = [
                candidate
                for candidate in candidates
                if candidate["nodule_probability"] >= threshold
            ]
            operating_candidates.sort(
                key=lambda row: row["detection_score"],
                reverse=True,
            )
            operating_candidates = operating_candidates[: int(args.max_detections)]
            operating_rows, operating_matches = match_candidates_to_annotations(
                operating_candidates,
                scan_annotations,
                image,
            )
            totals["matched_annotations"] += len(operating_matches)
            totals["detections"] += len(operating_rows)
            totals["false_positives"] += sum(
                not bool(row["matched_annotation"]) for row in operating_rows
            )
        retained = [
            candidate
            for candidate in candidates
            if candidate["nodule_probability"] >= float(args.nodule_threshold)
        ]
        retained.sort(key=lambda row: row["detection_score"], reverse=True)
        retained = retained[: int(args.max_detections)]
        matched_rows, matched_indices = match_candidates_to_annotations(
            retained,
            scan_annotations,
            image,
        )
        for row in matched_rows:
            row.update({
                "seriesuid": seriesuid,
                "subset": scan.get("subset", ""),
            })
        candidate_rows.extend(matched_rows)
        total_annotations += len(scan_annotations)
        total_proposal_matched += len(proposal_matched_indices)
        total_matched += len(matched_indices)
        scan_rows.append({
            "seriesuid": seriesuid,
            "subset": scan.get("subset", ""),
            "annotation_count": len(scan_annotations),
            "proposal_matched_annotation_count": len(proposal_matched_indices),
            "proposal_sensitivity": (
                len(proposal_matched_indices) / float(len(scan_annotations))
                if scan_annotations else ""
            ),
            "matched_annotation_count": len(matched_indices),
            "detection_sensitivity": (
                len(matched_indices) / float(len(scan_annotations))
                if scan_annotations else ""
            ),
            "proposal_count": len(proposal_rows),
            "detection_count": len(retained),
            "false_positive_detection_count": sum(
                not bool(row["matched_annotation"]) for row in matched_rows
            ),
            "lung_fraction": proposal_meta.get("lung_fraction", ""),
            "proposal_threshold": proposal_meta.get("proposal_threshold", ""),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "luna16_full_ct_scored_proposals.csv",
        proposal_rows_all,
    )
    write_csv(args.output_dir / "luna16_full_ct_candidate_matches.csv", candidate_rows)
    write_csv(args.output_dir / "luna16_full_ct_scan_summary.csv", scan_rows)
    operating_points = [
        {
            "nodule_threshold": threshold,
            "sensitivity": (
                totals["matched_annotations"] / float(total_annotations)
                if total_annotations else None
            ),
            "average_detections_per_scan": (
                totals["detections"] / float(len(selected))
                if selected else None
            ),
            "average_false_positives_per_scan": (
                totals["false_positives"] / float(len(selected))
                if selected else None
            ),
        }
        for threshold, totals in sorted(operating_totals.items())
    ]
    summary = {
        "scan_count": len(selected),
        "annotation_count": total_annotations,
        "proposal_matched_annotation_count": total_proposal_matched,
        "proposal_sensitivity": (
            total_proposal_matched / float(total_annotations)
            if total_annotations else None
        ),
        "matched_annotation_count": total_matched,
        "detection_sensitivity": (
            total_matched / float(total_annotations) if total_annotations else None
        ),
        "detection_count": len(candidate_rows),
        "average_detections_per_scan": (
            len(candidate_rows) / float(len(selected)) if selected else None
        ),
        "average_false_positive_detections_per_scan": (
            sum(not bool(row["matched_annotation"]) for row in candidate_rows)
            / float(len(selected))
            if selected else None
        ),
        "subsets": args.subsets,
        "scan_start": args.scan_start,
        "scan_end": args.scan_end,
        "response_percentile": args.response_percentile,
        "max_proposals": args.max_proposals,
        "min_distance_mm": args.min_distance_mm,
        "max_detections": args.max_detections,
        "nodule_threshold": args.nodule_threshold,
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "candidate_model_config": candidate_config,
        "operating_points": operating_points,
        "interpretation": (
            "Proposal sensitivity measures the high-recall image-processing stage. "
            "Detection sensitivity and false positives per scan are measured after the "
            "LUNA16-trained 3D CBAM false-positive reduction model."
        ),
    }
    with (args.output_dir / "luna16_full_ct_proposal_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    plot_operating_points(
        operating_points,
        args.output_dir / "luna16_full_ct_detection_operating_points.png",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
