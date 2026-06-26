"""Complete CT series nodule proposal, classification, and risk stratification."""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from LIDC_inference import load_model, normalization_stats, read_json
from lidc_full_ct_utils import (
    fixed_crop_or_pad,
    normalise_hu,
    numpy_index_to_world,
    propose_nodule_candidates,
    read_ct_image,
    resample_image,
    sitk_to_numpy,
)
from lidc_lightning_data import BINARY_LABELS, MULTICLASS_LABELS, normalise_tensor
from luna16_candidate_detector import (
    candidate_probabilities_from_volume,
    load_candidate_detector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = PROJECT_ROOT / "configs" / "lidc_stage4_selected_runs.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_reports" / "full_ct_detection"
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


def load_detection_defaults(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("full_ct_detection", {})


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_run(selection_path, task, model="resnet3d", input_dim="3d"):
    with Path(selection_path).open("r", encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    for item in selection["runs"]:
        if (
            item["task"] == task
            and item["model"] == model
            and item["input_dim"] == input_dim
        ):
            return item
    raise KeyError("No selected {} {} {} run.".format(input_dim, model, task))


def resolve_run(item):
    run_dir = PROJECT_ROOT / item["run_dir"]
    config_path = run_dir / "config.json"
    if item.get("checkpoint"):
        checkpoint = run_dir / item["checkpoint"]
    else:
        best = read_json(run_dir / "best_config.json")
        checkpoint = Path(best["best_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    return config_path, checkpoint


def model_probabilities(model, config, patches, device, batch_size=4):
    if not patches:
        return np.zeros((0, 2), dtype=np.float32)
    mean, std = normalization_stats(config)
    probabilities = []
    for start in range(0, len(patches), int(batch_size)):
        batch = np.stack(patches[start:start + int(batch_size)], axis=0)
        tensor = torch.from_numpy(batch).float().unsqueeze(1)
        tensor = torch.stack(
            [normalise_tensor(item, mean, std) for item in tensor],
            dim=0,
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor, pca_features=None), dim=1)
        probabilities.append(probs.cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def save_overview(volume_hu, candidates, output_path, max_panels=12):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = candidates[: int(max_panels)]
    if not selected:
        return
    columns = 4
    rows = int(np.ceil(len(selected) / float(columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, candidate in zip(axes, selected):
        z, y, x = [int(round(value)) for value in candidate["center_zyx"]]
        z = max(0, min(z, volume_hu.shape[0] - 1))
        axis.imshow(volume_hu[z], cmap="gray", vmin=-1000, vmax=400)
        axis.scatter([x], [y], s=70, facecolors="none", edgecolors="red")
        axis.set_title(
            "#{:02d} nodule={:.2f}\nmalignant={:.2f} risk={}".format(
                int(candidate["candidate_id"]),
                float(candidate["nodule_probability"]),
                float(candidate.get("prob_malignant", 0.0)),
                candidate.get("predicted_risk", ""),
            ),
            fontsize=8,
        )
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def parse_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    known, _ = pre_parser.parse_known_args(argv)
    defaults = load_detection_defaults(known.config)
    parser = argparse.ArgumentParser(
        description="Run complete CT nodule detection and risk classification.",
        parents=[pre_parser],
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--series-uid", default=None)
    parser.add_argument(
        "--selection",
        type=Path,
        default=resolve_project_path(
            defaults.get("selected_runs", DEFAULT_SELECTION)
        ),
    )
    parser.add_argument("--binary-config", type=Path, default=None)
    parser.add_argument("--binary-checkpoint", type=Path, default=None)
    parser.add_argument("--multiclass-config", type=Path, default=None)
    parser.add_argument("--multiclass-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=resolve_project_path(
            defaults.get("candidate_checkpoint", DEFAULT_CANDIDATE_CHECKPOINT)
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        "--min-proposal-score",
        type=float,
        default=float(defaults.get("min_proposal_score", 0.0)),
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=int(defaults.get("candidate_batch_size", 64)),
    )
    parser.add_argument("--save-rois", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def run_full_ct_detection(args, progress_callback=None):
    def report(message):
        if progress_callback is not None:
            progress_callback(message)

    report("Resolving selected model checkpoints")
    if args.binary_config and args.binary_checkpoint:
        binary_config_path = args.binary_config
        binary_checkpoint = args.binary_checkpoint
    else:
        binary_config_path, binary_checkpoint = resolve_run(
            selected_run(args.selection, "binary")
        )
    if args.multiclass_config and args.multiclass_checkpoint:
        multiclass_config_path = args.multiclass_config
        multiclass_checkpoint = args.multiclass_checkpoint
    else:
        multiclass_config_path, multiclass_checkpoint = resolve_run(
            selected_run(args.selection, "multiclass")
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    report("Reading the complete CT series")
    source_image, metadata = read_ct_image(args.input, series_uid=args.series_uid)
    report("Resampling CT to 1 mm isotropic spacing")
    image = resample_image(source_image, spacing_xyz=(1.0, 1.0, 1.0))
    volume_hu = sitk_to_numpy(image)
    report("Segmenting lungs and generating high-recall proposals")
    candidates, proposal_metadata = propose_nodule_candidates(
        volume_hu,
        response_percentile=args.response_percentile,
        min_distance_mm=args.min_distance_mm,
        max_candidates=args.max_proposals,
    )
    candidates = [
        candidate
        for candidate in candidates
        if float(candidate["proposal_score"]) >= float(args.min_proposal_score)
    ]
    candidate_model, candidate_config = load_candidate_detector(
        args.candidate_checkpoint,
        device,
    )
    report("Scoring proposals with the 3D CBAM candidate detector")
    proposal_count_before_candidate_model = len(candidates)
    nodule_probabilities = candidate_probabilities_from_volume(
        candidate_model,
        volume_hu,
        candidates,
        device,
        batch_size=args.candidate_batch_size,
    )
    for candidate, probability in zip(candidates, nodule_probabilities):
        candidate["nodule_probability"] = float(probability)
        candidate["detection_score"] = float(
            0.85 * probability + 0.15 * float(candidate["proposal_score"])
        )
    candidates = [
        candidate
        for candidate in candidates
        if candidate["nodule_probability"] >= float(args.nodule_threshold)
    ]
    candidates.sort(key=lambda row: row["detection_score"], reverse=True)
    candidates = candidates[: int(args.max_detections)]

    patches = [
        normalise_hu(
            fixed_crop_or_pad(
                volume_hu,
                candidate["center_zyx"],
                (64, 64, 64),
                pad_value=-1000.0,
            )
        )
        for candidate in candidates
    ]

    binary_config = read_json(binary_config_path)
    multiclass_config = read_json(multiclass_config_path)
    report("Loading malignancy and risk-stratification models")
    binary_model = load_model(binary_checkpoint, binary_config, device)
    multiclass_model = load_model(multiclass_checkpoint, multiclass_config, device)
    report("Classifying retained nodule candidates")
    binary_probs = model_probabilities(binary_model, binary_config, patches, device)
    multiclass_probs = model_probabilities(
        multiclass_model, multiclass_config, patches, device
    )

    report("Writing candidate tables, ROIs, metadata, and overview")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = args.output_dir / "candidate_rois"
    if args.save_rois:
        roi_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, candidate in enumerate(candidates):
        world = numpy_index_to_world(image, candidate["center_zyx"])
        binary_id = int(binary_probs[index].argmax())
        multiclass_id = int(multiclass_probs[index].argmax())
        row = dict(candidate)
        row.update({
            "candidate_id": index + 1,
            "world_x": float(world[0]),
            "world_y": float(world[1]),
            "world_z": float(world[2]),
            "predicted_binary_label": BINARY_LABELS[binary_id],
            "prob_benign": float(binary_probs[index, 0]),
            "prob_malignant": float(binary_probs[index, 1]),
            "predicted_risk": MULTICLASS_LABELS[multiclass_id],
            "prob_low_risk": float(multiclass_probs[index, 0]),
            "prob_intermediate_risk": float(multiclass_probs[index, 1]),
            "prob_high_risk": float(multiclass_probs[index, 2]),
            "combined_priority_score": float(
                candidate["detection_score"]
                * (0.5 + 0.5 * binary_probs[index, 1])
            ),
        })
        if args.save_rois:
            roi_path = roi_dir / "candidate_{:03d}.npz".format(index + 1)
            np.savez_compressed(
                str(roi_path),
                volume=patches[index].astype(np.float32),
                center_world_xyz=np.asarray(world, dtype=np.float32),
            )
            row["roi_path"] = str(roi_path)
        rows.append(row)

    rows.sort(key=lambda row: row["combined_priority_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["priority_rank"] = rank
    write_csv(args.output_dir / "full_ct_nodule_candidates.csv", rows)
    save_overview(
        volume_hu,
        rows,
        args.output_dir / "full_ct_detection_overview.png",
    )
    metadata.update({
        "resampled_size_xyz": list(image.GetSize()),
        "resampled_spacing_xyz": list(image.GetSpacing()),
        "proposal": proposal_metadata,
        "proposal_count_before_candidate_model": proposal_count_before_candidate_model,
        "candidate_count": len(rows),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "candidate_model_config": candidate_config,
        "nodule_probability_threshold": args.nodule_threshold,
        "max_detections": args.max_detections,
        "proposal_min_distance_mm": args.min_distance_mm,
        "binary_config": str(binary_config_path),
        "binary_checkpoint": str(binary_checkpoint),
        "multiclass_config": str(multiclass_config_path),
        "multiclass_checkpoint": str(multiclass_checkpoint),
        "candidate_generation": (
            "High-recall 3D LoG, axial-component, and full-resolution small-nodule "
            "local-maximum proposals inside a segmented lung mask, followed by a "
            "LUNA16-trained 3D CBAM false-positive reducer."
        ),
        "clinical_limit": (
            "Research prototype. It is not a certified clinical diagnostic system and "
            "requires prospective multi-centre validation before clinical use."
        ),
    })
    with (args.output_dir / "full_ct_detection_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    report("Complete CT detection finished")
    return {
        "candidate_csv": args.output_dir / "full_ct_nodule_candidates.csv",
        "overview_png": args.output_dir / "full_ct_detection_overview.png",
        "metadata_json": args.output_dir / "full_ct_detection_metadata.json",
        "candidate_count": len(rows),
        "rows": rows,
    }


def main(argv=None):
    args = parse_args(argv)
    result = run_full_ct_detection(
        args,
        progress_callback=lambda message: print("[full-ct] {}".format(message), flush=True),
    )
    print("Candidates: {}".format(result["candidate_csv"]))
    print("Candidate count: {}".format(result["candidate_count"]))


if __name__ == "__main__":
    main()
