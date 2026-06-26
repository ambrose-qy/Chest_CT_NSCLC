"""Prepare a balanced LUNA16 3D candidate patch dataset."""

from __future__ import print_function

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from luna16_external_utils import (
    extract_world_patch,
    load_luna_scan_index,
    normalise_hu,
    parse_shape,
    read_csv_rows,
    read_luna_scan,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = Path("E:/LUNA/candidates.csv")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "luna16_candidate_dataset"


def select_rows(rows, scan_index, negative_ratio, seed):
    by_subset = defaultdict(lambda: {"0": [], "1": []})
    for row in rows:
        scan = scan_index.get(row.get("seriesuid", ""))
        label = str(row.get("class", "")).strip()
        if scan and label in ("0", "1"):
            out = dict(row)
            out["subset"] = scan.get("subset", "")
            out["mhd_path"] = scan.get("mhd_path", "")
            by_subset[out["subset"]][label].append(out)

    rng = random.Random(int(seed))
    selected = []
    for subset, classes in sorted(by_subset.items()):
        positives = classes["1"]
        negatives = classes["0"]
        rng.shuffle(negatives)
        negative_count = min(
            len(negatives),
            int(round(len(positives) * float(negative_ratio))),
        )
        selected.extend(positives)
        selected.extend(negatives[:negative_count])
    rng.shuffle(selected)
    return selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract LUNA16 positive/negative candidate patches."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shape", default="32,32,32")
    parser.add_argument("--negative-ratio", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    shape = parse_shape(args.shape)
    scan_index = load_luna_scan_index()
    selected = select_rows(
        read_csv_rows(args.candidates),
        scan_index,
        args.negative_ratio,
        args.seed,
    )
    if args.max_samples is not None:
        selected = selected[: int(args.max_samples)]
    if not selected:
        raise RuntimeError("No LUNA16 candidate rows selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_path = args.output_dir / "candidate_patches.npy"
    label_path = args.output_dir / "candidate_labels.npy"
    if (
        patch_path.exists()
        and label_path.exists()
        and not args.force
    ):
        raise FileExistsError(
            "Candidate dataset exists. Pass --force to rebuild: {}".format(args.output_dir)
        )

    patches = np.lib.format.open_memmap(
        str(patch_path),
        mode="w+",
        dtype=np.float16,
        shape=(len(selected), 1) + tuple(shape),
    )
    labels = np.lib.format.open_memmap(
        str(label_path),
        mode="w+",
        dtype=np.int8,
        shape=(len(selected),),
    )
    by_series = defaultdict(list)
    for index, row in enumerate(selected):
        by_series[row["seriesuid"]].append((index, row))

    manifest_rows = []
    failures = []
    for scan_number, (seriesuid, items) in enumerate(sorted(by_series.items()), start=1):
        print(
            "[{}/{}] {} candidates={}".format(
                scan_number, len(by_series), seriesuid, len(items)
            ),
            flush=True,
        )
        try:
            volume_hu, info = read_luna_scan(items[0][1]["mhd_path"])
        except Exception as exc:
            for index, row in items:
                failures.append({
                    "array_index": index,
                    "seriesuid": seriesuid,
                    "reason": "scan_read_failed",
                    "detail": str(exc),
                })
            continue
        for index, row in items:
            try:
                world = [
                    float(row["coordX"]),
                    float(row["coordY"]),
                    float(row["coordZ"]),
                ]
                patch = normalise_hu(
                    extract_world_patch(
                        volume_hu,
                        info,
                        center_world_xyz=world,
                        output_shape=shape,
                    )
                )
                label = int(row["class"])
                patches[index, 0] = patch.astype(np.float16)
                labels[index] = label
                manifest_rows.append({
                    "array_index": index,
                    "seriesuid": seriesuid,
                    "subset": row.get("subset", ""),
                    "coordX": row.get("coordX", ""),
                    "coordY": row.get("coordY", ""),
                    "coordZ": row.get("coordZ", ""),
                    "label": label,
                    "patch_shape_zyx": "x".join(str(value) for value in shape),
                })
            except Exception as exc:
                failures.append({
                    "array_index": index,
                    "seriesuid": seriesuid,
                    "reason": "patch_extract_failed",
                    "detail": str(exc),
                })

    patches.flush()
    labels.flush()
    write_csv(args.output_dir / "candidate_patch_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "candidate_patch_failures.csv", failures)
    summary = {
        "selected_sample_count": len(selected),
        "written_sample_count": len(manifest_rows),
        "failure_count": len(failures),
        "patch_shape_zyx": shape,
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "label_counts": dict(Counter(row["label"] for row in manifest_rows)),
        "subset_counts": dict(Counter(row["subset"] for row in manifest_rows)),
        "patch_array": str(patch_path),
        "label_array": str(label_path),
        "manifest": str(args.output_dir / "candidate_patch_manifest.csv"),
    }
    with (args.output_dir / "candidate_dataset_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
