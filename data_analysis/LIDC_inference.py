"""
Single-case and batch inference for trained LIDC Lightning classifiers.

Examples:

    C:\\Users\\Ambro\\.conda\\envs\\torch-gpu\\python.exe data_analysis\\LIDC_inference.py ^
      --checkpoint <run_dir>\\checkpoints\\best_f1-epoch=024-val_f1=0.4162.ckpt ^
      --config <run_dir>\\config.json ^
      --input-npz data\\processed\\lidc_roi_3d\\volumes\\example.npz

    C:\\Users\\Ambro\\.conda\\envs\\torch-gpu\\python.exe data_analysis\\LIDC_inference.py ^
      --checkpoint <best.ckpt> --config <run_dir>\\config.json ^
      --input-csv my_cases.csv --output-csv data\\processed\\model_reports\\inference_predictions.csv

Batch CSV columns:
    case_id,volume_path,dicom_path,roi_id,patient_id
"""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lidc_lightning_data import (
    BINARY_LABELS,
    HU_MAX,
    HU_MIN,
    MULTICLASS_LABELS,
    center_crop_or_pad_3d,
    load_dicom_hu,
    normalise_hu,
    normalise_tensor,
)
from lidc_lightning_train_utils import configure_torch_matmul_precision, trusted_local_checkpoint_load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "model_reports" / "lidc_inference" / "inference_predictions.csv"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


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


def clean_value(value):
    if value is None:
        return ""
    return str(value).strip()


def config_value(config, key, default=None):
    args = config.get("args", {})
    return config.get(key, args.get(key, default))


def normalization_stats(config):
    stats = config.get("normalization_stats", {}) or {}
    return stats.get("mean"), stats.get("std")


def load_3d_npz(path, target_shape=(64, 64, 64)):
    with np.load(str(path)) as npz:
        if "volume" not in npz:
            raise KeyError("NPZ file must contain array key 'volume': {}".format(path))
        volume = npz["volume"].astype(np.float32)
    tensor = torch.from_numpy(volume).float().unsqueeze(0)
    if tuple(tensor.shape[-3:]) != tuple(target_shape):
        tensor = center_crop_or_pad_3d(tensor, target_shape)
    return tensor


def load_2d_dicom(path, image_size=224, in_channels=3):
    image = normalise_hu(load_dicom_hu(path), HU_MIN, HU_MAX)
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False).squeeze(0)
    if int(in_channels) > 1:
        tensor = tensor.repeat(int(in_channels), 1, 1)
    return tensor


def make_case_rows(args):
    if args.input_csv:
        return read_csv_rows(args.input_csv)
    rows = []
    if args.input_npz:
        rows.append({"case_id": Path(args.input_npz).stem, "volume_path": str(args.input_npz)})
    if args.input_dicom:
        rows.append({"case_id": Path(args.input_dicom).stem, "dicom_path": str(args.input_dicom)})
    if not rows:
        raise ValueError("Pass --input-npz, --input-dicom, or --input-csv.")
    return rows


def tensor_for_case(row, config):
    input_dim = config.get("input_dim")
    mean, std = normalization_stats(config)
    if input_dim == "3d":
        path = clean_value(row.get("volume_path")) or clean_value(row.get("npz_path"))
        if not path:
            raise ValueError("3D inference requires volume_path or npz_path in batch CSV.")
        tensor = load_3d_npz(path)
    elif input_dim == "2d":
        path = clean_value(row.get("dicom_path")) or clean_value(row.get("image_path"))
        if not path:
            raise ValueError("2D inference requires dicom_path or image_path in batch CSV.")
        image_size = int(config.get("args", {}).get("image_size", 224))
        in_channels = int(config.get("args", {}).get("in_channels", 3))
        tensor = load_2d_dicom(path, image_size=image_size, in_channels=in_channels)
    else:
        raise ValueError("config input_dim must be '2d' or '3d'.")
    return normalise_tensor(tensor, mean, std)


def load_model(checkpoint, config, device):
    configure_torch_matmul_precision("medium")
    from lidc_lightning_module import LIDCClassifier

    load_kwargs = {}
    if config.get("class_weights") is not None:
        load_kwargs["class_weights"] = config.get("class_weights")
    with trusted_local_checkpoint_load():
        model = LIDCClassifier.load_from_checkpoint(str(checkpoint), map_location=device, **load_kwargs)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_cases(model, rows, config, device):
    task = config.get("task") or config.get("args", {}).get("task", "binary")
    class_names = BINARY_LABELS if task == "binary" else MULTICLASS_LABELS
    output = []
    for idx, row in enumerate(rows):
        tensor = tensor_for_case(row, config).unsqueeze(0).to(device)
        logits = model(tensor, pca_features=None)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        out = {
            "case_id": clean_value(row.get("case_id")) or clean_value(row.get("roi_id")) or "case_{}".format(idx),
            "roi_id": clean_value(row.get("roi_id")),
            "patient_id": clean_value(row.get("patient_id")),
            "input_path": clean_value(row.get("volume_path")) or clean_value(row.get("npz_path")) or clean_value(row.get("dicom_path")) or clean_value(row.get("image_path")),
            "task": task,
            "predicted_label_id": pred_id,
            "predicted_label": class_names[pred_id] if pred_id < len(class_names) else str(pred_id),
            "confidence": float(probs[pred_id]),
        }
        for class_id, name in enumerate(class_names):
            out["prob_{}".format(name)] = float(probs[class_id])
        output.append(out)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run trained LIDC model inference on single or batch inputs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--input-dicom", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = read_json(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model = load_model(args.checkpoint, config, device)
    rows = make_case_rows(args)
    predictions = predict_cases(model, rows, config, device)
    write_csv(args.output_csv, predictions)
    print("Inference predictions: {}".format(args.output_csv))
    if len(predictions) == 1:
        print(json.dumps(predictions[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
