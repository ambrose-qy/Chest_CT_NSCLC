"""
Evaluate a trained LIDC-IDRI Lightning checkpoint on the independent test set.

Example:

    conda run -n torch-gpu python data_analysis/LIDC_evaluate_lightning.py ^
      --checkpoint data/processed/models/lidc_lightning/2d/2d_resnet18_binary_seed2026/checkpoints/last.ckpt ^
      --config data/processed/models/lidc_lightning/2d/2d_resnet18_binary_seed2026/config.json
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lidc_lightning_data import BINARY_LABELS, LIDCDataModule, MULTICLASS_LABELS
from lidc_lightning_utils import (
    binary_metrics,
    confusion_matrix_rows,
    import_lightning,
    log,
    multiclass_metrics,
    roc_curve_rows,
    write_csv,
    write_json,
)
from lidc_lightning_train_utils import trusted_local_checkpoint_load


DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_reports" / "lidc_lightning"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a LIDC Lightning checkpoint on the test set.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Lightning checkpoint path.")
    parser.add_argument("--config", type=Path, default=None, help="Training config.json from the run directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest path if --config is omitted.")
    parser.add_argument("--input-dim", choices=["2d", "3d"], default=None)
    parser.add_argument("--task", choices=["binary", "multiclass"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--enable-grad-cam", type=int, default=1, help="Write Grad-CAM and Grad-CAM++ visualisations.")
    parser.add_argument("--grad-cam-max-samples", type=int, default=12, help="Maximum test samples visualised with Grad-CAM.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    labels = []
    probabilities = []
    roi_ids = []
    patient_ids = []
    for batch in dataloader:
        images = batch["image"].to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities.append(probs)
        labels.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())
        roi_ids.extend(batch["roi_id"])
        patient_ids.extend(batch["patient_id"])
    probabilities = np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, 2))
    return labels, probabilities, roi_ids, patient_ids


def rows_for_predictions(labels, probabilities, roi_ids, patient_ids, class_names):
    rows = []
    predictions = probabilities.argmax(axis=1)
    for idx, label in enumerate(labels):
        row = {
            "roi_id": roi_ids[idx],
            "patient_id": patient_ids[idx],
            "true_label_id": int(label),
            "true_label": class_names[int(label)] if int(label) < len(class_names) else str(label),
            "predicted_label_id": int(predictions[idx]),
            "predicted_label": class_names[int(predictions[idx])] if int(predictions[idx]) < len(class_names) else str(predictions[idx]),
        }
        for class_id, class_name in enumerate(class_names):
            row["prob_{}".format(class_name)] = float(probabilities[idx, class_id])
        rows.append(row)
    return rows


def maybe_write_figures(output_dir, labels, probabilities, class_names, task):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve
    except Exception:
        return

    labels_np = np.asarray(labels, dtype=np.int64)
    preds = probabilities.argmax(axis=1)
    cm = confusion_matrix(labels_np, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, values_format="d", colorbar=False)
    fig.tight_layout()
    fig.savefig(str(Path(output_dir) / "confusion_matrix.png"), dpi=300)
    plt.close(fig)

    if task == "binary" and len(set(labels)) >= 2:
        fpr, tpr, _ = roc_curve(labels_np, probabilities[:, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
        fig.tight_layout()
        fig.savefig(str(Path(output_dir) / "roc_curve.png"), dpi=300)
        plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    import_lightning()
    from lidc_grad_cam import generate_grad_cam_visualizations
    from lidc_lightning_module import LIDCClassifier

    config = read_json(args.config) if args.config else {}
    config_args = config.get("args", {})
    manifest = args.manifest or Path(config.get("manifest_path", ""))
    input_dim = args.input_dim or config.get("input_dim")
    task = args.task or config_args.get("task", "binary")
    batch_size = args.batch_size or int(config_args.get("batch_size", 8))
    image_size = args.image_size or int(config_args.get("image_size", 224))
    num_workers = args.num_workers

    if not manifest or not Path(manifest).exists():
        raise FileNotFoundError("Missing manifest. Pass --manifest or --config.")
    if input_dim not in ("2d", "3d"):
        raise ValueError("Missing input_dim. Pass --input-dim or --config.")

    class_names = BINARY_LABELS if task == "binary" else MULTICLASS_LABELS
    data_module = LIDCDataModule(
        manifest_path=manifest,
        input_dim=input_dim,
        task=task,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        max_samples_per_split=args.max_samples,
    )
    data_module.setup("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_kwargs = {}
    if config.get("class_weights") is not None:
        load_kwargs["class_weights"] = config.get("class_weights")
    with trusted_local_checkpoint_load():
        model = LIDCClassifier.load_from_checkpoint(str(args.checkpoint), map_location=device, **load_kwargs)
    model.to(device)
    labels, probabilities, roi_ids, patient_ids = predict(model, data_module.test_dataloader(), device)

    if task == "binary":
        metrics = binary_metrics(labels, probabilities[:, 1])
        roc_rows = roc_curve_rows(labels, probabilities[:, 1])
    else:
        metrics = multiclass_metrics(labels, probabilities)
        roc_rows = []

    run_name = args.checkpoint.parent.parent.name if args.checkpoint.parent.name == "checkpoints" else args.checkpoint.stem
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "test_metrics.json", metrics)
    write_csv(output_dir / "test_predictions.csv", rows_for_predictions(labels, probabilities, roi_ids, patient_ids, class_names))
    write_csv(output_dir / "confusion_matrix.csv", confusion_matrix_rows(labels, probabilities, class_names))
    if roc_rows:
        write_csv(output_dir / "roc_curve.csv", roc_rows)
    maybe_write_figures(output_dir, labels, probabilities, class_names, task)
    if bool(args.enable_grad_cam):
        grad_cam_payload = generate_grad_cam_visualizations(
            model,
            data_module,
            output_dir,
            input_dim=input_dim,
            task=task,
            class_names=class_names,
            max_samples=args.grad_cam_max_samples,
            device=device,
        )
        write_json(output_dir / "grad_cam_summary.json", grad_cam_payload)

    log("Evaluation complete")
    log("  checkpoint: {}".format(args.checkpoint))
    log("  metrics: {}".format(output_dir / "test_metrics.json"))
    log("  predictions: {}".format(output_dir / "test_predictions.csv"))
    log("  confusion matrix: {}".format(output_dir / "confusion_matrix.csv"))


if __name__ == "__main__":
    main()
