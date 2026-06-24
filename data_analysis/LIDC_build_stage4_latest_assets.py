"""
Build the selected Stage-4 model summary and report figures.

The selected run list is maintained in configs/lidc_stage4_selected_runs.yaml.
This prevents historical duplicate runs from entering clinical, subgroup, and
ensemble comparisons.
"""

from __future__ import print_function

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = PROJECT_ROOT / "configs" / "lidc_stage4_selected_runs.yaml"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_results"
    / "lidc_lightning"
    / "stage4_latest_assets"
)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_value(metrics, name):
    value = metrics.get("test_{}".format(name), metrics.get(name, ""))
    try:
        return float(value)
    except Exception:
        return ""


def selected_checkpoint(run_dir, item, best_config):
    if item.get("checkpoint"):
        return run_dir / item["checkpoint"]
    value = best_config.get("best_checkpoint", "")
    return Path(value) if value else Path("")


def count_grad_cam_pngs(run_dir):
    return len(list((run_dir / "grad_cam").glob("*.png")))


def summary_row(item):
    run_dir = PROJECT_ROOT / item["run_dir"]
    config = read_json(run_dir / "config.json")
    best_config = read_json(run_dir / "best_config.json")
    metrics = read_json(run_dir / "test_metrics.json")
    args = config.get("args", {})
    checkpoint = selected_checkpoint(run_dir, item, best_config)
    prediction_path = run_dir / "test_predictions.csv"
    confusion_path = run_dir / "confusion_matrix.csv"
    roc_path = run_dir / "roc_curve.csv"

    required = [run_dir / "config.json", run_dir / "test_metrics.json", prediction_path, confusion_path]
    if item["task"] == "binary":
        required.append(roc_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Selected run {} is incomplete: {}".format(run_dir.name, ", ".join(missing))
        )

    return {
        "input_dim": item["input_dim"],
        "model": item["model"],
        "task": item["task"],
        "run_name": run_dir.name,
        "test_accuracy": metric_value(metrics, "accuracy"),
        "test_auc_roc": metric_value(metrics, "auc_roc"),
        "test_precision": metric_value(metrics, "precision"),
        "test_recall": metric_value(metrics, "recall"),
        "test_f1": metric_value(metrics, "f1"),
        "test_loss": metric_value(metrics, "loss"),
        "test_n": int(
            sum(
                float(value)
                for key, value in metrics.items()
                if key.startswith("test_label_") and key.endswith("_count")
            )
            or metrics.get("sample_count", 0)
        ),
        "lr": args.get("lr", ""),
        "batch_size": args.get("batch_size", ""),
        "epochs": args.get("epochs", ""),
        "weight_decay": args.get("weight_decay", ""),
        "scheduler": args.get("scheduler", ""),
        "early_stop_patience": args.get("early_stop_patience", ""),
        "dropout": config.get("dropout", args.get("dropout", "")),
        "class_weight_mode": config.get("class_weight_mode", args.get("class_weight_mode", "")),
        "balanced_sampler": config.get("balanced_sampler", args.get("balanced_sampler", "")),
        "balanced_sampler_power": config.get(
            "balanced_sampler_power", args.get("balanced_sampler_power", "")
        ),
        "attention": config.get("attention", args.get("attention", "")),
        "fusion": config.get("fusion", args.get("fusion", "")),
        "label_smoothing": config.get("label_smoothing", args.get("label_smoothing", "")),
        "parameter_count": config.get("parameter_count", ""),
        "checkpoint_source": item.get("checkpoint_source", ""),
        "selected_checkpoint": str(checkpoint),
        "grad_cam_png_count": count_grad_cam_pngs(run_dir),
        "prediction_path": str(prediction_path),
        "confusion_matrix_path": str(confusion_path),
        "roc_curve_path": str(roc_path) if roc_path.exists() else "",
        "run_dir": str(run_dir),
    }


def plot_task_comparison(rows, task, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_rows = [row for row in rows if row["task"] == task]
    labels = ["{} {}".format(row["input_dim"].upper(), row["model"]) for row in task_rows]
    metric_names = ["Accuracy", "AUC", "F1"]
    metric_keys = ["test_accuracy", "test_auc_roc", "test_f1"]
    values = np.asarray(
        [[float(row[key]) for key in metric_keys] for row in task_rows],
        dtype=np.float64,
    )
    x = np.arange(len(labels))
    width = 0.23

    fig, ax = plt.subplots(figsize=(9, 5.4))
    colors = ["#2F6B8A", "#D98B3A", "#4F8A5B"]
    for index, (metric_name, color) in enumerate(zip(metric_names, colors)):
        bars = ax.bar(x + (index - 1) * width, values[:, index], width, label=metric_name, color=color)
        ax.bar_label(bars, labels=["{:.3f}".format(value) for value in values[:, index]], padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("{} Model Comparison - Latest CBAM Runs".format(task.capitalize()))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def plot_all_models(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "{} {} {}".format(row["input_dim"].upper(), row["model"], "B" if row["task"] == "binary" else "M")
        for row in rows
    ]
    values = np.asarray(
        [[float(row["test_accuracy"]), float(row["test_auc_roc"]), float(row["test_f1"])] for row in rows],
        dtype=np.float64,
    )
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for index, (name, color) in enumerate(
        [("Accuracy", "#2F6B8A"), ("AUC", "#D98B3A"), ("F1", "#4F8A5B")]
    ):
        ax.bar(x + (index - 1) * width, values[:, index], width, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Latest CBAM Model Results")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def plot_binary_roc(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    colors = ["#2F6B8A", "#D98B3A", "#4F8A5B", "#8B5A83"]
    for row, color in zip(
        [candidate for candidate in rows if candidate["task"] == "binary"],
        colors,
    ):
        roc_path = Path(row["roc_curve_path"])
        points = []
        with roc_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                points.append(
                    (
                        float(item.get("fpr", item.get("false_positive_rate", 0.0))),
                        float(item.get("tpr", item.get("true_positive_rate", 0.0))),
                    )
                )
        points.sort(key=lambda value: value[0])
        label = "{} {} (AUC={:.3f})".format(
            row["input_dim"].upper(),
            row["model"],
            float(row["test_auc_roc"]),
        )
        ax.plot(
            [value[0] for value in points],
            [value[1] for value in points],
            linewidth=2.2,
            color=color,
            label=label,
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#777777", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Binary ROC Curves - Latest CBAM Runs")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def image_montage(items, output_path, title, columns=2, cell_size=(800, 520)):
    from PIL import Image, ImageDraw, ImageFont

    items = [(label, Path(path)) for label, path in items if Path(path).exists()]
    if not items:
        return False
    rows = int(np.ceil(len(items) / float(columns)))
    header_height = 54
    canvas = Image.new(
        "RGB",
        (columns * cell_size[0], rows * (cell_size[1] + header_height) + 70),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), title, fill="black")
    for index, (label, path) in enumerate(items):
        image = Image.open(str(path)).convert("RGB")
        image.thumbnail((cell_size[0] - 30, cell_size[1] - 20))
        x0 = (index % columns) * cell_size[0]
        y0 = 70 + (index // columns) * (cell_size[1] + header_height)
        draw.text((x0 + 15, y0 + 8), label, fill="black")
        image_x = x0 + (cell_size[0] - image.width) // 2
        image_y = y0 + header_height + (cell_size[1] - image.height) // 2
        canvas.paste(image, (image_x, image_y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), dpi=(200, 200))
    return True


def build_montages(rows, output_dir):
    training_items = []
    confusion_items = []
    grad_cam_items = []
    for row in rows:
        label = "{} {} {}".format(row["input_dim"].upper(), row["model"], row["task"])
        run_dir = Path(row["run_dir"])
        training_items.append(
            (
                label,
                output_dir
                / "training_curves"
                / row["run_name"]
                / "training_summary_curves.png",
            )
        )
        confusion_items.append((label, run_dir / "confusion_matrix.png"))
        grad_candidates = sorted((run_dir / "grad_cam").glob("*grad_cam_plus_plus.png"))
        if not grad_candidates:
            grad_candidates = sorted((run_dir / "grad_cam").glob("*grad_cam.png"))
        if grad_candidates:
            grad_cam_items.append((label, grad_candidates[0]))

    outputs = []
    specifications = [
        (
            training_items,
            output_dir / "stage4_training_curves_montage.png",
            "Training Curves - Latest CBAM Runs",
        ),
        (
            confusion_items,
            output_dir / "stage4_confusion_matrix_montage.png",
            "Confusion Matrices - Latest CBAM Runs",
        ),
        (
            grad_cam_items,
            output_dir / "stage4_gradcam_montage.png",
            "Grad-CAM++ Examples - Latest CBAM Runs",
        ),
    ]
    for items, path, title in specifications:
        if image_montage(items, path, title):
            outputs.append(str(path))
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build latest Stage-4 LIDC assets.")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with args.selection.open("r", encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    rows = [summary_row(item) for item in selection["runs"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "stage4_latest_model_summary.csv"
    write_csv(summary_path, rows)
    plot_all_models(rows, args.output_dir / "stage4_latest_all_models.png")
    plot_task_comparison(rows, "binary", args.output_dir / "stage4_latest_binary_models.png")
    plot_task_comparison(rows, "multiclass", args.output_dir / "stage4_latest_multiclass_models.png")
    plot_binary_roc(rows, args.output_dir / "stage4_latest_binary_roc.png")
    montage_paths = build_montages(rows, args.output_dir)
    figure_paths = [
        str(args.output_dir / "stage4_latest_all_models.png"),
        str(args.output_dir / "stage4_latest_binary_models.png"),
        str(args.output_dir / "stage4_latest_multiclass_models.png"),
        str(args.output_dir / "stage4_latest_binary_roc.png"),
    ] + montage_paths
    write_json(
        args.output_dir / "stage4_latest_asset_index.json",
        {
            "selection": str(args.selection),
            "selection_date": selection.get("selection_date", ""),
            "selection_rule": selection.get("selection_rule", ""),
            "model_count": len(rows),
            "summary_csv": str(summary_path),
            "figures": figure_paths,
        },
    )
    print("Stage-4 latest assets: {}".format(args.output_dir), flush=True)
    return rows


if __name__ == "__main__":
    main()
