"""
Plot training curves for a LIDC-IDRI Lightning experiment.

Edit the input block below each time you want to plot another run.
"""

from __future__ import print_function

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Inputs to modify each time
# =========================

RUN_DIR = Path("data/processed/model_results/lidc_lightning/3d/resnet/3d_resnet3d_binary_seed2026")

# Leave these as None for automatic discovery inside RUN_DIR.
METRICS_CSV = None
CONFIG_JSON = None
ROC_CSV = None
PREDICTIONS_CSV = None
OUTPUT_DIR = None

# Use "auto", "binary", or "multiclass".
TASK = "auto"

# Image style.
DPI = 300
FIGSIZE = (7, 4.5)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def latest_file(root, pattern):
    root = project_path(root)
    candidates = list(root.rglob(pattern)) if root and root.exists() else []
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_json(path):
    if not path or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def first_existing(*paths):
    for path in paths:
        if path and Path(path).exists():
            return Path(path)
    return None


def resolve_inputs():
    run_dir = project_path(RUN_DIR)
    metrics_csv = first_existing(project_path(METRICS_CSV), latest_file(run_dir, "metrics.csv"))
    config_json = first_existing(project_path(CONFIG_JSON), run_dir / "config.json")
    output_dir = project_path(OUTPUT_DIR) if OUTPUT_DIR else run_dir / "figures"
    roc_csv = first_existing(project_path(ROC_CSV), latest_file(run_dir, "roc_curve.csv"))
    predictions_csv = first_existing(project_path(PREDICTIONS_CSV), latest_file(run_dir, "test_predictions.csv"))

    if metrics_csv is None:
        raise FileNotFoundError("Could not find metrics.csv. Set RUN_DIR or METRICS_CSV at the top of this file.")

    output_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, metrics_csv, config_json, roc_csv, predictions_csv, output_dir


def last_valid(series):
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.iloc[-1]


def load_epoch_metrics(metrics_csv):
    df = pd.read_csv(metrics_csv)
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            df[column] = converted
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(len(df), dtype=float)
    numeric_cols = [column for column in df.columns if column != "epoch" and pd.api.types.is_numeric_dtype(df[column])]
    epoch_df = (
        df[["epoch"] + numeric_cols]
        .groupby("epoch", as_index=False)
        .agg(last_valid)
        .sort_values("epoch")
        .reset_index(drop=True)
    )
    return df, epoch_df


def available_columns(df, columns):
    return [column for column in columns if column in df.columns and df[column].notna().any()]


def plot_columns(df, columns, title, ylabel, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = available_columns(df, columns)
    if not columns:
        print("Skip {}: no matching columns".format(title))
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = df["epoch"].astype(float)
    for column in columns:
        y = pd.to_numeric(df[column], errors="coerce")
        ax.plot(x, y, marker="o", linewidth=1.8, label=column)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print("Wrote {}".format(output_path))
    return True


def plot_training_curves(epoch_df, output_dir):
    plot_columns(
        epoch_df,
        ["train_loss_epoch", "val_loss", "test_loss"],
        "Loss",
        "Loss",
        output_dir / "loss_curve.png",
    )
    plot_columns(
        epoch_df,
        ["train_accuracy", "val_accuracy", "test_accuracy"],
        "Accuracy",
        "Accuracy",
        output_dir / "accuracy_curve.png",
    )
    plot_columns(
        epoch_df,
        ["train_f1", "val_f1", "test_f1"],
        "F1 Score",
        "F1",
        output_dir / "f1_curve.png",
    )
    plot_columns(
        epoch_df,
        ["train_auc_roc", "val_auc_roc", "test_auc_roc"],
        "AUROC History",
        "AUROC",
        output_dir / "auroc_history.png",
    )
    plot_columns(
        epoch_df,
        ["train_avg_abs_gradient_epoch", "train_gradient_l2_norm_epoch", "train_max_abs_gradient_epoch"],
        "Gradient Statistics",
        "Gradient value",
        output_dir / "gradient_curve.png",
    )


def plot_combined_summary(epoch_df, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [
        ("Loss", "Loss", ["train_loss_epoch", "val_loss", "test_loss"]),
        ("Accuracy", "Accuracy", ["train_accuracy", "val_accuracy", "test_accuracy"]),
        ("F1 Score", "F1", ["train_f1", "val_f1", "test_f1"]),
        ("Gradient", "Gradient value", ["train_avg_abs_gradient_epoch", "train_gradient_l2_norm_epoch", "train_max_abs_gradient_epoch"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = epoch_df["epoch"].astype(float)
    for ax, (title, ylabel, columns) in zip(axes.ravel(), groups):
        cols = available_columns(epoch_df, columns)
        for column in cols:
            ax.plot(x, pd.to_numeric(epoch_df[column], errors="coerce"), marker="o", linewidth=1.5, label=column)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if cols:
            ax.legend(fontsize=8)

    fig.tight_layout()
    output_path = output_dir / "training_summary_curves.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print("Wrote {}".format(output_path))


def infer_task(config, predictions_csv):
    if TASK in ("binary", "multiclass"):
        return TASK
    task = config.get("task") or config.get("args", {}).get("task")
    if task:
        return task
    if predictions_csv and Path(predictions_csv).exists():
        df = pd.read_csv(predictions_csv, nrows=1)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        return "binary" if len(prob_cols) <= 2 else "multiclass"
    return "binary"


def plot_roc_from_csv(roc_csv, output_dir):
    if not roc_csv:
        return False
    roc_df = pd.read_csv(roc_csv)
    required = {"false_positive_rate", "true_positive_rate"}
    if not required.issubset(set(roc_df.columns)):
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(roc_df["false_positive_rate"], roc_df["true_positive_rate"], linewidth=2, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "roc_curve.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print("Wrote {}".format(output_path))
    return True


def plot_roc_from_predictions(predictions_csv, task, output_dir):
    if not predictions_csv or not Path(predictions_csv).exists():
        return False
    predictions = pd.read_csv(predictions_csv)
    if "true_label_id" not in predictions.columns:
        return False

    prob_cols = [column for column in predictions.columns if column.startswith("prob_")]
    if len(prob_cols) < 2:
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    labels = predictions["true_label_id"].astype(int).to_numpy()
    probabilities = predictions[prob_cols].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    wrote_curve = False
    if task == "binary":
        if len(set(labels.tolist())) >= 2:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            ax.plot(fpr, tpr, linewidth=2, label="ROC AUC={:.3f}".format(auc(fpr, tpr)))
            wrote_curve = True
    else:
        classes = list(range(len(prob_cols)))
        binary_labels = label_binarize(labels, classes=classes)
        for class_id, column in enumerate(prob_cols):
            if binary_labels[:, class_id].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary_labels[:, class_id], probabilities[:, class_id])
            ax.plot(fpr, tpr, linewidth=1.8, label="{} AUC={:.3f}".format(column.replace("prob_", ""), auc(fpr, tpr)))
            wrote_curve = True

    if not wrote_curve:
        plt.close(fig)
        return False

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "roc_curve.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print("Wrote {}".format(output_path))
    return True


def main():
    run_dir, metrics_csv, config_json, roc_csv, predictions_csv, output_dir = resolve_inputs()
    config = read_json(config_json)
    task = infer_task(config, predictions_csv)

    print("Run dir: {}".format(run_dir))
    print("Metrics: {}".format(metrics_csv))
    print("Config: {}".format(config_json if config_json and config_json.exists() else "not found"))
    print("Task: {}".format(task))
    print("Output: {}".format(output_dir))

    _, epoch_df = load_epoch_metrics(metrics_csv)
    epoch_df.to_csv(output_dir / "epoch_metrics_compact.csv", index=False)
    plot_training_curves(epoch_df, output_dir)
    plot_combined_summary(epoch_df, output_dir)

    if not plot_roc_from_csv(roc_csv, output_dir):
        if not plot_roc_from_predictions(predictions_csv, task, output_dir):
            print("ROC curve skipped: provide ROC_CSV or PREDICTIONS_CSV, or run LIDC_evaluate_lightning.py first.")


if __name__ == "__main__":
    main()
