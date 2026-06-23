"""
Shared training helpers for LIDC-IDRI Lightning scripts.
"""

from __future__ import print_function

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import torch

from lidc_lightning_data import BINARY_LABELS, LIDCDataModule, MULTICLASS_LABELS
from lidc_lightning_utils import (
    EXPERIMENT_DIR,
    confusion_matrix_rows,
    import_lightning,
    log,
    set_seed,
    write_csv,
    write_json,
)
from lidc_lightning_models import count_parameters


FIXED_MULTICLASS_CLASS_WEIGHTS = [1.20, 0.90, 1.30]
CHECKPOINT_MONITORS = {
    "auc_roc": "val_auc_roc",
    "f1": "val_f1",
}


@contextmanager
def trusted_local_checkpoint_load():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*torch.load.*weights_only=False.*",
            category=FutureWarning,
        )
        yield


def configure_torch_matmul_precision(value="medium"):
    if not torch.cuda.is_available():
        return None
    if not hasattr(torch, "set_float32_matmul_precision"):
        return None
    torch.set_float32_matmul_precision(value)
    return value


def class_weights_from_counts(counts, num_classes):
    total = sum(counts.values())
    weights = []
    for class_id in range(num_classes):
        count = counts.get(class_id, 0)
        weights.append(total / (num_classes * count) if count else 1.0)
    return weights


def sqrt_class_weights_from_counts(counts, num_classes):
    return [float(np.sqrt(weight)) for weight in class_weights_from_counts(counts, num_classes)]


def parse_class_weights(value):
    if value in (None, "", "auto"):
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def resolve_class_weights(args, counts, num_classes):
    if getattr(args, "no_class_weights", False):
        return None
    mode = getattr(args, "class_weight_mode", "balanced")
    if mode == "none":
        return None
    if mode == "fixed_multiclass":
        if num_classes == 3:
            return list(FIXED_MULTICLASS_CLASS_WEIGHTS)
        return sqrt_class_weights_from_counts(counts, num_classes)
    if mode == "custom":
        weights = parse_class_weights(getattr(args, "custom_class_weights", None))
        if weights is None or len(weights) != num_classes:
            raise ValueError("custom_class_weights must provide {} comma-separated values.".format(num_classes))
        return weights
    if mode == "sqrt_balanced":
        return sqrt_class_weights_from_counts(counts, num_classes)
    return class_weights_from_counts(counts, num_classes)


def serialise_args(args):
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def score_to_float(score):
    if score is None:
        return None
    try:
        return float(score.detach().cpu())
    except AttributeError:
        return float(score)


def checkpoint_summary(callback, monitor):
    return {
        "monitor_metric": monitor,
        "best_checkpoint": getattr(callback, "best_model_path", ""),
        "best_score": score_to_float(getattr(callback, "best_model_score", None)),
    }


def checkpoint_tag_for_monitor(monitor):
    for tag, metric in CHECKPOINT_MONITORS.items():
        if monitor == metric:
            return tag
    return "auc_roc"


def make_metric_guard_callback(pl, args):
    if not bool(getattr(args, "metric_guard_enabled", True)):
        return None

    class MetricGuardCallback(pl.Callback):
        def __init__(self):
            super().__init__()
            self.warmup_epochs = int(getattr(args, "metric_guard_warmup_epochs", 3))
            self.patience = max(int(getattr(args, "metric_guard_patience", 2)), 1)
            self.min_val_auc_roc = float(getattr(args, "metric_guard_min_val_auc_roc", 0.50))
            self.min_val_f1 = float(getattr(args, "metric_guard_min_val_f1", 0.10))
            self.min_val_accuracy = float(getattr(args, "metric_guard_min_val_accuracy", 0.0))
            self.max_val_loss = float(getattr(args, "metric_guard_max_val_loss", 5.0))
            self.bad_counts = {}
            self.stop_reason = ""

        @staticmethod
        def metric_float(trainer, name):
            value = trainer.callback_metrics.get(name)
            if value is None:
                return None
            try:
                return float(value.detach().cpu())
            except AttributeError:
                return float(value)

        @staticmethod
        def is_finite(value):
            return value is None or np.isfinite(value)

        def enabled_threshold(self, value):
            return value is not None and value > 0.0

        def on_validation_epoch_end(self, trainer, pl_module):
            if getattr(trainer, "sanity_checking", False):
                return

            epoch_number = int(trainer.current_epoch) + 1
            metrics = {
                "val_auc_roc": self.metric_float(trainer, "val_auc_roc"),
                "val_f1": self.metric_float(trainer, "val_f1"),
                "val_accuracy": self.metric_float(trainer, "val_accuracy"),
                "val_loss": self.metric_float(trainer, "val_loss"),
            }

            reasons = []
            for name, value in metrics.items():
                if not self.is_finite(value):
                    reasons.append("{}_non_finite".format(name))

            if epoch_number > self.warmup_epochs:
                if self.enabled_threshold(self.min_val_auc_roc) and metrics["val_auc_roc"] is not None:
                    if metrics["val_auc_roc"] < self.min_val_auc_roc:
                        reasons.append("val_auc_roc_below_{:.3f}".format(self.min_val_auc_roc))
                if self.enabled_threshold(self.min_val_f1) and metrics["val_f1"] is not None:
                    if metrics["val_f1"] < self.min_val_f1:
                        reasons.append("val_f1_below_{:.3f}".format(self.min_val_f1))
                if self.enabled_threshold(self.min_val_accuracy) and metrics["val_accuracy"] is not None:
                    if metrics["val_accuracy"] < self.min_val_accuracy:
                        reasons.append("val_accuracy_below_{:.3f}".format(self.min_val_accuracy))
                if self.enabled_threshold(self.max_val_loss) and metrics["val_loss"] is not None:
                    if metrics["val_loss"] > self.max_val_loss:
                        reasons.append("val_loss_above_{:.3f}".format(self.max_val_loss))

            active_reasons = set(reasons)
            for reason in list(self.bad_counts.keys()):
                if reason not in active_reasons:
                    self.bad_counts[reason] = 0
            for reason in active_reasons:
                self.bad_counts[reason] = self.bad_counts.get(reason, 0) + 1
                if self.bad_counts[reason] >= self.patience:
                    self.stop_reason = "{} for {} validation epochs".format(reason, self.bad_counts[reason])
                    log("Metric guard stopping training: {}. Metrics: {}".format(self.stop_reason, metrics))
                    trainer.should_stop = True
                    break

    return MetricGuardCallback()


def build_confusion_matrix(labels, probabilities, class_names):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        if 0 <= actual < len(class_names) and 0 <= predicted < len(class_names):
            matrix[int(actual), int(predicted)] += 1
    return matrix


def write_confusion_matrix_png(path, matrix, class_names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test Confusion Matrix")

    max_value = int(matrix.max()) if matrix.size else 0
    threshold = max_value / 2.0 if max_value else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=300)
    plt.close(fig)
    return True


def write_test_confusion_matrix(output_dir, model, task):
    labels, probabilities = model.collect_epoch_outputs("test")
    if len(labels) == 0:
        return {}

    class_names = BINARY_LABELS if task == "binary" else MULTICLASS_LABELS
    matrix = build_confusion_matrix(labels, probabilities, class_names)
    rows = confusion_matrix_rows(labels, probabilities, class_names)

    csv_path = Path(output_dir) / "test_confusion_matrix.csv"
    json_path = Path(output_dir) / "test_confusion_matrix.json"
    png_path = Path(output_dir) / "test_confusion_matrix.png"

    write_csv(csv_path, rows)
    write_json(json_path, {
        "class_names": class_names,
        "matrix": matrix.astype(int).tolist(),
        "rows": rows,
    })
    png_written = write_confusion_matrix_png(png_path, matrix, class_names)

    payload = {
        "confusion_matrix_csv": str(csv_path),
        "confusion_matrix_json": str(json_path),
    }
    if png_written:
        payload["confusion_matrix_png"] = str(png_path)
    return payload


def evaluate_checkpoint(
    trainer,
    model,
    data_module,
    output_dir,
    task,
    checkpoint_path,
    tag,
    write_legacy=False,
    metrics_filename=None,
    confusion_subdir=None,
):
    if not checkpoint_path:
        return {}

    with trusted_local_checkpoint_load():
        test_results = trainer.test(model, datamodule=data_module, ckpt_path=str(checkpoint_path))
    metrics = test_results[0] if test_results else {}

    output_dir = Path(output_dir)
    metrics_path = output_dir / (metrics_filename or "test_metrics_best_{}.json".format(tag))
    write_json(metrics_path, metrics)

    if write_legacy:
        write_json(output_dir / "test_metrics.json", metrics)
        confusion_payload = write_test_confusion_matrix(output_dir, model, task)
    else:
        subdir = confusion_subdir or "best_{}_test".format(tag)
        confusion_payload = write_test_confusion_matrix(output_dir / subdir, model, task)

    return {
        "checkpoint": str(checkpoint_path),
        "test_metrics": metrics,
        "test_metrics_path": str(metrics_path),
        "test_confusion_matrix": confusion_payload,
    }


def run_lightning_training(args, input_dim, manifest_path):
    pl = import_lightning()
    from lidc_grad_cam import generate_grad_cam_visualizations
    from lidc_lightning_module import LIDCClassifier
    set_seed(args.seed)
    matmul_precision = configure_torch_matmul_precision(getattr(args, "matmul_precision", "medium"))

    num_classes = 2 if args.task == "binary" else 3
    monitor_metric = args.monitor
    mode = "max" if monitor_metric not in ("val_loss", "train_loss") else "min"

    data_module = LIDCDataModule(
        manifest_path=manifest_path,
        input_dim=input_dim,
        task=args.task,
        batch_size=args.batch_size,
        image_size=getattr(args, "image_size", 224),
        num_workers=args.num_workers,
        max_samples_per_split=args.max_samples_per_split,
        in_channels=getattr(args, "in_channels", 3),
        normalization_mean=getattr(args, "normalization_mean", "auto"),
        normalization_std=getattr(args, "normalization_std", "auto"),
        normalization_stats_samples=getattr(args, "normalization_stats_samples", 128),
        augment_rotate=getattr(args, "augment_rotate", True),
        augment_flip=getattr(args, "augment_flip", True),
        augment_scale=getattr(args, "augment_scale", True),
        augment_scale_range=getattr(args, "augment_scale_range", 0.10),
        augment_noise_std=getattr(args, "augment_noise_std", 0.02),
        augment_intensity_shift=getattr(args, "augment_intensity_shift", 0.05),
        augment_contrast_range=getattr(args, "augment_contrast_range", 0.10),
        augment_cutout_fraction=getattr(args, "augment_cutout_fraction", 0.0),
        pca_features_enabled=getattr(args, "pca_features_enabled", 0),
        pca_n_components=getattr(args, "pca_n_components", 0),
        balanced_sampler=getattr(args, "balanced_sampler", 0),
        balanced_sampler_power=getattr(args, "balanced_sampler_power", 1.0),
    )
    data_module.setup()
    counts = data_module.class_counts()
    weights = resolve_class_weights(args, counts["train"], num_classes)

    model = LIDCClassifier(
        model_name=args.model,
        input_dim=input_dim,
        task=args.task,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=weights,
        label_smoothing=getattr(args, "label_smoothing", 0.0),
        pretrained=getattr(args, "pretrained", False),
        in_channels=getattr(args, "in_channels", None),
        scheduler=args.scheduler,
        max_epochs=args.epochs,
        dropout=getattr(args, "dropout", 0.2),
        gradient_clip_val=getattr(args, "gradient_clip_val", 0.0),
        attention=getattr(args, "attention", "none"),
        fusion=getattr(args, "fusion", "none"),
        pca_feature_dim=getattr(data_module, "pca_feature_dim", 0),
        pca_hidden_dim=getattr(args, "pca_hidden_dim", 16),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "{}_{}_{}_seed{}_{}".format(input_dim, args.model.lower(), args.task, args.seed, timestamp)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callbacks = {
        "auc_roc": pl.callbacks.ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="best_auc_roc-{epoch:03d}-{val_auc_roc:.4f}",
            monitor="val_auc_roc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        "f1": pl.callbacks.ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="best_f1-{epoch:03d}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
        ),
    }
    metric_guard_callback = make_metric_guard_callback(pl, args)
    callbacks = [
        checkpoint_callbacks["auc_roc"],
        checkpoint_callbacks["f1"],
        pl.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
        ),
    ]
    if metric_guard_callback is not None:
        callbacks.append(metric_guard_callback)
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    csv_logger = pl.loggers.CSVLogger(save_dir=str(output_dir), name="logs")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        logger=csv_logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=args.deterministic,
        enable_checkpointing=True,
        gradient_clip_val=getattr(args, "gradient_clip_val", 0.0),
        gradient_clip_algorithm=getattr(args, "gradient_clip_algorithm", "norm"),
    )

    config = {
        "args": serialise_args(args),
        "input_dim": input_dim,
        "manifest_path": str(manifest_path),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "class_counts": counts,
        "class_weights": weights,
        "class_weight_mode": getattr(args, "class_weight_mode", "balanced"),
        "balanced_sampler": bool(getattr(args, "balanced_sampler", 0)),
        "balanced_sampler_power": float(getattr(args, "balanced_sampler_power", 1.0)),
        "label_smoothing": float(getattr(args, "label_smoothing", 0.0)),
        "task": args.task,
        "split_column": "binary_split" if args.task == "binary" else "multiclass_split",
        "label_column": "binary_label_id" if args.task == "binary" else "multiclass_risk_label_id",
        "normalization_stats": data_module.normalization_stats,
        "augmentation": {
            "rotate": getattr(args, "augment_rotate", True),
            "flip": getattr(args, "augment_flip", True),
            "scale": getattr(args, "augment_scale", True),
            "scale_range": getattr(args, "augment_scale_range", 0.10),
            "noise_std": getattr(args, "augment_noise_std", 0.02),
            "intensity_shift": getattr(args, "augment_intensity_shift", 0.05),
            "contrast_range": getattr(args, "augment_contrast_range", 0.10),
            "cutout_fraction": getattr(args, "augment_cutout_fraction", 0.0),
        },
        "gradient_clip_val": getattr(args, "gradient_clip_val", 0.0),
        "gradient_clip_algorithm": getattr(args, "gradient_clip_algorithm", "norm"),
        "enable_grad_cam": bool(getattr(args, "enable_grad_cam", True)),
        "grad_cam_max_samples": getattr(args, "grad_cam_max_samples", 12),
        "dropout": getattr(args, "dropout", 0.2),
        "attention": getattr(args, "attention", "none"),
        "fusion": getattr(args, "fusion", "none"),
        "pca_features": {
            "enabled": bool(getattr(args, "pca_features_enabled", 0)),
            "requested_components": int(getattr(args, "pca_n_components", 0) or 0),
            "feature_dim": int(getattr(data_module, "pca_feature_dim", 0)),
            "hidden_dim": int(getattr(args, "pca_hidden_dim", 16) or 16),
            "source": data_module.pca_transform.get("source", "") if getattr(data_module, "pca_transform", None) else "",
            "explained_variance_ratio": data_module.pca_transform.get("explained_variance_ratio", []) if getattr(data_module, "pca_transform", None) else [],
        },
        "parameter_count": count_parameters(model),
        "matmul_precision": matmul_precision,
        "monitor_metric": monitor_metric,
        "checkpoint_monitors": CHECKPOINT_MONITORS,
        "primary_checkpoint_tag": checkpoint_tag_for_monitor(monitor_metric),
        "metric_guard": {
            "enabled": bool(getattr(args, "metric_guard_enabled", True)),
            "warmup_epochs": int(getattr(args, "metric_guard_warmup_epochs", 3)),
            "patience": int(getattr(args, "metric_guard_patience", 2)),
            "min_val_auc_roc": float(getattr(args, "metric_guard_min_val_auc_roc", 0.50)),
            "min_val_f1": float(getattr(args, "metric_guard_min_val_f1", 0.10)),
            "min_val_accuracy": float(getattr(args, "metric_guard_min_val_accuracy", 0.0)),
            "max_val_loss": float(getattr(args, "metric_guard_max_val_loss", 5.0)),
        },
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "hparams.json", serialise_args(args))

    log("Run: {}".format(run_name))
    log("Output: {}".format(output_dir))
    log("Class counts: {}".format(counts))
    log("Class weights: {}".format(weights))
    log("Task: {} using split column '{}'".format(args.task, config["split_column"]))
    log("Normalization stats: {}".format(data_module.normalization_stats))
    log("Trainable parameters: {}".format(config["parameter_count"]))
    if matmul_precision:
        log("Torch float32 matmul precision: {}".format(matmul_precision))

    trainer.fit(model, datamodule=data_module)

    final_checkpoint_path = output_dir / "checkpoints" / "final.ckpt"
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(final_checkpoint_path))

    checkpoint_payloads = {
        tag: checkpoint_summary(callback, CHECKPOINT_MONITORS[tag])
        for tag, callback in checkpoint_callbacks.items()
    }
    primary_tag = checkpoint_tag_for_monitor(monitor_metric)
    evaluation_order = [primary_tag] + [tag for tag in CHECKPOINT_MONITORS if tag != primary_tag]
    checkpoint_evaluations = {}
    for tag in evaluation_order:
        checkpoint_path = checkpoint_payloads[tag].get("best_checkpoint", "")
        checkpoint_evaluations[tag] = evaluate_checkpoint(
            trainer,
            model,
            data_module,
            output_dir,
            args.task,
            checkpoint_path,
            tag,
            write_legacy=tag == primary_tag,
        )
    final_evaluation = evaluate_checkpoint(
        trainer,
        model,
        data_module,
        output_dir,
        args.task,
        final_checkpoint_path,
        "final",
        write_legacy=False,
        metrics_filename="test_metrics_final.json",
        confusion_subdir="final_test",
    )

    primary_checkpoint = checkpoint_payloads.get(primary_tag, {})
    primary_evaluation = checkpoint_evaluations.get(primary_tag, {})
    best_path = primary_checkpoint.get("best_checkpoint", "")
    best_score = primary_checkpoint.get("best_score")
    confusion_payload = primary_evaluation.get("test_confusion_matrix", {})
    primary_test_metrics = primary_evaluation.get("test_metrics", {})
    grad_cam_payload = {}
    if bool(getattr(args, "enable_grad_cam", True)) and best_path:
        class_names = BINARY_LABELS if args.task == "binary" else MULTICLASS_LABELS
        with trusted_local_checkpoint_load():
            cam_model = LIDCClassifier.load_from_checkpoint(str(best_path), map_location="cpu")
        grad_cam_payload = generate_grad_cam_visualizations(
            cam_model,
            data_module,
            output_dir,
            input_dim=input_dim,
            task=args.task,
            class_names=class_names,
            max_samples=getattr(args, "grad_cam_max_samples", 12),
        )
    best_payload = {
        "run_name": run_name,
        "best_checkpoint": best_path,
        "best_score": best_score,
        "monitor_metric": monitor_metric,
        "primary_checkpoint_tag": primary_tag,
        "checkpoint_monitors": CHECKPOINT_MONITORS,
        "checkpoints": checkpoint_payloads,
        "checkpoint_evaluations": checkpoint_evaluations,
        "final_checkpoint": str(final_checkpoint_path),
        "final_evaluation": final_evaluation,
        "metric_guard_stop_reason": getattr(metric_guard_callback, "stop_reason", "") if metric_guard_callback is not None else "",
        "test_metrics": primary_test_metrics,
        "test_confusion_matrix": confusion_payload,
        "grad_cam": grad_cam_payload,
        "config_path": str(output_dir / "config.json"),
        "best_config_path": str(output_dir / "best_config.json"),
    }
    write_json(output_dir / "best_config.json", best_payload)
    log("Best checkpoint: {}".format(best_path))
    log("Best {}: {}".format(monitor_metric, best_payload["best_score"]))
    for tag, payload in checkpoint_payloads.items():
        log("Best {} checkpoint: {} ({})".format(tag, payload.get("best_checkpoint", ""), payload.get("best_score")))
    if confusion_payload:
        log("Test confusion matrix: {}".format(confusion_payload.get("confusion_matrix_csv")))
    if grad_cam_payload:
        log("Grad-CAM outputs: {}".format(grad_cam_payload.get("grad_cam_dir")))
    return best_payload


def add_common_training_args(parser, default_output_dir=None):
    parser.add_argument("--model", default=None, help="Model architecture.")
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"], help="Classification task.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir or EXPERIMENT_DIR, help="Experiment output directory.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau", "none"])
    parser.add_argument("--monitor", default="val_auc_roc", help="Early-stopping/checkpoint monitor metric.")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--metric-guard-enabled", type=int, default=1, help="Stop early when validation metrics are clearly invalid or too poor.")
    parser.add_argument("--metric-guard-warmup-epochs", type=int, default=3, help="Epochs to ignore before threshold-based metric guard checks.")
    parser.add_argument("--metric-guard-patience", type=int, default=2, help="Consecutive bad validation epochs before metric guard stops.")
    parser.add_argument("--metric-guard-min-val-auc-roc", type=float, default=0.50, help="Stop if validation AUROC stays below this after warmup. Use 0 to disable.")
    parser.add_argument("--metric-guard-min-val-f1", type=float, default=0.10, help="Stop if validation F1 stays below this after warmup. Use 0 to disable.")
    parser.add_argument("--metric-guard-min-val-accuracy", type=float, default=0.0, help="Stop if validation accuracy stays below this after warmup. Use 0 to disable.")
    parser.add_argument("--metric-guard-max-val-loss", type=float, default=5.0, help="Stop if validation loss stays above this after warmup. Use 0 to disable.")
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0 on Windows unless multiprocessing is configured.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--precision", default="32-true", help="Lightning precision setting, e.g. 32-true or 16-mixed.")
    parser.add_argument("--matmul-precision", default="medium", choices=["highest", "high", "medium"], help="Torch float32 matmul precision for CUDA Tensor Cores.")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Debug limit per split.")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--class-weight-mode", default="balanced", choices=["balanced", "sqrt_balanced", "fixed_multiclass", "custom", "none"])
    parser.add_argument("--custom-class-weights", default=None, help="Comma-separated class weights, e.g. 1.0,1.5.")
    parser.add_argument("--balanced-sampler", type=int, default=0, help="Use inverse-frequency weighted sampling for the training split.")
    parser.add_argument("--balanced-sampler-power", type=float, default=1.0, help="Strength of weighted sampling. 1.0 fully balances classes; 0.5 is gentler.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="CrossEntropy label smoothing. Helpful for unstable multiclass runs.")
    parser.add_argument("--normalization-mean", default="auto", help="auto or scalar/list mean for normalized HU values.")
    parser.add_argument("--normalization-std", default="auto", help="auto or scalar/list std for normalized HU values.")
    parser.add_argument("--normalization-stats-samples", type=int, default=128)
    parser.add_argument("--augment-rotate", type=int, default=1)
    parser.add_argument("--augment-flip", type=int, default=1)
    parser.add_argument("--augment-scale", type=int, default=1)
    parser.add_argument("--augment-scale-range", type=float, default=0.10)
    parser.add_argument("--augment-noise-std", type=float, default=0.02)
    parser.add_argument("--augment-intensity-shift", type=float, default=0.05)
    parser.add_argument("--augment-contrast-range", type=float, default=0.10)
    parser.add_argument("--augment-cutout-fraction", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pca-features-enabled", type=int, default=0, help="Fuse train-fitted PCA ROI statistic features with CNN logits.")
    parser.add_argument("--pca-n-components", type=int, default=0, help="Number of PCA ROI statistic components to use when enabled.")
    parser.add_argument("--pca-hidden-dim", type=int, default=16, help="Hidden size for the PCA/logit fusion head.")
    parser.add_argument("--attention", default="none", choices=["none", "se", "cbam"], help="Attention module. CBAM is recommended for 3D medical imaging.")
    parser.add_argument("--fusion", default="none", choices=["none", "multiscale", "multiview"], help="3D feature fusion mode.")
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--gradient-clip-algorithm", default="norm", choices=["norm", "value"])
    parser.add_argument("--enable-grad-cam", type=int, default=1, help="Write Grad-CAM and Grad-CAM++ visualisations after test.")
    parser.add_argument("--grad-cam-max-samples", type=int, default=12, help="Maximum test samples visualised with Grad-CAM.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    return parser
