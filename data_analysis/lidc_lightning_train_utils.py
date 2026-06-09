"""
Shared training helpers for LIDC-IDRI Lightning scripts.
"""

from __future__ import print_function

from pathlib import Path

import numpy as np

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


def run_lightning_training(args, input_dim, manifest_path):
    pl = import_lightning()
    from lidc_grad_cam import generate_grad_cam_visualizations
    from lidc_lightning_module import LIDCClassifier
    set_seed(args.seed)

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
        augment_noise_std=getattr(args, "augment_noise_std", 0.02),
        augment_intensity_shift=getattr(args, "augment_intensity_shift", 0.05),
        augment_contrast_range=getattr(args, "augment_contrast_range", 0.10),
        augment_cutout_fraction=getattr(args, "augment_cutout_fraction", 0.0),
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
        pretrained=getattr(args, "pretrained", False),
        in_channels=getattr(args, "in_channels", None),
        scheduler=args.scheduler,
        max_epochs=args.epochs,
        dropout=getattr(args, "dropout", 0.2),
        gradient_clip_val=getattr(args, "gradient_clip_val", 0.0),
    )

    run_name = "{}_{}_{}_seed{}".format(input_dim, args.model.lower(), args.task, args.seed)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="best-{epoch:03d}-{%s:.4f}" % monitor_metric,
            monitor=monitor_metric,
            mode=mode,
            save_top_k=1,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

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
        "task": args.task,
        "split_column": "binary_split" if args.task == "binary" else "multiclass_split",
        "label_column": "binary_label_id" if args.task == "binary" else "multiclass_risk_label_id",
        "normalization_stats": data_module.normalization_stats,
        "augmentation": {
            "rotate": getattr(args, "augment_rotate", True),
            "flip": getattr(args, "augment_flip", True),
            "scale": getattr(args, "augment_scale", True),
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
        "parameter_count": count_parameters(model.model),
        "monitor_metric": monitor_metric,
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

    trainer.fit(model, datamodule=data_module)
    test_results = trainer.test(model, datamodule=data_module, ckpt_path="best")
    write_json(output_dir / "test_metrics.json", test_results[0] if test_results else {})
    confusion_payload = write_test_confusion_matrix(output_dir, model, args.task)

    best_path = callbacks[0].best_model_path
    best_score = callbacks[0].best_model_score
    grad_cam_payload = {}
    if bool(getattr(args, "enable_grad_cam", True)) and best_path:
        class_names = BINARY_LABELS if args.task == "binary" else MULTICLASS_LABELS
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
        "best_score": None if best_score is None else float(best_score.detach().cpu()),
        "monitor_metric": monitor_metric,
        "test_metrics": test_results[0] if test_results else {},
        "test_confusion_matrix": confusion_payload,
        "grad_cam": grad_cam_payload,
        "config_path": str(output_dir / "config.json"),
        "best_config_path": str(output_dir / "best_config.json"),
    }
    write_json(output_dir / "best_config.json", best_payload)
    log("Best checkpoint: {}".format(best_path))
    log("Best {}: {}".format(monitor_metric, best_payload["best_score"]))
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
    parser.add_argument("--num-workers", type=int, default=0, help="Keep 0 on Windows unless multiprocessing is configured.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--precision", default="32-true", help="Lightning precision setting, e.g. 32-true or 16-mixed.")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Debug limit per split.")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--class-weight-mode", default="balanced", choices=["balanced", "sqrt_balanced", "custom", "none"])
    parser.add_argument("--custom-class-weights", default=None, help="Comma-separated class weights, e.g. 1.0,1.5.")
    parser.add_argument("--normalization-mean", default="auto", help="auto or scalar/list mean for normalized HU values.")
    parser.add_argument("--normalization-std", default="auto", help="auto or scalar/list std for normalized HU values.")
    parser.add_argument("--normalization-stats-samples", type=int, default=128)
    parser.add_argument("--augment-rotate", type=int, default=1)
    parser.add_argument("--augment-flip", type=int, default=1)
    parser.add_argument("--augment-scale", type=int, default=1)
    parser.add_argument("--augment-noise-std", type=float, default=0.02)
    parser.add_argument("--augment-intensity-shift", type=float, default=0.05)
    parser.add_argument("--augment-contrast-range", type=float, default=0.10)
    parser.add_argument("--augment-cutout-fraction", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--gradient-clip-algorithm", default="norm", choices=["norm", "value"])
    parser.add_argument("--enable-grad-cam", type=int, default=1, help="Write Grad-CAM and Grad-CAM++ visualisations after test.")
    parser.add_argument("--grad-cam-max-samples", type=int, default=12, help="Maximum test samples visualised with Grad-CAM.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    return parser
