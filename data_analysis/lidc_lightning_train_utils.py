"""
Shared training helpers for LIDC-IDRI Lightning scripts.
"""

from __future__ import print_function

from pathlib import Path

from lidc_lightning_data import LIDCDataModule
from lidc_lightning_utils import EXPERIMENT_DIR, import_lightning, log, set_seed, write_json
from lidc_lightning_models import count_parameters


def class_weights_from_counts(counts, num_classes):
    total = sum(counts.values())
    weights = []
    for class_id in range(num_classes):
        count = counts.get(class_id, 0)
        weights.append(total / (num_classes * count) if count else 1.0)
    return weights


def serialise_args(args):
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def run_lightning_training(args, input_dim, manifest_path):
    pl = import_lightning()
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
    )
    data_module.setup()
    counts = data_module.class_counts()
    weights = None if args.no_class_weights else class_weights_from_counts(counts["train"], num_classes)

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
    )

    config = {
        "args": serialise_args(args),
        "input_dim": input_dim,
        "manifest_path": str(manifest_path),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "class_counts": counts,
        "class_weights": weights,
        "parameter_count": count_parameters(model.model),
        "monitor_metric": monitor_metric,
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "hparams.json", serialise_args(args))

    log("Run: {}".format(run_name))
    log("Output: {}".format(output_dir))
    log("Class counts: {}".format(counts))
    log("Class weights: {}".format(weights))
    log("Trainable parameters: {}".format(config["parameter_count"]))

    trainer.fit(model, datamodule=data_module)
    test_results = trainer.test(model, datamodule=data_module, ckpt_path="best")
    write_json(output_dir / "test_metrics.json", test_results[0] if test_results else {})

    best_path = callbacks[0].best_model_path
    best_score = callbacks[0].best_model_score
    best_payload = {
        "run_name": run_name,
        "best_checkpoint": best_path,
        "best_score": None if best_score is None else float(best_score.detach().cpu()),
        "monitor_metric": monitor_metric,
        "test_metrics": test_results[0] if test_results else {},
        "config_path": str(output_dir / "config.json"),
    }
    write_json(output_dir / "best_config.json", best_payload)
    log("Best checkpoint: {}".format(best_path))
    log("Best {}: {}".format(monitor_metric, best_payload["best_score"]))
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
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    return parser
