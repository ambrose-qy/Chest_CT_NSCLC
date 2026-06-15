"""
LightningModule for LIDC-IDRI baseline classification experiments.
"""

from __future__ import print_function

from collections import Counter

import torch
import torch.nn as nn

from lidc_lightning_models import count_parameters, create_lidc_lightning_model
from lidc_lightning_utils import binary_metrics, import_lightning, multiclass_metrics


pl = import_lightning()


class LIDCClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name,
        input_dim,
        task="binary",
        num_classes=2,
        lr=1e-4,
        weight_decay=1e-4,
        class_weights=None,
        pretrained=False,
        in_channels=None,
        scheduler="cosine",
        max_epochs=50,
        dropout=0.2,
        gradient_clip_val=0.0,
        attention="none",
        fusion="none",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_lidc_lightning_model(
            model_name=model_name,
            input_dim=input_dim,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=in_channels,
            dropout=dropout,
            attention=attention,
            fusion=fusion,
        )
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights if class_weights is not None else torch.empty(0))
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights if self.class_weights.numel() else None)
        self._epoch_outputs = {"train": [], "val": [], "test": []}
        self._logged_static_hparams = False

    def forward(self, x):
        return self.model(x)

    @property
    def parameter_count(self):
        return count_parameters(self.model)

    def _step(self, batch, stage):
        images = batch["image"]
        labels = batch["label"]
        logits = self(images)
        loss = self.criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        accuracy = (predictions == labels).float().mean()

        batch_size = labels.size(0)
        if stage == "train":
            self._last_train_batch_size = int(batch_size)
        self.log("{}_loss".format(stage), loss, on_step=stage == "train", on_epoch=True, prog_bar=True, batch_size=batch_size)
        if stage == "train":
            self.log("train_accuracy_step", accuracy, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch_size)

        self._epoch_outputs[stage].append({
            "labels": labels.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
            "loss": loss.detach().cpu(),
        })
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_train_epoch_start(self):
        self._epoch_outputs["train"] = []
        self._last_train_batch_size = 1
        if not self._logged_static_hparams:
            self.log("hparam_dropout", float(self.hparams.dropout), on_epoch=True, prog_bar=False, batch_size=1)
            self.log("hparam_gradient_clip_val", float(self.hparams.gradient_clip_val), on_epoch=True, prog_bar=False, batch_size=1)
            self.log("hparam_learning_rate", float(self.hparams.lr), on_epoch=True, prog_bar=False, batch_size=1)
            self.log("hparam_weight_decay", float(self.hparams.weight_decay), on_epoch=True, prog_bar=False, batch_size=1)
            self._logged_static_hparams = True

    def on_validation_epoch_start(self):
        self._epoch_outputs["val"] = []

    def on_test_epoch_start(self):
        self._epoch_outputs["test"] = []

    def _epoch_end(self, stage):
        outputs = self._epoch_outputs[stage]
        if not outputs:
            return
        labels = torch.cat([item["labels"] for item in outputs]).numpy()
        probabilities = torch.cat([item["probabilities"] for item in outputs]).numpy()
        if self.hparams.task == "binary":
            metrics = binary_metrics(labels, probabilities[:, 1])
        else:
            metrics = multiclass_metrics(labels, probabilities)
        epoch_batch_size = int(len(labels))
        for key, value in metrics.items():
            if value == "":
                if key == "auc_roc":
                    value = 0.5
                else:
                    continue
            if key in ("tp", "tn", "fp", "fn", "sample_count"):
                continue
            self.log(
                "{}_{}".format(stage, key),
                float(value),
                on_epoch=True,
                prog_bar=key in ("accuracy", "auc_roc", "f1"),
                batch_size=epoch_batch_size,
            )

        label_counts = Counter(labels.tolist())
        for label, count in sorted(label_counts.items()):
            self.log("{}_label_{}_count".format(stage, label), float(count), on_epoch=True, prog_bar=False, batch_size=epoch_batch_size)

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def collect_epoch_outputs(self, stage="test"):
        outputs = self._epoch_outputs.get(stage, [])
        if not outputs:
            return [], []
        labels = torch.cat([item["labels"] for item in outputs]).numpy()
        probabilities = torch.cat([item["probabilities"] for item in outputs]).numpy()
        return labels, probabilities

    def on_before_optimizer_step(self, optimizer):
        grad_abs_sum = 0.0
        grad_sq_sum = 0.0
        grad_count = 0
        grad_max = 0.0
        param_sq_sum = 0.0

        for parameter in self.parameters():
            param_sq_sum += float(parameter.detach().pow(2).sum().cpu())
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            grad_abs_sum += float(grad.abs().sum().cpu())
            grad_sq_sum += float(grad.pow(2).sum().cpu())
            grad_count += grad.numel()
            grad_max = max(grad_max, float(grad.abs().max().cpu()))

        avg_abs_grad = grad_abs_sum / float(max(grad_count, 1))
        grad_norm_l2 = grad_sq_sum ** 0.5
        param_norm_l2 = param_sq_sum ** 0.5
        batch_size = int(getattr(self, "_last_train_batch_size", 1))
        self.log("train_avg_abs_gradient", avg_abs_grad, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train_gradient_l2_norm", grad_norm_l2, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train_max_abs_gradient", grad_max, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train_parameter_l2_norm", param_norm_l2, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        if self.hparams.scheduler == "none":
            return optimizer
        if self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_auc_roc" if self.hparams.task == "binary" else "val_f1",
                    "interval": "epoch",
                },
            }

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(self.hparams.max_epochs), 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
