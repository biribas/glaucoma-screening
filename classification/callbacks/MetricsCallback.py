from typing import Any, Literal, Optional, cast, override

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import (
    AUROC,
    ROC,
    ConfusionMatrix,
    HammingDistance,
    MetricCollection,
    Recall,
    SensitivityAtSpecificity,
    Specificity,
)


class MetricsCallback(L.Callback):
    def __init__(self, *, labels: list[str], ignore_index: Optional[int] = None):
        super().__init__()
        self.labels = labels

        self.binary_metrics = {
            "train": MetricCollection(
                {
                    "sens@spec": SensitivityAtSpecificity(
                        task="binary", min_specificity=0.95
                    ),
                    "confusion_matrix": ConfusionMatrix(task="binary"),
                    "AUROC": AUROC(task="binary"),
                    "sensitivity": Recall(task="binary"),
                    "specificity": Specificity(task="binary"),
                },
                prefix="train_",
            )
        }
        self.binary_metrics["val"] = self.binary_metrics["train"].clone(prefix="val_")
        self.binary_metrics["test"] = self.binary_metrics["train"].clone(prefix="test_")
        self.binary_metrics["test"].add_metrics({"ROC": ROC(task="binary")})

        kwargs = {"task": "multilabel", "num_labels": 10, "ignore_index": ignore_index}
        self.multilabel_metrics = {
            "train": MetricCollection(
                {
                    "hamming": HammingDistance(**kwargs),
                    "sensitivity": Recall(**kwargs, average=None),
                    "specificity": Specificity(**kwargs, average=None),
                },
                prefix="train_",
            )
        }
        self.multilabel_metrics["val"] = self.multilabel_metrics["train"].clone(
            prefix="val_"
        )
        self.multilabel_metrics["test"] = self.multilabel_metrics["train"].clone(
            prefix="test_"
        )

    @override
    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        self.log = pl_module.log
        self.logger = pl_module.logger
        self.num_labels = pl_module.num_labels
        self.device = pl_module.device
        self.safe_print = cast(ProgressBar, trainer.progress_bar_callback).print
        return super().setup(trainer, pl_module, stage)

    @override
    def on_train_start(self, trainer, pl_module) -> None:
        self._move_metrics_to_device()
        return super().on_train_start(trainer, pl_module)

    @override
    def on_validation_start(self, trainer, pl_module) -> None:
        self._move_metrics_to_device()
        return super().on_validation_start(trainer, pl_module)

    @override
    def on_test_start(self, trainer, pl_module) -> None:
        self._move_metrics_to_device()
        return super().on_test_start(trainer, pl_module)

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        self._update_metrics(outputs, batch, "train")
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    @override
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._update_metrics(outputs, batch, "val")
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    @override
    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._update_metrics(outputs, batch, "test")
        return super().on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    @override
    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._compute_metrics("train", trainer.current_epoch)
        return super().on_train_epoch_end(trainer, pl_module)

    @override
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._compute_metrics("val", trainer.current_epoch)
        return super().on_validation_epoch_end(trainer, pl_module)

    @override
    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._compute_metrics("test", trainer.current_epoch)
        return super().on_test_epoch_end(trainer, pl_module)

    def _compute_metrics(
        self, stage: Literal["train", "val", "test"], epoch: int
    ) -> None:
        if self.num_labels != 10:
            metrics = self.binary_metrics[stage].compute()
            self._log_binary_metrics(stage, metrics, epoch)
            self.binary_metrics[stage].reset()

        if self.num_labels != 1:
            metrics = self.multilabel_metrics[stage].compute()
            self._log_multilabel_metrics(stage, metrics, epoch)
            self.multilabel_metrics[stage].reset()

    def _log_binary_metrics(
        self,
        stage: Literal["train", "val", "test"],
        metrics: dict,
        epoch: int,
    ):
        for k, v in metrics.items():
            if "confusion_matrix" in k:
                self.safe_print(f"{k} {epoch}:\n {v.cpu()}\n")
            elif "sens@spec" in k:
                sens_at_spec, threshold = v
                self.log(f"{stage}_sens@spec", sens_at_spec)
                self.log(f"{stage}_threshold", threshold)
                self.safe_print(f"{stage} {epoch} sens@spec: {sens_at_spec}")
                self.safe_print(f"{stage} {epoch} threshold: {threshold}\n")
            elif "test_ROC" in k:
                if not isinstance(self.logger, WandbLogger):
                    continue

                fpr, tpr, thresholds = v
                fpr = fpr.cpu()
                tpr = tpr.cpu()
                thresholds = thresholds.cpu()

                table = wandb.Table(columns=["FPR", "TPR", "Threshold"])
                for i in range(len(fpr)):
                    table.add_data(fpr[i], tpr[i], thresholds[i])

                self.logger.experiment.log(
                    {
                        "ROC Curve": wandb.plot.line(
                            table, x="FPR", y="TPR", title=f"{stage} ROC Curve"
                        )
                    }
                )
            elif type(v) is torch.Tensor and v.ndim == 0:
                self.log(k, v)
                self.safe_print(f"{k} {epoch}: {v}\n")

    def _log_multilabel_metrics(
        self,
        stage: Literal["train", "val", "test"],
        metrics: dict,
        epoch: int,
    ):
        for k, v in metrics.items():
            if "hamming" in k:
                self.log(k, v)
                self.safe_print(f"{k} {epoch}: {v}\n")
            else:
                for i, label in enumerate(self.labels):
                    self.safe_print(f"{k}_{label} {epoch}: {v[i]}")
                    if stage == "test":
                        self.log(f"{k}_{label}", v[i])
                self.safe_print("")

    def _update_metrics(
        self, outputs, batch, stage: Literal["train", "val", "test"]
    ) -> None:
        if outputs is None:
            return

        preds: torch.Tensor = outputs["preds"]
        targets: torch.Tensor = batch["labels"]

        if self.num_labels == 1:
            return self.binary_metrics[stage].update(preds, targets.int())

        if self.num_labels == 10:
            return self.multilabel_metrics[stage].update(preds, targets)

    def _move_metrics_to_device(self) -> None:
        for v in self.binary_metrics.values():
            v.to(self.device)
        for v in self.multilabel_metrics.values():
            v.to(self.device)
