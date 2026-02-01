from abc import ABC
from typing import Dict, Literal, Optional, override

import lightning as L
import torch
from torch import Tensor, nn
from torch.optim import AdamW


class BaseClassifier(L.LightningModule, ABC):
    def __init__(
        self,
        num_labels: Literal[1, 10, 11],
        lr: float = 1e-3,
        alpha: float = -1,
        gamma: float = 2.0,
        pos_weight: Tensor | None = None,
        ignore_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

        self.optimizer_cls = AdamW
        self.loss_fn = FocalLoss(
            gamma=self.gamma,
            alpha=self.alpha,
            pos_weight=self.pos_weight,
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        # Must be set by subclasses
        self.backbone: nn.Module
        self.head: nn.Module
        self.gradcam_layer: nn.Module

    @override
    def forward(self, inputs: Tensor) -> Tensor:
        features = self.backbone(inputs)
        return self.head(features)

    @override
    def configure_optimizers(self):
        return self.optimizer_cls(self.head.parameters(), lr=self.lr)

    @override
    def training_step(self, batch):
        return self._step(batch)

    @override
    def validation_step(self, batch):
        return self._step(batch)

    @override
    def test_step(self, batch):
        return self._step(batch)

    def _step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor] | None:
        logits = self(batch["image"])
        targets = batch["labels"]

        loss = self.loss_fn(logits, targets)

        if loss is None:
            return None

        return {"loss": loss, "preds": logits}

    def _init_head(self, in_features: int) -> None:
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.num_labels),
        )


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        """
        Args:
            alpha (float): Weighting factor in range [0, 1] to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            pos_weight (Tensor): Weights for positive examples.
                    Default: ``None``.
            ignore_index (int): Specifies a target value that is ignored and does
                    not contribute to the loss. Default: ``None``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        """
        if not (0 <= alpha <= 1) and alpha != -1:
            raise ValueError(
                f"Invalid alpha value: {alpha}. alpha must be in the range [0,1] or -1 for ignore."
            )

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor | None:
        """
        Args:
            logits (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor with the reduction option applied.
        """
        if self.ignore_index:
            mask = torch.ne(targets, self.ignore_index)
            if not mask.any():
                return None
            safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
            float_mask = mask.float()
            denom = float_mask.sum()
        else:
            safe_targets = targets
            float_mask = 1
            denom = targets.numel()

        bce: Tensor = self.bce(logits, safe_targets)
        probs = torch.sigmoid(logits)
        pt = probs * safe_targets + (1 - probs) * (1 - safe_targets)

        loss = (1 - pt).pow(self.gamma) * bce

        if self.alpha != -1:
            alpha_t = self.alpha * safe_targets + (1 - self.alpha) * (1 - safe_targets)
            loss *= alpha_t

        loss *= float_mask

        if self.reduction == "mean":
            return loss.sum() / denom
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
