from lightning.pytorch.callbacks.finetuning import BackboneFinetuning


class CustomFinetuning(BackboneFinetuning):
    def __init__(
        self,
        *,
        head_initial_ratio_lr: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.head_initial_ratio_lr = head_initial_ratio_lr

    def finetune_function(self, pl_module, epoch, optimizer):
        if epoch == self.unfreeze_backbone_at_epoch:
            optimizer.param_groups[0]["lr"] *= self.head_initial_ratio_lr

        super().finetune_function(pl_module, epoch, optimizer)
