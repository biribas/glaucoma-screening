from typing import override

from lightning.pytorch.callbacks import EarlyStopping


class BackboneEarlyStopping(EarlyStopping):
    def __init__(
        self,
        *,
        unfreeze_backbone_at_epoch: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch

    @override
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.unfreeze_backbone_at_epoch:
            super().on_train_epoch_end(trainer, pl_module)

    @override
    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch >= self.unfreeze_backbone_at_epoch:
            return super().on_validation_end(trainer, pl_module)
