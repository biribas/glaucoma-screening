from typing import Literal

import torchvision.models as models
from torch import nn

from classifiers import BaseClassifier


class ConvNeXtClassifier(BaseClassifier):
    models = {
        "t": models.convnext_tiny,
        "s": models.convnext_small,
        "b": models.convnext_base,
        "l": models.convnext_large,
    }

    def __init__(
        self,
        version: Literal["tiny", "small", "base", "large"] = "small",
        **kwargs,
    ):
        if version not in self.models:
            raise ValueError(f"Invalid ConvNext version: {version}")

        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.backbone = self.models[version](weights="DEFAULT")
        self._init_head(self.backbone.classifier[-1].in_features)  # type: ignore
        self.backbone.classifier[-1] = nn.Identity()
        self.gradcam_layer = self.backbone.features[-1]
