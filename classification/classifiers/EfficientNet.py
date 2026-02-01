from typing import Literal

import torchvision.models as models
from torch import nn

from classifiers import BaseClassifier


class EfficientNetClassifier(BaseClassifier):
    models = {
        "b0": models.efficientnet_b0,
        "b1": models.efficientnet_b1,
        "b2": models.efficientnet_b2,
        "b3": models.efficientnet_b3,
        "b4": models.efficientnet_b4,
        "b5": models.efficientnet_b5,
        "b6": models.efficientnet_b6,
        "b7": models.efficientnet_b7,
    }

    def __init__(
        self,
        version: Literal["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"] = "b4",
        **kwargs,
    ):
        if version not in self.models:
            raise ValueError(f"Invalid EfficientNet version: {version}")

        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.backbone = self.models[version](weights="DEFAULT")
        self._init_head(self.backbone.classifier[-1].in_features)  # type: ignore
        self.backbone.classifier[-1] = nn.Identity()
        self.gradcam_layer = self.backbone.features[-1]
