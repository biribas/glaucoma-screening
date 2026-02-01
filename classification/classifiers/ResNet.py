from typing import Literal

import torch
import torchvision.models as models

from classifiers import BaseClassifier


class ResNetClassifier(BaseClassifier):
    models = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(
        self,
        version: Literal[18, 34, 50, 101, 152] = 50,
        **kwargs,
    ):
        if version not in self.models:
            raise ValueError(f"Invalid ResNet version: {version}")

        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.backbone = self.models[version](weights="DEFAULT")
        self._init_head(self.backbone.fc.in_features)
        self.backbone.fc = torch.nn.Identity()  # type: ignore
        self.gradcam_layer = self.backbone.layer4[-1]
