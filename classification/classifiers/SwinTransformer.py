from typing import Literal, cast

import torchvision.models as models
from torch import nn

from classifiers import BaseClassifier


class SwinTransformerClassifier(BaseClassifier):
    models = {
        "t": models.swin_t,
        "s": models.swin_s,
        "b": models.swin_b,
        "v2_t": models.swin_v2_t,
        "v2_s": models.swin_v2_s,
        "v2_b": models.swin_v2_b,
    }

    def __init__(
        self,
        version: Literal["t", "s", "b", "v2_t", "v2_s", "v2_b"] = "v2_t",
        **kwargs,
    ):
        if version not in self.models:
            raise ValueError(f"Invalid SWin Transformer version: {version}")

        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.backbone = self.models[version](weights="DEFAULT")
        self._init_head(in_features=self.backbone.head.in_features)
        self.backbone.head = cast(nn.Linear, nn.Identity())
        self.gradcam_layer = self.backbone.features[-1]
