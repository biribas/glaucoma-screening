from argparse import ArgumentParser
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from callbacks import BackboneEarlyStopping, CustomFinetuning, MetricsCallback
from classifiers import (
    BaseClassifier,
    ConvNeXtClassifier,
    EfficientNetClassifier,
    ResNetClassifier,
    SwinTransformerClassifier,
)
from JustRAIGS import JustRAIGS_DataModule, JustRAIGS_Dataset

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("medium")

    parser = ArgumentParser()
    parser.add_argument(
        "--data", help="""Path to the data directory.""", type=Path, required=True
    )
    parser.add_argument(
        "--labels", help="""Path to the labels file.""", type=Path, required=True
    )
    parser.add_argument(
        "-m",
        "--model",
        help="""Choose one of the predefined models provided by torchvision.""",
        choices=["resnet", "efficientnet", "convnext", "swin"],
        type=str,
    )
    parser.add_argument(
        "-mv",
        "--model_version",
        help="""Version of the model to use.""",
        type=str,
    )
    parser.add_argument(
        "-nl",
        "--num_labels",
        help=(
            "Number of labels for the model output. "
            "Available options:\n"
            "  • 1  – Binary glaucoma classification\n"
            "  • 10 – Multilabel classification of glaucomatous features\n"
        ),
        choices=[1, 10],
        type=int,
        required=True,
    )
    parser.add_argument(
        "-ne",
        "--num_epochs",
        help="""Number of Epochs to Run.""",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-ue",
        "--unfreeze_epoch",
        help="""Epoch at which to unfreeze the backbone.""",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Adjust learning rate of optimizer.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-g",
        "--gamma",
        help="Adjust gamma of focal loss.",
        type=float,
        default=2,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="""Manually determine batch size. Defaults to 16.""",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-is",
        "--image_size",
        help="Side length of the input image after resizing. If not provided, the original image resolution is used.",
        type=int,
    )
    parser.add_argument(
        "--logging",
        help="Enable logging with Wandb.",
        action="store_true",
    )
    args = parser.parse_args()

    dataset = JustRAIGS_Dataset(
        labels_file=args.labels,
        data_dir=args.data,
        num_labels=args.num_labels,
        image_size=args.image_size,
    )
    datamodule = JustRAIGS_DataModule(
        dataset=dataset,
        batch_size=args.batch_size,
    )

    model_classes: dict[str, type[BaseClassifier]] = {
        "resnet": ResNetClassifier,
        "efficientnet": EfficientNetClassifier,
        "convnext": ConvNeXtClassifier,
        "swin": SwinTransformerClassifier,
    }

    try:
        ModelClass = model_classes[args.model]
    except KeyError:
        raise ValueError(f"Unknown model: {args.model}")

    model_version = args.model_version
    if model_version.isdigit():
        model_version = int(model_version)

    model = ModelClass(
        num_labels=args.num_labels,
        version=model_version,
        lr=args.learning_rate,
        gamma=args.gamma,
        pos_weight=datamodule.pos_weights,
        ignore_index=dataset.IGNORE_INDEX,
    )

    if args.num_labels == 1:
        monitor = "val_sens@spec"
        mode = "max"
        min_delta = 0.005
        patience = 10
    else:
        monitor = "val_hamming"
        mode = "min"
        min_delta = 5e-4
        patience = 10

    trainer_args = {
        "max_epochs": args.num_epochs,
        "enable_model_summary": False,
        "precision": "bf16-mixed",
        "num_sanity_val_steps": 0,
        "logger": WandbLogger(save_dir=".wandb") if args.logging else None,
        "callbacks": [
            MetricsCallback(
                labels=dataset.labels,
                ignore_index=dataset.IGNORE_INDEX,
            ),
            ModelCheckpoint(
                dirpath=".checkpoints/Tipo 3",
                filename=f"tipo3-{args.model}-{model_version}-{{epoch}}-{{{monitor}:.3f}}",
                save_top_k=3,
                monitor=monitor,
                mode=mode,
            ),
            BackboneEarlyStopping(
                unfreeze_backbone_at_epoch=args.unfreeze_epoch,
                check_on_train_epoch_end=False,
                patience=patience,
                min_delta=min_delta,
                monitor=monitor,
                mode=mode,
            ),
            CustomFinetuning(
                unfreeze_backbone_at_epoch=args.unfreeze_epoch,
                head_initial_ratio_lr=0.1,
                backbone_initial_ratio_lr=0.1,
                verbose=True,
                lambda_func=lambda _: 1,
            ),
        ],
    }
    trainer = L.Trainer(**trainer_args)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)
