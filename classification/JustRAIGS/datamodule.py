import os
from typing import Literal, Sequence, cast, override

import albumentations as A
import lightning as L
import numpy as np
import torch
from pandas import Series
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    Subset,
    WeightedRandomSampler,
    random_split,
)

from JustRAIGS.dataset import JustRAIGS_Dataset, JustRAIGS_Subset


class JustRAIGS_DataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        dataset: JustRAIGS_Dataset,
        batch_size: int,
        num_workers: int | None = os.cpu_count(),
    ):
        super().__init__()
        self.dataset = dataset
        self.sample_weights = []
        self.batch_size = batch_size
        self.num_workers: int = num_workers if num_workers is not None else 0

        self.train, self.val, self.test = self._split_dataset()
        self.train_sampler = (
            RandomSampler(self.train)
            if self.dataset.num_labels == 10
            else WeightedRandomSampler(
                weights=self._get_sample_weights(), num_samples=len(self.train)
            )
        )

        self.pos_weights = self._get_pos_weights()

    @override
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
        )

    @override
    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @override
    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _get_pos_weights(self) -> torch.Tensor | None:
        if self.dataset.num_labels == 1:
            return None

        dist = self.class_distribution(stage="train")
        values = np.sqrt(
            [
                (len(self.train) - dist[label]) / dist[label]
                for label in self.dataset.labels
            ]
        )
        return torch.tensor(values, dtype=torch.float32)

    def _get_sample_weights(self):
        train_labels = Series(self.dataset.df["label"][self.train.indices])
        class_count = np.bincount(train_labels)
        return [float(1 / class_count[i]) for i in train_labels.values]

    def _split_dataset(
        self, lengths=[0.7, 0.2, 0.1]
    ) -> tuple[
        JustRAIGS_Subset,
        JustRAIGS_Subset,
        JustRAIGS_Subset,
    ]:
        self._setup_transforms()

        if self.dataset.num_labels == 10:
            train, val, test = random_split(
                self.dataset,
                lengths,
                generator=torch.Generator(),
            )
            return (
                JustRAIGS_Subset(train, self.train_transform),
                JustRAIGS_Subset(val, self.val_transform),
                JustRAIGS_Subset(test, self.val_transform),
            )

        train_size, val_size, test_size = lengths

        labels = Series(self.dataset.df["label"]).values
        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            random_state=42,
            test_size=test_size,
        )
        train_val_idx, test_idx = next(sss1.split(np.arange(len(self.dataset)), labels))

        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            random_state=42,
            test_size=val_size / (train_size + val_size),
        )
        tr, vl = next(sss2.split(train_val_idx, labels[train_val_idx]))

        train_idx = cast(Sequence[int], train_val_idx[tr])
        val_idx = cast(Sequence[int], train_val_idx[vl])
        test_idx = cast(Sequence[int], test_idx)

        train = Subset(self.dataset, train_idx)
        val = Subset(self.dataset, val_idx)
        test = Subset(self.dataset, test_idx)

        return (
            JustRAIGS_Subset(train, self.train_transform),
            JustRAIGS_Subset(val, self.val_transform),
            JustRAIGS_Subset(test, self.val_transform),
        )

    def class_distribution(
        self, stage: Literal["train", "val", "test"]
    ) -> dict[str, int]:
        subsets = {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }

        try:
            subset = subsets[stage]
        except KeyError:
            raise ValueError(f"Invalid stage: {stage}")

        labels = self.dataset.labels
        values = self.dataset.df[labels].iloc[subset.indices].values

        distribution = {}
        for label in labels:
            values = self.dataset.df[label].iloc[subset.indices].values
            distribution[label] = int((values == 1).sum())

        return distribution

    def _setup_transforms(self) -> None:
        resize = []
        if self.dataset.image_size is not None:
            resize.append(A.Resize(self.dataset.image_size, self.dataset.image_size))

        train_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0,
                contrast_limit=0.1,
                p=1,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=5,
                val_shift_limit=5,
                p=1,
            ),
        ]

        common_transforms = [
            A.Normalize(),
            A.pytorch.ToTensorV2(),
        ]

        self.train_transform = A.Compose(
            transforms=resize + train_transforms + common_transforms,
            seed=42,
        )

        self.val_transform = A.Compose(
            transforms=resize + common_transforms,
            seed=42,
        )
