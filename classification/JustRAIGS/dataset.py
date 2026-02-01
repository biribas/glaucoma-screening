import os
from typing import Literal, Optional

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset, Subset


class JustRAIGS_Dataset(Dataset):
    def __init__(
        self,
        *,
        labels_file: str,
        data_dir: str,
        num_labels: Literal[1, 10, 11],
        image_size: Optional[int] = None,
    ):
        self.img_dir: str = data_dir
        self.num_labels: int = num_labels
        self.image_size = image_size
        self.IGNORE_INDEX: Optional[int] = None

        self.df: DataFrame = (
            pd.read_csv(
                labels_file,
                sep=";",
                converters={"Final Label": lambda x: 1 if x == "RG" else 0},
            )
            .rename(columns={"Final Label": "label"})
            .pipe(self._resolve_columns)
            .pipe(lambda df: df[df["path"].apply(os.path.exists)])
            .reset_index(drop=True)
        )
        self.labels = self.df.columns.drop(["path", "id"]).to_list()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        return {
            "id": row["id"],
            "image": plt.imread(row["path"]),
            "labels": torch.tensor(
                [row[label] for label in self.labels], dtype=torch.float
            ),
        }

    def _resolve_columns(self, df) -> pd.DataFrame:
        if self.num_labels == 10:
            df = df[df["label"] == 1].reset_index(drop=True)

        out = pd.DataFrame(index=df.index)

        out["id"] = df["Eye ID"]
        out["path"] = df["Eye ID"].apply(
            lambda x: os.path.join(self.img_dir, f"{x}.JPG")
        )

        if self.num_labels != 10:
            out["label"] = df["label"]

        if self.num_labels == 1:
            return out

        self.IGNORE_INDEX = -1

        for feat in [
            "ANRS",
            "ANRI",
            "RNFLDS",
            "RNFLDI",
            "BCLVS",
            "BCLVI",
            "NVT",
            "DH",
            "LD",
            "LC",
        ]:
            g1 = df[f"G1 {feat}"]
            g2 = df[f"G2 {feat}"]
            g3 = df[f"G3 {feat}"]

            out[feat] = np.where(
                df["label"] == 0,  # if non-glaucoma sample
                self.IGNORE_INDEX,  # features are not labeled
                np.where(  # else
                    g3.notna(),  # if g3 is provided
                    g3,  # use g3
                    np.where(  # else
                        g1.eq(g2),  # if g1 and g2 agree
                        g1,  # use agreement value
                        self.IGNORE_INDEX,  # else, no g3 and no agreement
                    ),
                ),
            )

        return out


class JustRAIGS_Subset(Dataset):
    def __init__(
        self,
        subset: Subset[JustRAIGS_Dataset],
        transform: A.Compose,
    ):
        self.subset = subset
        self.indices = subset.indices
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        assert isinstance(sample, dict)

        np_image = sample["image"]
        sample["image"] = self.transform(image=np_image)["image"]
        return sample
