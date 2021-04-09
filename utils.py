import os
import random
from typing import *

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as T
from fastcore.all import L, Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.utils import make_grid


def folder2df(
    directory: Union[str, Path],
    extensions: list = IMG_EXTENSIONS,
    shuffle: bool = False,
    seed: int = 42,
    exclude: Union[List, L] = [],
):
    """
    Parses all the Images in `directory` and puts them in a `DataFrame` object.
    """

    random.seed(seed)

    image_list = L()
    target_list = L()

    if not isinstance(directory, Path):
        directory = Path(directory)

    for label in directory.ls():
        cls_label = str(label).split(os.path.sep)[-1]
        if cls_label not in exclude:
            # grab if the label is not in exclude list
            label = Path(label)
            if os.path.isdir(label):
                for img in label.ls():
                    if str(img).lower().endswith(extensions):
                        image_list.append(img)
                        target_list.append(str(label).split(os.path.sep)[-1])

    dataframe: pd.DataFrame = pd.DataFrame()
    dataframe["image_id"] = image_list.map(str)
    dataframe["target"] = target_list
    if shuffle:
        dataframe = dataframe.sample(frac=1, random_state=seed).reset_index(
            inplace=False, drop=True
        )
    return dataframe


def plot_images(dls: DataLoader, mapping=None, n: int = 8):
    "plot images from a dataloader"
    ims, lbls = next(iter(dls))
    ims, lbls = ims[:n], lbls[:n]
    grid = make_grid(ims, normalize=True).permute(1, 2, 0)
    fig = plt.figure(figsize=(13, 13))
    plt.imshow(grid, cmap="binary")
    if mapping is not None:
        plt.title([mapping[o] for o in lbls.data.cpu().numpy()])
    else:
        plt.title([lbls.data.cpu().numpy()])
    plt.axis("off")
    plt.pause(0.05)


class DatasetFromPandas(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: T.Compose):
        self._dataframe = df
        self._transforms = transforms

    @property
    def transforms(self):
        return self._transforms

    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self, idx):
        im = self._dataframe["image_id"][idx]

        # Load and apply transformations to the Image
        im = Image.open(im)
        im = self._transforms(im)
        im = im / 255.0

        lbl = self._dataframe["cat_label"][idx]
        return im, lbl


class ToFloat(object):
    """
    Rescale the values between 0, 1.
    """

    def __init__(self, max_value=None) -> None:
        self.max_value = max_value

    def __call__(self, img):
        return img.type(torch.float32) / self.max_value

    def __repr__(self):
        return self.__class__.__name__ + "()"
