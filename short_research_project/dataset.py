import pickle
from collections.abc import Callable

import numpy as np
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset


def _unpickle(path: str) -> dict:
    logger.info(f"File loaded: {path}")
    with open(path, "rb") as fo:
        loaded_dict = pickle.load(fo, encoding="bytes")
    logger.info(f"Loaded dict batch label: {loaded_dict[b'batch_label']}")

    return loaded_dict


class CIFAR10Dataset(Dataset):
    def __init__(
        self, paths, n_images: int | None = None, transform: Callable | None = None
    ):
        self.labels = []
        self.images = None
        self.transform = transform

        for path in paths:
            data_batch = _unpickle(path)

            self.labels += data_batch[b"labels"]
            if self.images is None:
                self.images = data_batch[b"data"]
            else:
                self.images = np.concat([self.images, data_batch[b"data"]])

        if self.images is None:
            raise ValueError("images array is None.")

        if n_images is not None:
            self.labels = self.labels[0:n_images]
            self.images = self.images[0:n_images]

        logger.info("Dataset info:")
        logger.info(f"\tShape: {self.images.shape}")
        logger.info(f"\tSize: {self.images.nbytes / 10e6} MB")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(
                Image.fromarray(image.reshape(3, 32, 32).transpose(1, 2, 0))
            )
        return image, label
