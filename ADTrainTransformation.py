import random

import numpy as np
import torch


class ADTrainTransformation:
    """Rotate by one of the random angles."""

    def __init__(self):
        # 90, 180, 270 degree
        self.angles = [1, 2, 3]

    def __call__(self, data):
        images = data["images"]
        label = data["label"]

        # Rotation
        angle = random.choice(self.angles)
        images = np.rot90(images, angle, axes=[1, 2]).copy()

        # Flip (mirroring)
        if random.random() < 0.5:
            images = np.flip(images, axis=2).copy()

        # images = np.expand_dims(images, axis=1)
        # images = images.astype(np.float32)

        return {
            'images': images,
            'label': label
        }


class ToTensor():
    def __call__(self, data):
        images = data["images"]
        label = data["label"]

        images = np.expand_dims(images, axis=1)
        images = images.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'images': torch.from_numpy(images),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample
