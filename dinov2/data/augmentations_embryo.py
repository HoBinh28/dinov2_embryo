# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Multi-crop augmentation for grayscale embryo microscopy (gray replicated to RGB).

Differences with DataAugmentationDINO:
  - orientation invariance: vertical flip + random 90-degree rotations (+ optional free rotation)
  - photometric: brightness / contrast only (saturation, hue, grayscale are no-ops on gray images)
  - mild Gaussian blur with configurable sigma, applied with the probability actually written
    (dinov2.data.transforms.GaussianBlur applies blur with probability 1 - p)
  - solarization disabled by default
Output format is identical to DataAugmentationDINO (dict with global/local crops).
"""

import logging
import random
from typing import Sequence

from PIL import Image
from torchvision import transforms

from .transforms import make_normalize_transform


logger = logging.getLogger("dinov2")


class RandomRot90:
    """Rotates a PIL image by 0/90/180/270 degrees uniformly at random (lossless)."""

    _OPS = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)

    def __call__(self, img):
        op = random.choice(self._OPS)
        return img if op is None else img.transpose(op)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def _maybe(transform, p: float):
    """Applies `transform` with probability p (p <= 0: never, p >= 1: always)."""
    if p <= 0:
        return None
    if p >= 1:
        return transform
    return transforms.RandomApply([transform], p=p)


def _compose(*ts):
    return transforms.Compose([t for t in ts if t is not None])


class DataAugmentationEmbryo:
    def __init__(
        self,
        global_crops_scale: Sequence[float],
        local_crops_scale: Sequence[float],
        local_crops_number: int,
        global_crops_size: int = 224,
        local_crops_size: int = 98,
        *,
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rot90: bool = True,
        rotation_degrees: float = 0.0,
        rotation_fill: int = 0,
        color_jitter_p: float = 0.8,
        brightness: float = 0.4,
        contrast: float = 0.4,
        blur_sigma: Sequence[float] = (0.1, 1.0),
        blur_kernel_size: int = 9,
        blur_p_global1: float = 0.5,
        blur_p_global2: float = 0.2,
        blur_p_local: float = 0.3,
        solarize_p_global2: float = 0.0,
        solarize_threshold: int = 128,
    ):
        self.local_crops_number = local_crops_number
        self.params = dict(
            global_crops_scale=tuple(global_crops_scale),
            local_crops_scale=tuple(local_crops_scale),
            local_crops_number=local_crops_number,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            hflip_p=hflip_p,
            vflip_p=vflip_p,
            rot90=rot90,
            rotation_degrees=rotation_degrees,
            color_jitter_p=color_jitter_p,
            brightness=brightness,
            contrast=contrast,
            blur_sigma=tuple(blur_sigma),
            blur_p_global1=blur_p_global1,
            blur_p_global2=blur_p_global2,
            blur_p_local=blur_p_local,
            solarize_p_global2=solarize_p_global2,
        )
        logger.info("###################################")
        logger.info("Using embryo data augmentation parameters:")
        for k, v in self.params.items():
            logger.info(f"{k}: {v}")
        logger.info("###################################")

        # orientation (applied to the full image before cropping, so free rotation corners
        # are mostly cropped out by the random resized crops)
        orientation = _compose(
            RandomRot90() if rot90 else None,
            transforms.RandomRotation(
                rotation_degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill=rotation_fill
            )
            if rotation_degrees > 0
            else None,
        )

        def geometric(size, scale):
            return _compose(
                orientation,
                transforms.RandomResizedCrop(size, scale=tuple(scale), interpolation=transforms.InterpolationMode.BICUBIC),
                _maybe(transforms.RandomHorizontalFlip(p=1.0), hflip_p),
                _maybe(transforms.RandomVerticalFlip(p=1.0), vflip_p),
            )

        self.geometric_augmentation_global = geometric(global_crops_size, global_crops_scale)
        self.geometric_augmentation_local = geometric(local_crops_size, local_crops_scale)

        photometric = _maybe(transforms.ColorJitter(brightness=brightness, contrast=contrast), color_jitter_p)

        def blur(p):
            return _maybe(transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=tuple(blur_sigma)), p)

        self.normalize = transforms.Compose([transforms.ToTensor(), make_normalize_transform()])

        self.global_transfo1 = _compose(photometric, blur(blur_p_global1), self.normalize)
        self.global_transfo2 = _compose(
            photometric,
            blur(blur_p_global2),
            _maybe(transforms.RandomSolarize(threshold=solarize_threshold, p=1.0), solarize_p_global2),
            self.normalize,
        )
        self.local_transfo = _compose(photometric, blur(blur_p_local), self.normalize)

    def __call__(self, image):
        output = {}

        global_crop_1 = self.global_transfo1(self.geometric_augmentation_global(image))
        global_crop_2 = self.global_transfo2(self.geometric_augmentation_global(image))
        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        output["local_crops"] = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["offsets"] = ()
        return output
