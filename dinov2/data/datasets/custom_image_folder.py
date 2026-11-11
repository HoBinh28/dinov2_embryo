# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class CustomImageFolder(ExtendedVisionDataset):
    """
    Generic image-folder dataset.

    Dataset string examples:
        CustomImageFolder:root=/data/embryo                 -> uses /data/embryo
        CustomImageFolder:root=/data/embryo:split=train     -> uses /data/embryo/train

    Supported layouts (inside the selected directory):
        1) Labeled (ImageFolder style): one sub-directory per class
               class_a/img1.png, class_b/img2.jpg, ...
        2) Unlabeled: images directly in the directory (or nested arbitrarily
           with `extra=unlabeled`); every target is 0.

    Labels are only needed for evaluation (kNN / linear); SSL pretraining ignores them.
    `extra=unlabeled` forces unlabeled mode and scans all sub-directories recursively.
    """

    def __init__(
        self,
        *,
        root: str,
        split: Optional[str] = None,
        extra: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        data_dir = os.path.join(root, split) if split else root
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f'dataset directory not found: "{data_dir}"')

        force_unlabeled = extra is not None and extra.lower() == "unlabeled"
        self._samples, self.classes = self._scan(data_dir, force_unlabeled)
        if len(self._samples) == 0:
            raise RuntimeError(f'no images ({", ".join(IMAGE_EXTENSIONS)}) found in "{data_dir}"')
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        logger.info(f'CustomImageFolder: {len(self._samples):,d} images, {len(self.classes)} classes in "{data_dir}"')

    @staticmethod
    def _is_image(name: str) -> bool:
        return name.lower().endswith(IMAGE_EXTENSIONS)

    def _scan(self, data_dir: str, force_unlabeled: bool) -> Tuple[List[Tuple[str, int]], List[str]]:
        class_dirs = sorted(e.name for e in os.scandir(data_dir) if e.is_dir())
        top_images = sorted(e.path for e in os.scandir(data_dir) if e.is_file() and self._is_image(e.name))

        if force_unlabeled or not class_dirs:
            paths = []
            for dirpath, _, filenames in os.walk(data_dir):
                paths.extend(os.path.join(dirpath, f) for f in filenames if self._is_image(f))
            return [(p, 0) for p in sorted(paths)], ["unlabeled"]

        if top_images:
            logger.warning(f"ignoring {len(top_images)} images at top level of {data_dir} (class sub-directories found)")

        samples = []
        for idx, cls in enumerate(class_dirs):
            for dirpath, _, filenames in os.walk(os.path.join(data_dir, cls)):
                samples.extend((os.path.join(dirpath, f), idx) for f in sorted(filenames) if self._is_image(f))
        return samples, class_dirs

    def get_image_data(self, index: int) -> bytes:
        with open(self._samples[index][0], "rb") as f:
            return f.read()

    def get_target(self, index: int) -> int:
        return self._samples[index][1]

    def get_targets(self) -> np.ndarray:
        return np.array([t for _, t in self._samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._samples)
