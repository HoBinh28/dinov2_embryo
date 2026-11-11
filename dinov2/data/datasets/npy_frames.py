# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Unlabeled dataset of grayscale frames stored in stacked .npy files.

Each .npy file holds an array of shape (T, H, W, 1) or (T, H, W), dtype uint8
(e.g. 11 time-lapse frames of an embryo). Every selected frame of every file is
one sample, returned as an RGB PIL image (gray replicated on 3 channels) so that
the standard DINOv2 augmentations / ImageNet normalization / pretrained weights apply.

The list of files is read from a JSON file, which may contain either:
    - a list of paths:                 ["a/b/0001.npy", ...]
    - a dict of lists (e.g. splits):   {"train": [...], "val": [...]}
      -> use `split="train"` to select one key, or `split=None` to concatenate all lists.
"""

import json
import logging
import os
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset


logger = logging.getLogger("dinov2")

DEFAULT_FRAME_INDICES = (1, 2, 3, 4, 5, 6, 7, 8)  # drop frames 0, 9, 10


def _load_path_list(json_path: str, split: Optional[str]) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if isinstance(content, list):
        if split is not None:
            logger.warning(f'json "{json_path}" is a plain list, ignoring split="{split}"')
        paths = content
    elif isinstance(content, dict):
        if split is not None:
            if split not in content:
                raise KeyError(f'split "{split}" not found in {json_path}; available keys: {list(content.keys())}')
            paths = content[split]
        else:
            paths = [p for v in content.values() if isinstance(v, list) for p in v]
    else:
        raise ValueError(f"unsupported json content type {type(content)} in {json_path}")

    paths = [str(p) for p in paths]
    if not all(p.endswith(".npy") for p in paths):
        bad = [p for p in paths if not p.endswith(".npy")][:5]
        raise ValueError(f"json must only contain .npy paths, got e.g. {bad}")
    return paths


class NpyFramesDataset(VisionDataset):
    def __init__(
        self,
        *,
        root: str,
        split: Optional[str] = None,
        data_root: str = "",
        frame_indices: Sequence[int] = DEFAULT_FRAME_INDICES,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extra: Optional[str] = None,
    ) -> None:
        """
        Args:
            root: path to the JSON file listing the .npy files.
            split: optional key when the JSON is a dict of lists.
            data_root: prefix prepended to relative paths in the JSON ("" = relative to the CWD).
            frame_indices: which frames of each .npy are used as samples.
            extra: alias of `data_root`, so it can be given in a dataset string
                   (e.g. "NpyFrames:root=/x/list.json:extra=/mnt").
        """
        super().__init__(root, transforms, transform, target_transform)
        if extra and not data_root:
            data_root = extra
        self.json_path = root
        self.data_root = data_root
        self.frame_indices = tuple(int(i) for i in frame_indices)
        if len(self.frame_indices) == 0:
            raise ValueError("frame_indices must not be empty")

        rel_paths = _load_path_list(root, split)
        self.files = [p if (os.path.isabs(p) or not data_root) else os.path.join(data_root, p) for p in rel_paths]
        if len(self.files) == 0:
            raise RuntimeError(f"no .npy file listed in {root}")

        logger.info(
            f"NpyFramesDataset: {len(self.files):,d} files x {len(self.frame_indices)} frames "
            f"= {len(self):,d} samples (frames {self.frame_indices})"
        )

    def __len__(self) -> int:
        return len(self.files) * len(self.frame_indices)

    def _locate(self, index: int) -> Tuple[str, int]:
        n_frames = len(self.frame_indices)
        return self.files[index // n_frames], self.frame_indices[index % n_frames]

    def load_frame(self, index: int) -> np.ndarray:
        """Returns the raw (H, W) uint8 frame for a sample index."""
        path, frame_idx = self._locate(index)
        arr = np.load(path, mmap_mode="r")  # only the requested frame is read from disk
        if arr.ndim == 4:
            if arr.shape[-1] != 1:
                raise ValueError(f"{path}: expected last dim 1, got shape {arr.shape}")
            frame = arr[frame_idx, :, :, 0]
        elif arr.ndim == 3:
            frame = arr[frame_idx]
        else:
            raise ValueError(f"{path}: expected shape (T,H,W,1) or (T,H,W), got {arr.shape}")
        frame = np.array(frame)  # copy out of the memmap
        del arr
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def get_image(self, index: int) -> Image.Image:
        return Image.fromarray(self.load_frame(index)).convert("RGB")  # 2D uint8 -> "L" -> RGB

    def get_target(self, index: int) -> int:
        return 0  # unlabeled

    def get_targets(self) -> np.ndarray:
        return np.zeros(len(self), dtype=np.int64)

    def __getitem__(self, index: int):
        try:
            image = self.get_image(index)
        except Exception as e:
            path, frame_idx = self._locate(index)
            raise RuntimeError(f"can not read frame {frame_idx} of {path} (sample {index})") from e
        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
