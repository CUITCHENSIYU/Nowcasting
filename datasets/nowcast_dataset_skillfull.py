"""Radar sequence dataset for Skillful Nowcasting / NowcastNet.

Same continuous-frame format as PredRNN: [T, H, W, C].
Image size should be divisible by 32 (noise projector requirement).
"""

from __future__ import annotations

import os
from glob import glob
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.nowcast_dataset import center_crop
from utils.registry import register_module


def collate_skillfull_batch(batch):
    frames = torch.stack([item["frames"] for item in batch], dim=0)
    return {"frames": frames}


@register_module(parent="datasets")
class NimrodSkillfullDataset(Dataset):
    """Per-sample .h5 radar sequences for Skillful Nowcasting."""

    def __init__(
        self,
        path: str,
        seq_len: int,
        forecast_steps: int,
        input_size: int = 256,
        scale: float = 47.83,
        norm: Optional[float] = 47.83,
        img_channel: int = 1,
        **kwargs,
    ):
        if not os.path.isdir(path):
            raise NotADirectoryError(
                f"dataset path must be a folder of .h5 files, got: {path}"
            )
        if input_size % 32 != 0:
            raise ValueError(f"input_size must be divisible by 32, got {input_size}")

        self.path = path
        self.seq_len = seq_len
        self.forecast_steps = forecast_steps
        self.input_size = input_size
        self.scale = scale
        self.norm = norm
        self.img_channel = img_channel
        self.total_length = seq_len + forecast_steps

        self.files = sorted(glob(os.path.join(path, "*.h5")))
        if not self.files:
            raise FileNotFoundError(f"no .h5 files found under: {path}")

        with h5py.File(self.files[0], "r") as f:
            total_frames = f["images"].shape[0]
        if total_frames != self.total_length:
            raise ValueError(
                f"h5 frames={total_frames} != seq_len({seq_len})+"
                f"forecast_steps({forecast_steps})={self.total_length}"
            )

    def __getitem__(self, index):
        with h5py.File(self.files[index], "r") as f:
            radar = np.asarray(f["images"], dtype=np.float32)
        radar = np.squeeze(radar) * self.scale
        if radar.ndim != 3:
            raise ValueError(f"expected [T, H, W], got {radar.shape}")

        radar = center_crop(radar, self.input_size, self.input_size)
        if self.norm is not None and self.norm > 0:
            radar = radar / self.norm

        if radar.ndim == 3:
            radar = radar[..., None]
        if radar.shape[-1] != self.img_channel:
            raise ValueError(
                f"img_channel={self.img_channel} but data has C={radar.shape[-1]}"
            )

        return {"frames": torch.from_numpy(radar.astype(np.float32))}

    def __len__(self):
        return len(self.files)

    collate_fn = staticmethod(collate_skillfull_batch)
