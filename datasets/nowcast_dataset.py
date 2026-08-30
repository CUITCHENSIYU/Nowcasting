import os
import re
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.registry import register_module


def rain_to_bins(rain: np.ndarray, num_bins: int = 512, bin_size: float = 0.2) -> np.ndarray:
    bins = np.floor(rain / bin_size).astype(np.int64)
    return np.clip(bins, 0, num_bins - 1)


def center_crop(arr: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return arr[..., top : top + crop_h, left : left + crop_w]

def prepare_sample(
    history: np.ndarray,
    future: np.ndarray,
    input_size: int,
    num_bins: int,
    bin_size: float,
):
    history = center_crop(history, input_size, input_size)
    out_size = input_size // 4
    future = center_crop(future, out_size, out_size)

    if history.ndim == 3:
        history = history[:, None, :, :]
    target_bins = rain_to_bins(future, num_bins=num_bins, bin_size=bin_size)
    return history.astype(np.float32), target_bins.astype(np.int64)


def collate_nowcast_batch(batch):
    inputs = torch.stack([item["inputs"] for item in batch], dim=0)
    targets = torch.stack([item["targets"] for item in batch], dim=0)
    return {"inputs": inputs, "targets": targets}


@register_module(parent="datasets")
class NimrodDataset(Dataset):
    """Radar sequence dataset that reads per-sample .h5 files from a folder.

    Each file is expected to contain dataset ``images`` with shape [T, H, W], where
    T = seq_len + forecast_steps. Sequence lengths are inferred from a parent
    directory name like ``seq_12_out-seq_6_threshold_20``.
    """

    def __init__(
        self,
        path: str,
        seq_len: int,
        forecast_steps: int,
        input_size: int = 256,
        num_bins: int = 512,
        bin_size: float = 0.2,
        scale: float = 47.83,
        **kwargs,
    ):
        if not os.path.isdir(path):
            raise NotADirectoryError(f"dataset path must be a folder of .h5 files, got: {path}")

        self.path = path
        self.input_size = input_size
        self.num_bins = num_bins
        self.bin_size = bin_size
        self.seq_len = seq_len
        self.scale = scale
        self.forecast_steps = forecast_steps

        self.files = sorted(glob(os.path.join(path, "*.h5")))
        if not self.files:
            raise FileNotFoundError(f"no .h5 files found under: {path}")

        with h5py.File(self.files[0], "r") as f:
            total_frames = f["images"].shape[0]
        expected = self.seq_len + self.forecast_steps
        if total_frames != expected:
            raise ValueError(
                f"h5 frames={total_frames} != seq_len({self.seq_len})+"
                f"forecast_steps({self.forecast_steps})={expected}"
            )

    def __getitem__(self, index):
        with h5py.File(self.files[index], "r") as f:
            radar = np.asarray(f["images"], dtype=np.float32)
        radar = np.squeeze(radar) * self.scale
        history = radar[: self.seq_len]
        future = radar[self.seq_len : self.seq_len + self.forecast_steps]
        history, target_bins = prepare_sample(
            history,
            future,
            self.input_size,
            self.num_bins,
            self.bin_size,
        )
        return {
            "inputs": torch.from_numpy(history),
            "targets": torch.from_numpy(target_bins),
        }

    def __len__(self):
        return len(self.files)

    collate_fn = staticmethod(collate_nowcast_batch)