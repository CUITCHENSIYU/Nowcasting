import json

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


def prepare_metnet_sample(
    history: np.ndarray,
    future: np.ndarray,
    seq_len: int,
    forecast_steps: int,
    input_size: int,
    num_bins: int,
    bin_size: float,
):
    history = history[:seq_len]
    future = future[:forecast_steps]
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
    def __init__(
        self,
        path: str,
        seq_len: int = 16,
        forecast_steps: int = 96,
        input_size: int = 256,
        num_bins: int = 512,
        bin_size: float = 0.2,
        **kwargs,
    ):
        self.seq_len = seq_len
        self.forecast_steps = forecast_steps
        self.input_size = input_size
        self.num_bins = num_bins
        self.bin_size = bin_size
        with open(path, "r") as f:
            self.data_list = f.readlines()

    def prepare_input_and_target(self, info):
        radar = np.load(info["radar_path"])["arr_0"]
        mask = np.load(info["mask_path"])["arr_0"]
        radar = np.squeeze(radar).astype(np.float32)
        mask = np.squeeze(mask)
        radar[~mask] = 0.0
        history = radar[: self.seq_len]
        future = radar[self.seq_len : self.seq_len + self.forecast_steps]
        return prepare_metnet_sample(
            history,
            future,
            self.seq_len,
            self.forecast_steps,
            self.input_size,
            self.num_bins,
            self.bin_size,
        )

    def __getitem__(self, index):
        info = json.loads(self.data_list[index])
        history, target_bins = self.prepare_input_and_target(info)
        return {
            "inputs": torch.from_numpy(history),
            "targets": torch.from_numpy(target_bins),
        }

    def __len__(self):
        return len(self.data_list)

    collate_fn = staticmethod(collate_nowcast_batch)


@register_module(parent="datasets")
class SyntheticNowcastDataset(Dataset):
    """Random dataset for smoke testing training pipeline."""

    def __init__(
        self,
        length: int = 128,
        seq_len: int = 16,
        forecast_steps: int = 96,
        input_size: int = 256,
        num_bins: int = 512,
        bin_size: float = 0.2,
        **kwargs,
    ):
        self.length = length
        self.seq_len = seq_len
        self.forecast_steps = forecast_steps
        self.input_size = input_size
        self.num_bins = num_bins
        self.bin_size = bin_size

    def __getitem__(self, index):
        history = np.random.rand(self.seq_len, 1, self.input_size, self.input_size).astype(
            np.float32
        )
        future = np.random.rand(
            self.forecast_steps, self.input_size // 4, self.input_size // 4
        ).astype(np.float32)
        history, target_bins = prepare_metnet_sample(
            history[:, 0],
            future,
            self.seq_len,
            self.forecast_steps,
            self.input_size,
            self.num_bins,
            self.bin_size,
        )
        return {
            "inputs": torch.from_numpy(history),
            "targets": torch.from_numpy(target_bins),
        }

    def __len__(self):
        return self.length

    collate_fn = staticmethod(collate_nowcast_batch)
