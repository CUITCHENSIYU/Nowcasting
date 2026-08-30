#!/usr/bin/env python3
"""Smoke-test dataset + DataLoader built from rainfall_forecast.yaml."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import datasets.nowcast_dataset  # noqa: F401
from utils.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Test Nowcast DataLoader")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "rainfall_forecast.yaml"),
        help="path to yaml config",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=3, help="how many batches to iterate")
    return parser.parse_args()


def summarize_batch(batch, batch_idx: int):
    inputs = batch["inputs"]
    targets = batch["targets"]
    print(
        f"[batch {batch_idx}] "
        f"inputs={tuple(inputs.shape)} dtype={inputs.dtype} "
        f"range=[{inputs.min().item():.4g}, {inputs.max().item():.4g}] | "
        f"targets={tuple(targets.shape)} dtype={targets.dtype} "
        f"bins=[{targets.min().item()}, {targets.max().item()}]"
    )


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    runner_cfg = config.get("runner", {})
    batch_size = args.batch_size or runner_cfg.get("batch_size", 1)
    num_workers = args.num_workers if args.num_workers is not None else runner_cfg.get("num_workers", 0)

    if args.split not in dataset_cfg:
        raise KeyError(f"split '{args.split}' not found in dataset config keys={list(dataset_cfg)}")

    loader = build_dataloader(
        dataset_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        split=args.split,
    )
    dataset = loader.dataset

    print(f"config: {args.config}")
    print(f"split: {args.split}")
    print(f"dataset: {type(dataset).__name__}")
    print(f"path: {getattr(dataset, 'path', None)}")
    print(f"num_samples: {len(dataset)}")
    print(
        f"seq_len={getattr(dataset, 'seq_len', None)}, "
        f"forecast_steps={getattr(dataset, 'forecast_steps', None)}, "
        f"input_size={getattr(dataset, 'input_size', None)}"
    )
    print(f"batch_size={batch_size}, num_workers={num_workers}, num_batches={len(loader)}")

    t0 = time.time()
    sample = dataset[0]
    print(
        f"[sample 0] inputs={tuple(sample['inputs'].shape)}, "
        f"targets={tuple(sample['targets'].shape)}"
    )

    for i, batch in enumerate(loader):
        summarize_batch(batch, i)
        if i + 1 >= args.num_batches:
            break

    elapsed = time.time() - t0
    print(f"done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
