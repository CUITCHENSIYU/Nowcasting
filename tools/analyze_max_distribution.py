#!/usr/bin/env python3
"""Analyze per-sample max radar values in an h5 sequence folder and plot the distribution."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Plot distribution of per-sample max radar values")
    parser.add_argument(
        "--path",
        default=(
            "/mnt/share-gpuserver-0-disk2-nfs/datasets/depth_estimation_S/"
            "data_mining/seq_12_out-seq_6_threshold_20/train"
        ),
        help="folder containing per-sequence .h5 files",
    )
    parser.add_argument(
        "--out",
        default="vis/train_max_distribution.png",
        help="output figure path",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply raw values by this factor before computing max (e.g. 47.83)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="only use the first N files (default: all)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="parallel workers for reading h5 files",
    )
    parser.add_argument("--bins", type=int, default=80, help="histogram bins")
    parser.add_argument(
        "--logy",
        action="store_true",
        help="use log scale on y-axis",
    )
    return parser.parse_args()


def sample_max(path: str, scale: float) -> float:
    with h5py.File(path, "r") as f:
        images = f["images"]
        # Stream by frame to avoid loading huge arrays when unnecessary.
        vmax = 0.0
        for i in range(images.shape[0]):
            frame_max = float(np.max(images[i]))
            if frame_max > vmax:
                vmax = frame_max
    return vmax * scale


def main():
    args = parse_args()
    files = sorted(glob(os.path.join(args.path, "*.h5")))
    if not files:
        raise FileNotFoundError(f"no .h5 files under: {args.path}")
    if args.max_files is not None:
        files = files[: args.max_files]

    print(f"scanning {len(files)} files from {args.path}")
    print(f"scale={args.scale}, workers={args.num_workers}")

    max_vals = np.empty(len(files), dtype=np.float64)
    if args.num_workers <= 1:
        for i, path in enumerate(tqdm(files, desc="reading")):
            max_vals[i] = sample_max(path, args.scale)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(sample_max, path, args.scale): i for i, path in enumerate(files)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="reading"):
                i = futures[fut]
                max_vals[i] = fut.result()

    finite = max_vals[np.isfinite(max_vals)]
    print(
        f"max stats: n={finite.size}, min={finite.min():.6g}, "
        f"max={finite.max():.6g}, mean={finite.mean():.6g}, "
        f"median={np.median(finite):.6g}"
    )
    qs = np.percentile(finite, [50, 90, 95, 99, 99.9])
    print(
        "percentiles "
        f"p50={qs[0]:.6g}, p90={qs[1]:.6g}, p95={qs[2]:.6g}, "
        f"p99={qs[3]:.6g}, p99.9={qs[4]:.6g}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(finite, bins=args.bins, color="#3b6ea5", edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel(f"per-sample max (scale={args.scale:g})")
    axes[0].set_ylabel("count")
    axes[0].set_title("Max value histogram")
    if args.logy:
        axes[0].set_yscale("log")

    # CDF
    sorted_vals = np.sort(finite)
    cdf = np.arange(1, sorted_vals.size + 1) / sorted_vals.size
    axes[1].plot(sorted_vals, cdf, color="#c45c26", linewidth=1.5)
    axes[1].set_xlabel(f"per-sample max (scale={args.scale:g})")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Max value CDF")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Train max distribution | n={finite.size} | "
        f"min={finite.min():.4g}, max={finite.max():.4g}, mean={finite.mean():.4g}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # also dump raw maxima for later reuse
    npy_path = os.path.splitext(args.out)[0] + "_values.npy"
    np.save(npy_path, finite)
    print(f"saved figure: {args.out}")
    print(f"saved values: {npy_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
