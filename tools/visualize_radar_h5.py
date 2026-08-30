#!/usr/bin/env python3
"""Visualize radar precipitation sequences stored in HDF5 files.

Supports:
  1) Per-sequence files written by datasets/netherlands/create_dataset.py
     keys: images [T, H, W], timestamps [T] / [T, 1]
  2) Source NL radar file with train/test groups
     keys: {group}/images, {group}/timestamps
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import Normalize


def _decode_timestamp(ts) -> str:
    if isinstance(ts, np.ndarray):
        ts = ts.item() if ts.ndim == 0 else ts[0]
    if isinstance(ts, (bytes, np.bytes_)):
        ts = ts.decode("utf-8", errors="replace")
    return str(ts)


def _load_sequence(
    path: str,
    group: Optional[str] = None,
    start: int = 0,
    length: Optional[int] = None,
) -> Tuple[np.ndarray, Sequence[str]]:
    with h5py.File(path, "r") as f:
        if group:
            if group not in f:
                available = list(f.keys())
                raise KeyError(f"group '{group}' not found. available={available}")
            root = f[group]
        else:
            root = f if "images" in f else None
            if root is None:
                # fall back to first group that contains images
                for key in f.keys():
                    if isinstance(f[key], h5py.Group) and "images" in f[key]:
                        root = f[key]
                        group = key
                        break
                if root is None:
                    raise KeyError(
                        f"no 'images' dataset found in {path}. top-level keys={list(f.keys())}"
                    )

        images_ds = root["images"]
        n_total = images_ds.shape[0]
        if start < 0 or start >= n_total:
            raise IndexError(f"start={start} out of range for length={n_total}")
        end = n_total if length is None else min(n_total, start + length)
        images = np.asarray(images_ds[start:end], dtype=np.float32)

        if "timestamps" in root:
            timestamps = [_decode_timestamp(t) for t in root["timestamps"][start:end]]
        else:
            timestamps = [f"t={start + i}" for i in range(images.shape[0])]

    images = np.squeeze(images)
    if images.ndim != 3:
        raise ValueError(f"expected images with shape [T, H, W], got {images.shape}")
    return images, timestamps


def _vmax(images: np.ndarray, percentile: float) -> float:
    finite = images[np.isfinite(images)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(finite, percentile))
    if vmax <= 0:
        vmax = float(finite.max()) if finite.max() > 0 else 1.0
    return vmax


def plot_grid(
    images: np.ndarray,
    timestamps: Sequence[str],
    out_path: str,
    cmap: str,
    vmax: float,
    ncols: int,
    dpi: int,
    title: str,
) -> None:
    t = images.shape[0]
    ncols = max(1, min(ncols, t))
    nrows = int(np.ceil(t / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols, 2.6 * nrows),
        squeeze=False,
    )
    norm = Normalize(vmin=0.0, vmax=vmax)
    im = None
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i < t:
            im = ax.imshow(images[i], cmap=cmap, norm=norm, origin="upper")
            ax.set_title(timestamps[i], fontsize=8)
        ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="precip")
    fig.suptitle(title, fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved grid: {out_path}")


def plot_animation(
    images: np.ndarray,
    timestamps: Sequence[str],
    out_path: str,
    cmap: str,
    vmax: float,
    fps: int,
    dpi: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    norm = Normalize(vmin=0.0, vmax=vmax)
    im = ax.imshow(images[0], cmap=cmap, norm=norm, origin="upper")
    title_artist = ax.set_title(timestamps[0], fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="precip")
    fig.suptitle(title, fontsize=12)

    def _update(frame: int):
        im.set_data(images[frame])
        title_artist.set_text(timestamps[frame])
        return im, title_artist

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=images.shape[0],
        interval=1000 / max(fps, 1),
        blit=False,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".gif":
        anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    else:
        try:
            anim.save(out_path, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
        except (RuntimeError, ValueError) as exc:
            gif_path = os.path.splitext(out_path)[0] + ".gif"
            print(f"ffmpeg unavailable ({exc}); falling back to {gif_path}")
            anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
            out_path = gif_path
    plt.close(fig)
    print(f"saved animation: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize radar data from an HDF5 file")
    parser.add_argument("h5_path", help="path to .h5 file")
    parser.add_argument(
        "--group",
        default=None,
        help="HDF5 group name for source files, e.g. train / test",
    )
    parser.add_argument("--start", type=int, default=0, help="start frame index")
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="number of frames to visualize (default: all / whole sequence)",
    )
    parser.add_argument(
        "--mode",
        choices=("grid", "anim", "both"),
        default="both",
        help="visualization mode",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path prefix or file. default: ./vis/<h5_stem>",
    )
    parser.add_argument("--cmap", default="turbo", help="matplotlib colormap")
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.0,
        help="percentile used as color scale max (ignore zeros-heavy frames)",
    )
    parser.add_argument("--vmax", type=float, default=None, help="fixed color scale max")
    parser.add_argument("--ncols", type=int, default=6, help="grid columns")
    parser.add_argument("--fps", type=int, default=4, help="animation fps")
    parser.add_argument("--dpi", type=int, default=120, help="output dpi")
    return parser.parse_args()


def main():
    args = parse_args()
    images, timestamps = _load_sequence(
        args.h5_path,
        group=args.group,
        start=args.start,
        length=args.length,
    )
    vmax = args.vmax if args.vmax is not None else _vmax(images, args.vmax_percentile)
    stem = os.path.splitext(os.path.basename(args.h5_path))[0]
    if args.group:
        stem = f"{stem}_{args.group}"
    if args.out is None:
        out_prefix = os.path.join("vis", stem)
    else:
        out_prefix = args.out
        if out_prefix.endswith((".png", ".gif", ".mp4")):
            out_prefix = os.path.splitext(out_prefix)[0]

    title = f"{os.path.basename(args.h5_path)}"
    if args.group:
        title += f" [{args.group}]"
    title += f" | frames={images.shape[0]} shape={images.shape[1:]} vmax={vmax:.4g}"

    print(
        f"loaded images={images.shape}, range=[{images.min():.4g}, {images.max():.4g}], "
        f"mean={images.mean():.4g}, vmax={vmax:.4g}"
    )

    if args.mode in ("grid", "both"):
        plot_grid(
            images,
            timestamps,
            out_path=f"{out_prefix}_grid.png",
            cmap=args.cmap,
            vmax=vmax,
            ncols=args.ncols,
            dpi=args.dpi,
            title=title,
        )
    if args.mode in ("anim", "both"):
        plot_animation(
            images,
            timestamps,
            out_path=f"{out_prefix}.gif",
            cmap=args.cmap,
            vmax=vmax,
            fps=args.fps,
            dpi=args.dpi,
            title=title,
        )


if __name__ == "__main__":
    main()
