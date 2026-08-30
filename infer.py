#!/usr/bin/env python3
"""Load a MetNet checkpoint, run nowcast inference, and visualize results.

Training hyperparameters (model / dataset) default to ``config.yaml`` in the
same directory as ``model_path``. Inference I/O settings come from
``configs/inference.yaml``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.colors import Normalize

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.nowcast_dataset import center_crop, prepare_sample  # noqa: E402
from utils.builder import build_model  # noqa: E402
from utils.logger import get_logger  # noqa: E402


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_train_config(model_path: str, train_config: Optional[str] = None) -> str:
    if train_config is not None:
        return os.path.abspath(train_config)
    candidate = os.path.join(os.path.dirname(os.path.abspath(model_path)), "config.yaml")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"train config not found next to checkpoint: {candidate}. "
            "Pass --train-config explicitly."
        )
    return candidate


def load_modules():
    import datasets.nowcast_dataset  # noqa: F401
    import datasets.nowcast_dataset_rnn  # noqa: F401
    import datasets.nowcast_dataset_skillfull  # noqa: F401
    import models.metnet.metnet  # noqa: F401
    import models.predrnn.rnn  # noqa: F401
    import models.skillfull_nowcasting.skillfull_nowcasting  # noqa: F401


def _decode_timestamp(ts) -> str:
    if isinstance(ts, np.ndarray):
        ts = ts.item() if ts.ndim == 0 else ts[0]
    if isinstance(ts, (bytes, np.bytes_)):
        ts = ts.decode("utf-8", errors="replace")
    return str(ts)


def load_infer_sequence(
    path: str,
    seq_len: int,
    forecast_steps: int,
    scale: float,
    input_size: Optional[int] = None,
    norm: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Sequence[str]]:
    """Load one h5 sample and split into history / optional future rain maps."""
    with h5py.File(path, "r") as f:
        if "images" not in f:
            raise KeyError(f"'images' not found in {path}, keys={list(f.keys())}")
        radar = np.asarray(f["images"], dtype=np.float32)
        if "timestamps" in f:
            timestamps = [_decode_timestamp(t) for t in f["timestamps"][:]]
        else:
            timestamps = [f"t={i}" for i in range(radar.shape[0])]

    radar = np.squeeze(radar) * scale
    if radar.ndim != 3:
        raise ValueError(f"expected images [T, H, W], got {radar.shape}")

    if input_size is not None:
        radar = center_crop(radar, input_size, input_size)
    if norm is not None and norm > 0:
        radar = radar / norm

    min_frames = seq_len
    if radar.shape[0] < min_frames:
        raise ValueError(
            f"{path} has {radar.shape[0]} frames, need at least seq_len={seq_len}"
        )

    history = radar[:seq_len]
    future = None
    if radar.shape[0] >= seq_len + forecast_steps:
        future = radar[seq_len : seq_len + forecast_steps]
    return history, future, timestamps


def bins_to_rain(bins: np.ndarray, bin_size: float) -> np.ndarray:
    return bins.astype(np.float32) * bin_size


def decode_logits(logits: torch.Tensor, bin_size: float, method: str) -> np.ndarray:
    """Convert [C, H, W] or [B, C, H, W] logits to rain rate [H, W] / [B, H, W]."""
    squeeze_batch = False
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
        squeeze_batch = True

    if method == "argmax":
        rain = bins_to_rain(logits.argmax(dim=1).cpu().numpy(), bin_size)
    elif method == "expectation":
        probs = F.softmax(logits.float(), dim=1)
        centers = torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32)
        centers = centers * bin_size
        rain = (probs * centers.view(1, -1, 1, 1)).sum(dim=1).cpu().numpy()
    else:
        raise ValueError(f"unknown decode method: {method}")

    if squeeze_batch:
        rain = rain[0]
    return rain.astype(np.float32)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    forecast_steps: int,
    bin_size: float,
    decode: str,
    device: torch.device,
    enable_amp: bool,
) -> np.ndarray:
    """Return predicted rain maps with shape [forecast_steps, H, W]."""
    model.eval()
    preds = []
    for lead_time in range(forecast_steps):
        with torch.cuda.amp.autocast(enabled=enable_amp and device.type == "cuda"):
            logits = model(inputs, lead_time=lead_time)
        rain = decode_logits(logits[0], bin_size=bin_size, method=decode)
        preds.append(rain)
    return np.stack(preds, axis=0)


@torch.no_grad()
def run_skillfull_inference(
    model: torch.nn.Module,
    frames: torch.Tensor,
    device: torch.device,
    enable_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gen_pred, evo_pred) rain maps, each [forecast_steps, H, W]."""
    model.eval()
    with torch.cuda.amp.autocast(enabled=enable_amp and device.type == "cuda"):
        gen_pred, evo_pred, *_ = model(frames)
    gen = gen_pred[0, :, :, :, 0].detach().cpu().numpy().astype(np.float32)
    evo = evo_pred[0, :, :, :, 0].detach().cpu().numpy().astype(np.float32)
    return gen, evo


def _shared_vmax(*arrays: Optional[np.ndarray], percentile: float = 99.0) -> float:
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 1.0
    all_vals = np.concatenate([v.ravel() for v in vals])
    vmax = float(np.percentile(all_vals, percentile))
    if vmax <= 0:
        vmax = float(all_vals.max()) if all_vals.max() > 0 else 1.0
    return vmax


def _plot_row(
    axes_row,
    frames: np.ndarray,
    label: str,
    norm: Normalize,
    cmap: str,
    titles: Optional[Sequence[str]] = None,
) -> None:
    for c in range(frames.shape[0]):
        ax = axes_row[c]
        ax.imshow(frames[c], cmap=cmap, norm=norm, origin="upper")
        if titles is not None:
            ax.set_title(titles[c], fontsize=8)
        ax.axis("off")
    axes_row[0].text(
        -0.08,
        0.5,
        label,
        transform=axes_row[0].transAxes,
        va="center",
        ha="right",
        fontsize=10,
        rotation=90,
    )


def visualize(
    future: Optional[np.ndarray],
    pred: np.ndarray,
    timestamps: Sequence[str],
    seq_len: int,
    save_path: Optional[str],
    show: bool,
    evo: Optional[np.ndarray] = None,
    model_name: str = "nowcast",
    cmap: str = "turbo",
    dpi: int = 140,
) -> None:
    """Plot forecast panels.

    Rows (when available): GT, Evo, Gen/Pred.
    ``evo`` is only used for SkillfullNowcasting.
    """
    forecast_steps = pred.shape[0]
    rows = []
    if future is not None:
        rows.append(("GT", future))
    if evo is not None:
        rows.append(("Evo", evo))
    rows.append(("Gen" if evo is not None else "Pred", pred))

    nrows = len(rows)
    ncols = forecast_steps
    vmax = _shared_vmax(*(arr for _, arr in rows))
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.4 * ncols, 2.4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for r, (label, frames) in enumerate(rows):
        if r == 0 and label == "GT":
            titles = []
            for c in range(ncols):
                ts_idx = seq_len + c
                titles.append(
                    timestamps[ts_idx] if ts_idx < len(timestamps) else f"+{c + 1}"
                )
        else:
            titles = [f"+{c + 1}" for c in range(ncols)]
        _plot_row(axes[r], frames, label, norm, cmap, titles=titles)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        fraction=0.02,
        pad=0.01,
        label="precip",
    )
    fig.suptitle(
        f"{model_name} | pred={pred.shape[1:]} steps={forecast_steps} vmax={vmax:.3g}",
        fontsize=12,
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"saved visualization: {save_path}")

    if show:
        # Fall back to save-only when no GUI backend is available.
        backend = matplotlib.get_backend().lower()
        if "agg" in backend:
            print(f"show_result=true but backend={backend}; skip interactive display")
            plt.close(fig)
        else:
            plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="MetNet nowcast inference + visualization")
    parser.add_argument(
        "--config",
        "-c",
        default=os.path.join(ROOT, "configs", "inference.yaml"),
        help="inference yaml (model_path / infer_file / save options)",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="override train config; default: <model_dir>/config.yaml",
    )
    parser.add_argument("--model-path", default=None, help="override checkpoint path")
    parser.add_argument("--infer-file", default=None, help="override input .h5 path")
    parser.add_argument("--save-path", default=None, help="override output image path")
    parser.add_argument(
        "--decode",
        choices=("argmax", "expectation"),
        default=None,
        help="override decode method; default: value from inference.yaml",
    )
    parser.add_argument("--gpu", type=int, default=0, help="cuda device index")
    parser.add_argument("--no-amp", action="store_true", help="disable autocast")
    parser.add_argument("--cmap", default="turbo")
    return parser.parse_args()


def main():
    args = parse_args()
    load_modules()

    infer_cfg = load_yaml(args.config)
    model_path = os.path.abspath(args.model_path or infer_cfg["model_path"])
    infer_file = os.path.abspath(args.infer_file or infer_cfg["infer_file"])
    decode = args.decode or infer_cfg.get("decode", "expectation")
    if decode not in ("argmax", "expectation"):
        raise ValueError(f"decode must be 'argmax' or 'expectation', got: {decode}")
    save_result = bool(infer_cfg.get("save_result", True))
    show_result = bool(infer_cfg.get("show_result", False))
    save_path = args.save_path or infer_cfg.get("save_path")
    if save_path:
        save_path = os.path.abspath(save_path)
    if not save_result:
        save_path = None

    train_config_path = resolve_train_config(model_path, args.train_config)
    train_cfg = load_yaml(train_config_path)
    dataset_cfg = train_cfg["dataset"]
    model_cfg = train_cfg["model"]
    model_type = model_cfg.get("type", "MetNet")

    seq_len = int(dataset_cfg["seq_len"])
    forecast_steps = int(dataset_cfg.get("forecast_steps", model_cfg.get("forecast_steps")))
    input_size = int(dataset_cfg["input_size"])
    scale = float(dataset_cfg.get("scale", 1.0))
    norm = dataset_cfg.get("norm")
    norm = float(norm) if norm is not None and norm != "" else None

    num_bins = int(dataset_cfg.get("num_bins", model_cfg.get("output_channels", 250)))
    bin_size = float(dataset_cfg.get("bin_size", 0.1))

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"checkpoint not found: {model_path}")
    if not os.path.isfile(infer_file):
        raise FileNotFoundError(f"infer file not found: {infer_file}")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    logger = get_logger("infer")
    logger.info(f"train_config={train_config_path}")
    logger.info(f"model_path={model_path}")
    logger.info(f"infer_file={infer_file}")
    logger.info(f"model_type={model_type}")
    logger.info(f"decode={decode}")
    logger.info(f"device={device}")

    model = build_model(
        model_cfg,
        device=device,
        logger=logger,
        pretrained_weight_path=model_path,
    )

    history_full, future_full, timestamps = load_infer_sequence(
        infer_file,
        seq_len=seq_len,
        forecast_steps=forecast_steps,
        scale=scale,
        input_size=input_size,
        norm=norm,
    )

    evo_vis = None
    if model_type == "SkillfullNowcasting":
        if decode != "expectation":
            logger.info(f"decode={decode} ignored for SkillfullNowcasting (regression output)")
        if future_full is None:
            raise ValueError(
                f"infer file needs seq_len+forecast_steps={seq_len + forecast_steps} frames"
            )
        full_seq = np.concatenate([history_full, future_full], axis=0)
        frames = torch.from_numpy(full_seq.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
        logger.info(f"frames shape={tuple(frames.shape)}")
        pred, evo_vis = run_skillfull_inference(
            model,
            frames,
            device=device,
            enable_amp=not args.no_amp,
        )
        future_vis = future_full
        logger.info(
            f"evo shape={evo_vis.shape}, range=[{evo_vis.min():.4g}, {evo_vis.max():.4g}]"
        )
    elif model_type == "MetNet":
        # Match training crop: history -> input_size, future -> input_size // 4
        if future_full is not None:
            history, target_bins = prepare_sample(
                history_full,
                future_full,
                input_size=input_size,
                num_bins=num_bins,
                bin_size=bin_size,
            )
            future_vis = bins_to_rain(target_bins, bin_size)
        else:
            history = center_crop(history_full, input_size, input_size)
            if history.ndim == 3:
                history = history[:, None, :, :]
            future_vis = None

        inputs = torch.from_numpy(history).unsqueeze(0).to(device)  # [1, T, 1, H, W]
        logger.info(f"inputs shape={tuple(inputs.shape)}")

        pred = run_inference(
            model,
            inputs,
            forecast_steps=forecast_steps,
            bin_size=bin_size,
            decode=decode,
            device=device,
            enable_amp=not args.no_amp,
        )
    else:
        raise ValueError(
            f"infer.py does not support model type '{model_type}'. "
            "Supported: MetNet, SkillfullNowcasting"
        )
    logger.info(
        f"pred shape={pred.shape}, range=[{pred.min():.4g}, {pred.max():.4g}], "
        f"decode={decode}"
    )

    visualize(
        future=future_vis,
        pred=pred,
        evo=evo_vis,
        timestamps=timestamps,
        seq_len=seq_len,
        save_path=save_path,
        show=show_result,
        model_name=model_type,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
