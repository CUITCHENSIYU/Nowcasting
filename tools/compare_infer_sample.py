#!/usr/bin/env python3
"""Compare MetNet vs Skillful predictions on one h5 sample."""
import os
import sys
import yaml
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import datasets.nowcast_dataset  # noqa: F401
import datasets.nowcast_dataset_skillfull  # noqa: F401
import models.metnet.metnet  # noqa: F401
import models.skillfull_nowcasting.skillfull_nowcasting  # noqa: F401

from datasets.nowcast_dataset import center_crop, prepare_sample
from infer import (
    bins_to_rain,
    load_infer_sequence,
    run_inference,
    run_skillfull_inference,
)
from utils.builder import build_model


def metrics(pred, gt, name):
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mask = gt > 0.01
    if mask.sum() > 0:
        corr = float(np.corrcoef(pred[mask], gt[mask])[0, 1])
        mae_rain = float(np.mean(np.abs(diff[mask])))
        max_gt = float(gt.max())
        max_pred = float(pred.max())
        peak_ratio = max_pred / max_gt if max_gt > 0 else 0.0
    else:
        corr, mae_rain, peak_ratio = 0.0, 0.0, 0.0
    return {
        "name": name,
        "mse": mse,
        "mae": mae,
        "corr_rain": corr,
        "mae_rain": mae_rain,
        "max_gt": float(gt.max()),
        "max_pred": float(pred.max()),
        "peak_ratio": peak_ratio,
    }


def main():
    infer_file = (
        "/mnt/share-gpuserver-0-disk2-nfs/datasets/depth_estimation_S/data_mining/"
        "seq_12_out-seq_6_threshold_20/test/20190814_164500.h5"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    runs = {
        "skillful": (
            "/mnt/gpuserver-1-disk0-nfs/chensiyu/github/Nowcasting/runs/"
            "2026-08-23-skillfull-nowcast-seq12-out6-size256-time2-space3"
        ),
        "metnet": (
            "/mnt/gpuserver-1-disk0-nfs/chensiyu/github/Nowcasting/runs/"
            "2026-08-16-metnet-nowcast-seq12-out6-size256-time2-space3"
        ),
    }

    results = {}
    for key, run_dir in runs.items():
        with open(os.path.join(run_dir, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        ds, mc = cfg["dataset"], cfg["model"]
        seq_len = int(ds["seq_len"])
        forecast_steps = int(ds.get("forecast_steps", mc.get("forecast_steps")))
        input_size = int(ds["input_size"])
        scale = float(ds.get("scale", 1.0))
        norm = ds.get("norm")
        norm = float(norm) if norm is not None and norm != "" else None
        model_path = os.path.join(run_dir, "latest.pth")

        model = build_model(mc, device=device, logger=None, pretrained_weight_path=model_path)

        if key == "skillful":
            hist, fut, _ = load_infer_sequence(
                infer_file, seq_len, forecast_steps, scale, input_size, norm
            )
            full = np.concatenate([hist, fut], axis=0)
            frames = torch.from_numpy(full.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
            pred = run_skillfull_inference(model, frames, device, True)
            gt = fut
        else:
            num_bins = int(ds.get("num_bins", 250))
            bin_size = float(ds.get("bin_size", 0.1))
            hist, fut, _ = load_infer_sequence(infer_file, seq_len, forecast_steps, scale)
            hist, target_bins = prepare_sample(hist, fut, input_size, num_bins, bin_size)
            gt = bins_to_rain(target_bins, bin_size)
            inputs = torch.from_numpy(hist).unsqueeze(0).to(device)
            pred = run_inference(model, inputs, forecast_steps, bin_size, "expectation", device, True)

        results[key] = {"pred": pred, "gt": gt}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gt64 = results["metnet"]["gt"]
    sk64 = np.stack(
        [center_crop(results["skillful"]["pred"][i], 64, 64) for i in range(6)], axis=0
    )
    gt256 = results["skillful"]["gt"]
    mn256 = np.repeat(np.repeat(results["metnet"]["pred"], 4, axis=1), 4, axis=2)

    print("=== Native resolution ===")
    for key in results:
        h, w = results[key]["pred"].shape[1], results[key]["pred"].shape[2]
        print(metrics(results[key]["pred"], results[key]["gt"], f"{key} @ {h}x{w}"))

    print("\n=== Fair compare @ 64x64 ===")
    print(metrics(results["metnet"]["pred"], gt64, "metnet"))
    print(metrics(sk64, gt64, "skillful (center-crop to 64)"))

    print("\n=== Fair compare @ 256x256 ===")
    print(metrics(results["skillful"]["pred"], gt256, "skillful"))
    print(metrics(mn256, gt256, "metnet (nearest upsample x4)"))

    print("\n=== Per-step MSE @ 64x64 ===")
    for t in range(6):
        m_mse = float(np.mean((results["metnet"]["pred"][t] - gt64[t]) ** 2))
        s_mse = float(np.mean((sk64[t] - gt64[t]) ** 2))
        winner = "skillful" if s_mse < m_mse else "metnet"
        print(f"step+{t + 1}: metnet={m_mse:.5f} skillful={s_mse:.5f} -> {winner}")


if __name__ == "__main__":
    main()
