from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.skillfull_nowcasting.lasyers.evolution.evolution_network import Evolution_Network
from models.skillfull_nowcasting.lasyers.generation.generative_network import (
    Generative_Decoder,
    Generative_Encoder,
)
from models.skillfull_nowcasting.lasyers.generation.noise_projector import Noise_Projector
from models.skillfull_nowcasting.lasyers.utils import make_grid, warp
from utils.registry import register_module


@register_module(parent="models")
class SkillfullNowcasting(nn.Module):
    """NowcastNet-style skilful nowcasting generator.

    Input:  all_frames [B, T, H, W, C], T = input_length + pred_length
    Output: (gen_pred, evo_pred)
            gen_pred [B, pred_length, H, W, 1]
            evo_pred [B, pred_length, H, W, 1]  (before evo_div scaling)
    """

    def __init__(
        self,
        input_length: int = 12,
        pred_length: Optional[int] = None,
        total_length: Optional[int] = None,
        img_height: int = 256,
        img_width: int = 256,
        ngf: int = 32,
        evo_base_c: int = 32,
        evo_div: float = 1.0,
        ic_feature: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if pred_length is None:
            if total_length is None:
                raise ValueError("Provide pred_length or total_length")
            pred_length = int(total_length) - int(input_length)
        total_length = int(input_length) + int(pred_length)

        if img_height % 32 != 0 or img_width % 32 != 0:
            raise ValueError(
                f"img_height/img_width must be divisible by 32, got {(img_height, img_width)}"
            )

        self.input_length = int(input_length)
        self.pred_length = int(pred_length)
        self.total_length = total_length
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.ngf = int(ngf)
        self.evo_div = float(evo_div)

        # Generative encoder outputs ngf*8 @ H/8; noise projector contributes ngf*2 @ H/8.
        self.ic_feature = int(ic_feature) if ic_feature is not None else self.ngf * 10

        opt = SimpleNamespace(
            ngf=self.ngf,
            ic_feature=self.ic_feature,
            evo_ic=self.pred_length,
            gen_oc=self.pred_length,
        )
        self.opt = opt

        self.evo_net = Evolution_Network(
            self.input_length, self.pred_length, base_c=evo_base_c
        )
        self.gen_enc = Generative_Encoder(self.total_length, base_c=self.ngf)
        self.gen_dec = Generative_Decoder(opt)
        self.proj = Noise_Projector(self.ngf)

        sample_tensor = torch.zeros(1, 1, self.img_height, self.img_width)
        self.register_buffer("grid", make_grid(sample_tensor), persistent=False)

    def forward(self, all_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep only the first channel: [B, T, H, W, 1]
        all_frames = all_frames[:, :, :, :, :1]
        frames = all_frames.permute(0, 1, 4, 2, 3).contiguous()
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        device = frames.device

        input_frames = frames[:, : self.input_length]
        input_frames = input_frames.reshape(batch, self.input_length, height, width)

        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)

        series = []
        last_frames = all_frames[
            :, (self.input_length - 1) : self.input_length, :, :, 0
        ]
        grid = self.grid.repeat(batch, 1, 1, 1)
        for i in range(self.pred_length):
            last_frames = warp(
                last_frames,
                motion_[:, i],
                grid,
                mode="nearest",
                padding_mode="border",
            )
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)

        evo_for_gen = evo_result / self.evo_div

        evo_feature = self.gen_enc(torch.cat([input_frames, evo_for_gen], dim=1))

        noise = torch.randn(
            batch, self.ngf, height // 32, width // 32, device=device, dtype=frames.dtype
        )
        noise_feature = (
            self.proj(noise)
            .reshape(batch, -1, 4, 4, 8, 8)
            .permute(0, 1, 4, 5, 2, 3)
            .reshape(batch, -1, height // 8, width // 8)
        )

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_for_gen)

        gen_pred = gen_result.unsqueeze(-1)
        evo_pred = evo_result.unsqueeze(-1)
        return gen_pred, evo_pred
