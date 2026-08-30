from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.predrnn.lasyers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
from utils.registry import register_module


def reshape_patch(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, T, H, W, C] -> [B, T, H/ps, W/ps, C*ps*ps]."""
    batch, seq, height, width, channel = frames.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"H/W must be divisible by patch_size={patch_size}, got {(height, width)}"
        )
    frames = frames.view(
        batch,
        seq,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channel,
    )
    frames = frames.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return frames.view(
        batch,
        seq,
        height // patch_size,
        width // patch_size,
        channel * patch_size * patch_size,
    )


def reshape_patch_back(patches: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, T, H/ps, W/ps, C*ps*ps] -> [B, T, H, W, C]."""
    batch, seq, height, width, channel = patches.shape
    out_ch = channel // (patch_size * patch_size)
    patches = patches.view(
        batch, seq, height, width, patch_size, patch_size, out_ch
    )
    patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return patches.view(
        batch, seq, height * patch_size, width * patch_size, out_ch
    )


@register_module(parent="models")
class PredRNN(nn.Module):
    """PredRNN-V2 adapted for radar nowcasting (no action conditioning).

    Expected inputs:
      all_frames: [B, T, H, W, C] already patch-reshaped
      mask_true:  [B, T-1, H, W, C] scheduled-sampling mask in patch space
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_hidden: Optional[Union[Sequence[int], int]] = None,
        img_channel: int = 1,
        img_size: int = 64,
        patch_size: int = 4,
        filter_size: int = 5,
        stride: int = 1,
        layer_norm: bool = True,
        decouple_beta: float = 0.1,
        conv_on_input: int = 0,
        res_on_conv: int = 0,
        total_length: int = 18,
        **kwargs,
    ):
        super().__init__()
        if num_hidden is None:
            num_hidden = [128] * num_layers
        if isinstance(num_hidden, int):
            num_hidden = [num_hidden] * num_layers
        num_hidden = list(num_hidden)
        if len(num_hidden) != num_layers:
            raise ValueError(
                f"num_hidden length ({len(num_hidden)}) must equal num_layers ({num_layers})"
            )

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.img_channel = img_channel
        self.img_size = img_size
        self.patch_size = patch_size
        self.filter_size = filter_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.beta = decouple_beta
        self.conv_on_input = int(conv_on_input)
        self.res_on_conv = int(res_on_conv)
        self.total_length = total_length

        self.patch_height = img_size // patch_size
        self.patch_width = img_size // patch_size
        self.patch_ch = img_channel * (patch_size ** 2)
        self.rnn_height = self.patch_height
        self.rnn_width = self.patch_width

        if self.conv_on_input == 1:
            self.rnn_height = self.patch_height // 4
            self.rnn_width = self.patch_width // 4
            self.conv_input1 = nn.Conv2d(
                self.patch_ch,
                num_hidden[0] // 2,
                filter_size,
                stride=2,
                padding=filter_size // 2,
                bias=False,
            )
            self.conv_input2 = nn.Conv2d(
                num_hidden[0] // 2,
                num_hidden[0],
                filter_size,
                stride=2,
                padding=filter_size // 2,
                bias=False,
            )
            self.deconv_output1 = nn.ConvTranspose2d(
                num_hidden[num_layers - 1],
                num_hidden[num_layers - 1] // 2,
                filter_size,
                stride=2,
                padding=filter_size // 2,
                bias=False,
            )
            self.deconv_output2 = nn.ConvTranspose2d(
                num_hidden[num_layers - 1] // 2,
                self.patch_ch,
                filter_size,
                stride=2,
                padding=filter_size // 2,
                bias=False,
            )

        cell_list = []
        for i in range(num_layers):
            if i == 0:
                in_channel = num_hidden[0] if self.conv_on_input == 1 else self.patch_ch
            else:
                in_channel = num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(
                    in_channel,
                    num_hidden[i],
                    self.rnn_width,
                    filter_size,
                    stride,
                    layer_norm,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        if self.conv_on_input == 0:
            self.conv_last = nn.Conv2d(
                num_hidden[num_layers - 1],
                self.patch_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        self.adapter = nn.Conv2d(
            num_hidden[num_layers - 1],
            num_hidden[num_layers - 1],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.mse_criterion = nn.MSELoss()

    def forward(self, all_frames: torch.Tensor, mask_true: torch.Tensor):
        # [B, T, H, W, C] -> [B, T, C, H, W]
        frames = all_frames.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        device = frames.device

        next_frames = []
        h_t: List[torch.Tensor] = []
        c_t: List[torch.Tensor] = []
        delta_c_list: List[torch.Tensor] = []
        delta_m_list: List[torch.Tensor] = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                batch_size,
                self.num_hidden[i],
                self.rnn_height,
                self.rnn_width,
                device=device,
                dtype=frames.dtype,
            )
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros(
            batch_size,
            self.num_hidden[0],
            self.rnn_height,
            self.rnn_width,
            device=device,
            dtype=frames.dtype,
        )
        decouple_loss = []

        for t in range(self.total_length - 1):
            if t == 0:
                net = frames[:, t]
            else:
                net = (
                    mask_true[:, t - 1] * frames[:, t]
                    + (1.0 - mask_true[:, t - 1]) * x_gen
                )

            if self.conv_on_input == 1:
                net_shape1 = net.size()
                net = self.conv_input1(net)
                if self.res_on_conv == 1:
                    input_net1 = net
                net_shape2 = net.size()
                net = self.conv_input2(net)
                if self.res_on_conv == 1:
                    input_net2 = net

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](
                net, h_t[0], c_t[0], memory
            )
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1),
                dim=2,
            )
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1),
                dim=2,
            )

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1),
                    dim=2,
                )
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1),
                    dim=2,
                )

            for i in range(self.num_layers):
                decouple_loss.append(
                    torch.mean(
                        torch.abs(
                            torch.cosine_similarity(
                                delta_c_list[i], delta_m_list[i], dim=2
                            )
                        )
                    )
                )

            if self.conv_on_input == 1:
                if self.res_on_conv == 1:
                    x_gen = self.deconv_output1(
                        h_t[self.num_layers - 1] + input_net2, output_size=net_shape2
                    )
                    x_gen = self.deconv_output2(
                        x_gen + input_net1, output_size=net_shape1
                    )
                else:
                    x_gen = self.deconv_output1(
                        h_t[self.num_layers - 1], output_size=net_shape2
                    )
                    x_gen = self.deconv_output2(x_gen, output_size=net_shape1)
            else:
                x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [T-1, B, C, H, W] -> [B, T-1, H, W, C]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = (
            self.mse_criterion(next_frames, all_frames[:, 1:])
            + self.beta * decouple_loss
        )
        return next_frames, loss
