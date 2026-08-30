import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm  # noqa: F401


def make_grid(input_tensor: torch.Tensor) -> torch.Tensor:
    """Build a pixel-coordinate grid [B, 2, H, W] for flow warping."""
    batch, _, height, width = input_tensor.size()
    xx = torch.arange(0, width).view(1, -1).repeat(height, 1)
    yy = torch.arange(0, height).view(-1, 1).repeat(1, width)
    xx = xx.view(1, 1, height, width).repeat(batch, 1, 1, 1)
    yy = yy.view(1, 1, height, width).repeat(batch, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    return grid


def warp(
    input_tensor: torch.Tensor,
    flow: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Warp ``input_tensor`` by optical-flow ``flow`` using ``grid``."""
    _, _, height, width = input_tensor.size()
    vgrid = grid + flow
    vgrid = vgrid.clone()
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(width - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(height - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(
        input_tensor,
        vgrid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
