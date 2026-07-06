import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class StochasticDepth(nn.Module):
    """Drop entire residual branches randomly during training (DropPath).

    Each sample in the batch independently decides whether to keep or
    drop the residual, scaled by 1/(1-drop_prob) to preserve expectation.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device)
        noise = noise.bernoulli_(keep_prob).div_(keep_prob)
        return x * noise


class ConvBlock(nn.Module):
    """
    ConvNeXt-style depthwise separable block.

    LayerNorm → DW Conv 7×7 → PW Conv 1×1 (expand) → GELU → PW Conv 1×1 (squeeze) + residual
    Includes stochastic depth (DropPath) for regularization.
    """

    def __init__(self, dim: int, expansion: int = 3, dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv_dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.conv_pw1 = nn.Conv2d(dim, dim * expansion, 1, bias=False)
        self.act = nn.GELU()
        self.conv_pw2 = nn.Conv2d(dim * expansion, dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(drop_path)

        # Layer scale for stable training (init near 1.0)
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        h = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.conv_dw(h)
        h = self.conv_pw1(h)
        h = self.act(h)
        h = self.conv_pw2(h)
        h = self.drop(h)
        h = self.layer_scale * h
        return shortcut + self.drop_path(h)


class SpectralAttnBlock(nn.Module):
    """
    Spectral Attention block: FFT spectral fusion + FeedForward.

    LayerNorm → SpectralFusion (FFT gate) → FFN (expand 2×) + residual
    Includes stochastic depth (DropPath) for regularization.
    """

    def __init__(self, dim: int, dropout: float = 0.1, drop_path: float = 0.0,
                 spectral_fusion: bool = True):
        super().__init__()
        from .fusion import SpectralFusion
        self.norm = nn.LayerNorm(dim)
        self.spectral = SpectralFusion(dim) if spectral_fusion else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.drop_path = StochasticDepth(drop_path)

        # Layer scale for stable training
        self.layer_scale = nn.Parameter(torch.ones(1, 1, dim) * 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        h = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.spectral(h)
        h = h.permute(0, 2, 3, 1)
        h = self.ffn_norm(h)
        h = self.layer_scale * self.ffn(h)
        h = self.drop_path(h)
        return (shortcut.permute(0, 2, 3, 1) + h).permute(0, 3, 1, 2)


class DownsampleBlock(nn.Module):
    """Strided convolution for spatial downsampling + channel doubling."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.conv(x)


class FLUXStage(nn.Module):
    """One stage of FLUX-Net: repeated ConvBlock + SpectralAttnBlock pairs.

    Stochastic depth probability increases linearly across blocks,
    providing stronger regularization in deeper stages.
    """

    def __init__(self, dim: int, depth: int, expansion: int = 3, dropout: float = 0.1,
                 drop_path_base: float = 0.0, drop_path_max: float = 0.2,
                 spectral_fusion: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList()
        total_blocks = depth * 2  # ConvBlock + SpectralAttnBlock per depth step
        for i in range(depth):
            # Linear schedule: earlier blocks have lower drop rate
            dp_conv = drop_path_base + (drop_path_max - drop_path_base) * (2 * i) / max(total_blocks - 1, 1)
            dp_spec = drop_path_base + (drop_path_max - drop_path_base) * (2 * i + 1) / max(total_blocks - 1, 1)
            self.blocks.append(ConvBlock(dim, expansion, dropout, drop_path=dp_conv))
            self.blocks.append(SpectralAttnBlock(
                dim, dropout, drop_path=dp_spec, spectral_fusion=spectral_fusion))

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x
