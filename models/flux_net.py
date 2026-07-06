import torch
import torch.nn as nn
from typing import Dict, List

from .blocks import ConvBlock, DownsampleBlock, FLUXStage
from .ssm import S6LiteBottleneck
from .head import GAPHead, MSAPHead
from .config import FLUXConfig


class FLUXNet(nn.Module):
    """
    FLUX-Net v2: Frequency Lightweight Unified X-attention Network.

    A grid-based hybrid architecture with interleaved Conv + Spectral Attention
    blocks, FFT-based spectral fusion, S6-Lite SSM bottleneck, and multi-scale
    attention-pooled classification head.

    v2 improvements:
    - Wider dims (64, 128, 256, 384) for better capacity
    - Per-stage stochastic depth with linear schedule
    - Layer scale for training stability
    - Deeper SSM bottleneck (2 layers)
    - Fully from scratch — no pretrained weights
    """

    def __init__(self, config: FLUXConfig = None):
        super().__init__()
        cfg = config or FLUXConfig()
        dims = cfg.dims
        depths = cfg.depths
        drop_path_rates = cfg.drop_path_rates
        num_classes = cfg.num_classes

        # Stem: reduce 224×224×3 → 56×56×dim0
        # Use 4-layer stem for richer early features
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        # Stages with per-stage stochastic depth
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(dims)):
            dp_base = drop_path_rates[i - 1] if i > 0 else 0.0
            dp_max = drop_path_rates[i]
            self.stages.append(FLUXStage(
                dims[i], depths[i], cfg.expansion, cfg.dropout,
                drop_path_base=dp_base, drop_path_max=dp_max,
                spectral_fusion=cfg.spectral_fusion,
            ))
            if i < len(dims) - 1:
                self.downsamples.append(DownsampleBlock(dims[i], dims[i + 1]))

        # SSM bottleneck at deepest scale (now 2 layers)
        self.ssm = S6LiteBottleneck(
            d_model=dims[-1],
            d_state=cfg.ssm_d_state,
            expand=cfg.ssm_expand,
            num_layers=cfg.ssm_layers,
        ) if cfg.use_ssm else nn.Identity()

        # Classification head
        if cfg.use_msap:
            self.head = MSAPHead(
                scale_dims=list(dims),
                proj_dim=cfg.msap_proj_dim,
                num_classes=num_classes,
                dropout=cfg.dropout,
            )
        else:
            self.head = GAPHead(dims[-1], num_classes=num_classes, dropout=cfg.dropout)

        self.num_classes = num_classes
        self.dims = dims

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Apply careful weight initialization for stable training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        multi_scale = []
        x = self.stem(x)
        multi_scale.append(self.stages[0](x))
        for i in range(1, len(self.dims)):
            x = self.downsamples[i - 1](multi_scale[-1])
            multi_scale.append(self.stages[i](x))
        multi_scale[-1] = self.ssm(multi_scale[-1])
        logits = self.head(multi_scale)
        return {"logits": logits}

    def count_parameters(self) -> Dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
