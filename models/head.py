import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ScalePooler(nn.Module):
    """Pools one scale using avg + max, projects to common dimension."""
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim * 2, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        return self.proj(torch.cat([avg, mx], dim=-1))


class MSAPHead(nn.Module):
    """
    Multi-Scale Attention-Pooled Classification Head.
    
    Pools from all 4 encoder stages, learns input-dependent scale importance,
    and classifies with a refined MLP.
    """
    def __init__(self, scale_dims: List[int], proj_dim: int = 128,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_scales = len(scale_dims)
        self.poolers = nn.ModuleList([
            ScalePooler(d, proj_dim) for d in scale_dims
        ])
        self.scale_attn = nn.Sequential(
            nn.Linear(proj_dim * self.num_scales, self.num_scales * 4),
            nn.GELU(),
            nn.Linear(self.num_scales * 4, self.num_scales),
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        pooled = [self.poolers[i](features[i]) for i in range(self.num_scales)]
        stacked = torch.stack(pooled, dim=1)
        weights = F.softmax(self.scale_attn(torch.cat(pooled, dim=-1)), dim=-1)
        weighted = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return self.classifier(weighted)


class GAPHead(nn.Module):
    """Single-scale global pooling head for the no-MSAP ablation."""

    def __init__(self, in_dim: int, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return self.classifier(features[-1])
