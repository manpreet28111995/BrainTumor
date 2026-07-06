import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smoothed = torch.zeros_like(log_probs)
            smoothed.fill_(self.smoothing / (num_classes - 1))
            smoothed.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(smoothed * log_probs).sum(dim=-1).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


class ClassificationLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, focal_weight: float = 0.0,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.ce = LabelSmoothingCrossEntropy(smoothing)
        self.focal_weight = focal_weight
        if focal_weight > 0:
            self.focal = FocalLoss(focal_gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(pred, target)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(pred, target)
        return loss
