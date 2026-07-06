import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epochs: int = 5,
                 T_0: int = 30, T_mult: int = 2, eta_min: float = 1e-6,
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            return [base * (epoch + 1) / max(self.warmup_epochs, 1)
                    for base in self.base_lrs]
        epoch_adj = epoch - self.warmup_epochs
        T_cur = self.T_0
        cum = 0
        while cum + T_cur <= epoch_adj:
            cum += T_cur
            T_cur = int(T_cur * self.T_mult)
        T_i = epoch_adj - cum
        cos = 0.5 * (1 + math.cos(math.pi * T_i / T_cur))
        return [self.eta_min + (base - self.eta_min) * cos for base in self.base_lrs]
