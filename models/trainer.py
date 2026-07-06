import os, json, math, time, shutil, traceback
import numpy as np
from typing import Optional
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .flux_net import FLUXNet
from .config import Config
from .data import get_dataloaders
from .losses import ClassificationLoss
from .scheduler import WarmupCosineScheduler
from .metrics import MetricTracker


# ─────────────────────────────────────────────────────────────────────────────
# Model EMA
# ─────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average of model weights.

    During validation, use ema.apply() context to temporarily swap in EMA
    weights. EMA weights generalize better than the latest SGD weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        shadow_params = dict(self.shadow.named_parameters())
        model_params = dict(model.named_parameters())
        for name, s_param in shadow_params.items():
            m_param = model_params[name]
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

        # Buffers such as BatchNorm running_mean/running_var are not trainable
        # parameters, but they must match the EMA model used for inference.
        shadow_buffers = dict(self.shadow.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, s_buffer in shadow_buffers.items():
            s_buffer.data.copy_(model_buffers[name].data)

    def apply(self, model: nn.Module):
        """Context manager: temporarily replace model weights with EMA weights."""
        return _EMAContext(self, model)


class _EMAContext:
    def __init__(self, ema: ModelEMA, model: nn.Module):
        self.ema = ema
        self.model = model
        self._backup = {}

    def __enter__(self):
        self._backup = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }
        self.model.load_state_dict(self.ema.shadow.state_dict(), strict=True)
        return self

    def __exit__(self, *args):
        self.model.load_state_dict(self._backup, strict=True)
        self._backup.clear()


# ─────────────────────────────────────────────────────────────────────────────
# JSON encoder
# ─────────────────────────────────────────────────────────────────────────────

class EpochHistoryEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: Config, device: torch.device,
                 resume_path: Optional[str] = None):
        if hasattr(cfg, "refresh_paths"):
            cfg.refresh_paths()
        self.cfg = cfg
        self.device = device

        self.model = self._build_model().to(device)
        self.criterion = ClassificationLoss(
            smoothing=cfg.training.label_smoothing)
        self.optimizer = self._build_optimizer()
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=cfg.training.warmup_epochs,
            T_0=cfg.training.T_0,
            T_mult=cfg.training.T_mult,
            eta_min=cfg.training.eta_min,
        )
        self.scaler = GradScaler(enabled=cfg.training.use_amp)

        # Model EMA for smoother validation
        self.ema = ModelEMA(self.model, decay=cfg.training.ema_decay)

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = -1
        self.epochs_since_improvement = 0
        self.start_epoch = 0
        self.global_step = 0
        self.train_history = []
        self._nan_streak = 0  # consecutive NaN epoch counter

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.metrics_dir, exist_ok=True)

        self.train_loader, self.val_loader, self.test_loader = \
            get_dataloaders(cfg)

        # Print model summary
        params = self.model.count_parameters()
        arch_name = getattr(cfg.model, "architecture", "fluxnet")
        print(f"{arch_name} | Params: {params['total']:,} total, "
              f"{params['trainable']:,} trainable")

        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)
            print(f"Resumed from checkpoint: epoch {self.start_epoch}")

    def _build_model(self):
        arch = getattr(self.cfg.model, "architecture", "fluxnet").lower()
        arch = arch.replace("-", "_")
        if arch in ("fluxnet_no_spectral_fusion", "flux_no_spectral_fusion"):
            self.cfg.model.spectral_fusion = False
            return FLUXNet(self.cfg.model)
        if arch in ("fluxnet_no_ssm", "flux_no_ssm"):
            self.cfg.model.use_ssm = False
            return FLUXNet(self.cfg.model)
        if arch in ("fluxnet_no_msap", "flux_no_msap"):
            self.cfg.model.use_msap = False
            return FLUXNet(self.cfg.model)
        if arch in ("fluxnet", "flux_net", "fluxnet_v2", "flux_v2"):
            return FLUXNet(self.cfg.model)

        from .baselines import build_baseline_model

        return build_baseline_model(
            arch,
            num_classes=self.cfg.model.num_classes,
            pretrained=getattr(self.cfg.model, "pretrained", False),
        )

    def _build_optimizer(self):
        cfg = self.cfg.training
        # Separate weight decay: don't apply to biases, LayerNorm, BatchNorm params
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas)
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def save_checkpoint(self, path: str, is_best: bool = False,
                        epoch_metric: Optional[dict] = None):
        state = {
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.shadow.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "cfg": self.cfg,
            "epoch": getattr(self, "current_epoch", 0),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "train_history": self.train_history[-50:],
        }
        if epoch_metric:
            state["epoch_metric"] = epoch_metric
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(os.path.dirname(path), "best_model.pt")
            shutil.copy2(path, best_path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        if "ema_state_dict" in state:
            self.ema.shadow.load_state_dict(state["ema_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.start_epoch = state.get("epoch", 0) + 1
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        self.best_val_acc = state.get("best_val_acc", 0.0)
        self.best_epoch = state.get("best_epoch", -1)
        self.train_history = state.get("train_history", [])
        return state

    def _run_train_epoch(self, epoch: int) -> dict:
        """Training epoch with gradient clipping and NaN-safe updates."""
        cfg = self.cfg.training
        tracker = MetricTracker(
            num_classes=self.cfg.model.num_classes,
            class_names=self.cfg.data.class_names)
        self.model.train()
        loader = self.train_loader
        batch_log_interval = max(1, len(loader) // 5)
        t0 = time.time()
        amp_enabled = cfg.use_amp and self.device.type == "cuda"
        grad_clip = cfg.grad_clip
        nan_batches = 0

        # zero_grad at start of epoch (set_to_none is more memory-efficient)
        self.optimizer.zero_grad(set_to_none=True)
        pending_grad_steps = 0
        grad_norm = torch.tensor(float("nan"), device=self.device)

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            with autocast(device_type=self.device.type, enabled=amp_enabled):
                output = self.model(images)
                logits = output["logits"]
                loss = self.criterion(logits, targets)

            # Detect NaN loss — skip this batch but DON'T poison the optimizer
            if not torch.isfinite(loss):
                nan_batches += 1
                print(f"  [WARN] NaN loss at batch {batch_idx} — skipping update")
                self.optimizer.zero_grad(set_to_none=True)
                tracker.update(logits, targets, float("nan"), batch_size)
                continue

            self.scaler.scale(loss).backward()
            pending_grad_steps += 1

            is_accum_boundary = pending_grad_steps >= cfg.grad_accumulation_steps
            is_last_batch = batch_idx == len(loader) - 1
            if pending_grad_steps > 0 and (is_accum_boundary or is_last_batch):
                # Unscale before clipping so clip threshold is in true gradient units
                self.scaler.unscale_(self.optimizer)

                # ★ Gradient clipping — CRITICAL for preventing NaN explosion
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip)

                # Skip optimizer step if gradients are still NaN after unscaling
                did_step = False
                if torch.isfinite(grad_norm):
                    self.scaler.step(self.optimizer)
                    did_step = True
                else:
                    print(f"  [WARN] Non-finite grad norm {grad_norm:.4f} at "
                          f"batch {batch_idx} — skipping optimizer step")
                    nan_batches += 1

                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                pending_grad_steps = 0

                # Update EMA after every valid optimizer step
                if did_step:
                    self.ema.update(self.model)
                    self.global_step += 1

            tracker.update(logits, targets, loss.item(), batch_size)

            if batch_idx % batch_log_interval == 0 or batch_idx == len(loader) - 1:
                partial = tracker.compute()
                gn = grad_norm.item() if torch.isfinite(grad_norm) else float("nan")
                print(f"  E{epoch+1} [{batch_idx+1}/{len(loader)}]  "
                      f"loss={partial['loss']:.4f}  acc={partial['accuracy']:.4f}  "
                      f"gnorm={gn:.3f}")

        epoch_time = time.time() - t0
        metrics = tracker.compute()
        metrics["epoch"] = epoch
        metrics["time_seconds"] = round(epoch_time, 2)
        metrics["throughput"] = round(
            metrics["num_samples"] / max(epoch_time, 1e-6), 1)
        metrics["cm"] = tracker.get_confusion_matrix()
        metrics["nan_batches"] = nan_batches
        return metrics

    def _run_eval_epoch(self, loader: DataLoader, epoch: int,
                        use_ema: bool = True) -> dict:
        """Evaluation epoch — uses EMA weights by default."""
        tracker = MetricTracker(
            num_classes=self.cfg.model.num_classes,
            class_names=self.cfg.data.class_names)
        amp_enabled = self.cfg.training.use_amp and self.device.type == "cuda"
        t0 = time.time()

        ctx = self.ema.apply(self.model) if use_ema else _NoopContext()
        with ctx:
            self.model.eval()
            with torch.no_grad():
                for images, targets in loader:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    with autocast(device_type=self.device.type, enabled=amp_enabled):
                        output = self.model(images)
                        logits = output["logits"]
                        loss = self.criterion(logits, targets)
                    tracker.update(logits, targets, loss.item(), images.size(0))

        epoch_time = time.time() - t0
        metrics = tracker.compute()
        metrics["epoch"] = epoch
        metrics["time_seconds"] = round(epoch_time, 2)
        metrics["throughput"] = round(
            metrics["num_samples"] / max(epoch_time, 1e-6), 1)
        metrics["cm"] = tracker.get_confusion_matrix()
        self._last_eval_tracker = tracker
        return metrics

    def train(self) -> dict:
        cfg = self.cfg.training
        epochs = cfg.epochs
        total_batches = len(self.train_loader)

        print(f"\nTraining FLUX-Net v2 for {epochs} epochs "
              f"({total_batches} batches/epoch)")
        print(f"LR={cfg.lr:.2e}  WD={cfg.weight_decay}  "
              f"Dropout={self.cfg.model.dropout}  GradClip={cfg.grad_clip}")
        print(f"Device: {self.device}  AMP: {cfg.use_amp}  EMA: {cfg.ema_decay}")

        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch
            epoch_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}  LR={epoch_lr:.2e}  "
                  f"Best: {self.best_val_acc:.4f} @ ep{self.best_epoch+1}")
            print("="*60)

            # ── Train ────────────────────────────────────────────────────────
            train_metrics = self._run_train_epoch(epoch)

            # Detect sustained NaN training (model diverged)
            train_loss = train_metrics["loss"]
            if not np.isfinite(train_loss):
                self._nan_streak += 1
                print(f"  [ERROR] NaN train loss for {self._nan_streak} epoch(s)!")
                if self._nan_streak >= 3:
                    print("  [ERROR] Training diverged — stopping.")
                    break
            else:
                self._nan_streak = 0

            # ── Validate (EMA weights) ───────────────────────────────────────
            val_metrics = self._run_eval_epoch(self.val_loader, epoch, use_ema=True)

            # ── Scheduler step ───────────────────────────────────────────────
            if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
                self.scheduler.step()

            # ── Logging ──────────────────────────────────────────────────────
            combined = {f"train_{k}": v for k, v in train_metrics.items()}
            combined.update({f"val_{k}": v for k, v in val_metrics.items()})
            combined["lr"] = epoch_lr
            combined["epoch"] = epoch
            self.train_history.append(combined)
            self._save_history_json()
            self._log_epoch_summary(epoch, train_metrics, val_metrics)

            # ── Early stopping / checkpoint ──────────────────────────────────
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            improved = False
            if val_acc > self.best_val_acc + cfg.early_stopping_min_delta:
                improved = True
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_since_improvement = 0
                if cfg.save_best:
                    self.save_checkpoint(
                        os.path.join(self.cfg.checkpoint_dir, f"epoch_{epoch}.pt"),
                        is_best=True, epoch_metric=combined)
                    print(f"  ★ Best model saved (val_acc={val_acc:.4f})")
            elif val_loss < self.best_val_loss - cfg.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_since_improvement = 0
                if cfg.save_best:
                    self.save_checkpoint(
                        os.path.join(self.cfg.checkpoint_dir, f"epoch_{epoch}.pt"),
                        is_best=True, epoch_metric=combined)
                    print(f"  ★ Best val loss saved (val_loss={val_loss:.4f})")
            else:
                self.epochs_since_improvement += 1

            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                ckpt_path = os.path.join(
                    self.cfg.checkpoint_dir, f"epoch_{epoch}.pt")
                self.save_checkpoint(ckpt_path, epoch_metric=combined)
                print(f"  ✔ Periodic checkpoint saved")

            if cfg.test_every_n_epochs > 0 and \
                    (epoch + 1) % cfg.test_every_n_epochs == 0 and \
                    self.test_loader is not None:
                print("  --- Mid-training test evaluation (EMA weights) ---")
                test_m = self._run_eval_epoch(self.test_loader, epoch, use_ema=True)
                test_entry = {f"test_{k}": v for k, v in test_m.items()}
                test_entry["epoch"] = epoch
                combined.update(test_entry)
                print(f"  Test | loss={test_m['loss']:.4f}  "
                      f"acc={test_m['accuracy']:.4f}  "
                      f"f1={test_m['f1_weighted']:.4f}  "
                      f"auc={test_m['auc_roc']:.4f}")

            if self.epochs_since_improvement >= cfg.early_stopping_patience:
                print(f"\nEarly stopping @ epoch {epoch+1}. "
                      f"Best: ep{self.best_epoch+1} (val_acc={self.best_val_acc:.4f})")
                break

        last_ckpt = os.path.join(self.cfg.checkpoint_dir, "last_model.pt")
        self.save_checkpoint(last_ckpt)
        self._save_final_summary()

        print(f"\nTraining complete. "
              f"Best epoch: {self.best_epoch+1}, "
              f"best val acc: {self.best_val_acc:.4f}")

        return {
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_epochs": epoch + 1,
        }

    def evaluate(self, test_loader: Optional[DataLoader] = None,
                 checkpoint_path: Optional[str] = None) -> dict:
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        loader = test_loader or self.test_loader
        if loader is None:
            raise ValueError("No test loader available.")

        print(f"\n{'='*60}")
        print("Running held-out test evaluation (EMA weights)...")
        metrics = self._run_eval_epoch(loader, epoch=-1, use_ema=True)
        metrics["confusion_matrix"] = metrics.pop("cm")
        metrics["classification_report"] = \
            self._last_eval_tracker.get_classification_report()

        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        for k, v in metrics.items():
            if k not in ("confusion_matrix", "classification_report"):
                print(f"  {k}: {v}")
        print(f"\nConfusion Matrix:")
        for row in metrics["confusion_matrix"]:
            print(f"  {row}")

        safe = {}
        for k, v in metrics.items():
            if k == "classification_report":
                safe[k] = str(v)
            elif isinstance(v, (np.integer,)):
                safe[k] = int(v)
            elif isinstance(v, (np.floating,)):
                safe[k] = float(v)
            elif isinstance(v, np.ndarray):
                safe[k] = v.tolist()
            else:
                try:
                    json.dumps(v)
                    safe[k] = v
                except (TypeError, ValueError):
                    safe[k] = str(v)

        metrics_path = os.path.join(self.cfg.metrics_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(safe, f, indent=2, cls=EpochHistoryEncoder)
        print(f"\nTest metrics saved to {metrics_path}")
        return metrics

    def _save_history_json(self):
        safe = []
        for entry in self.train_history:
            clean = {}
            for k, v in entry.items():
                if k in ("cm", "train_cm", "val_cm"):
                    clean[k] = v
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                elif isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                else:
                    clean[k] = v
            safe.append(clean)
        path = os.path.join(self.cfg.log_dir, "train_history.json")
        with open(path, "w") as f:
            json.dump(safe, f, indent=2, cls=EpochHistoryEncoder)

    def _log_epoch_summary(self, epoch: int, train_m: dict, val_m: dict):
        t_loss = train_m["loss"]
        v_loss = val_m["loss"]
        t_acc = train_m["accuracy"]
        v_acc = val_m["accuracy"]
        gap_acc = t_acc - v_acc
        gap_loss = (v_loss - t_loss) if (np.isfinite(v_loss) and np.isfinite(t_loss)) else float("nan")

        nan_b = train_m.get("nan_batches", 0)
        nan_warn = f"  ⚠ {nan_b} NaN batch(es) skipped" if nan_b > 0 else ""

        print(f"\n  {'─'*52}")
        print(f"  Train — Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  "
              f"F1: {train_m['f1_weighted']:.4f}")
        print(f"  Val   — Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  "
              f"F1: {val_m['f1_weighted']:.4f}  AUC: {val_m['auc_roc']:.4f}")
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}  "
              f"Gap — Acc: {gap_acc:+.4f}  Loss: {gap_loss:+.4f}")
        if nan_warn:
            print(nan_warn)
        print(f"  {'─'*52}")

        # Per-class F1
        for name in self.cfg.data.class_names:
            tf1 = train_m.get(f"{name}_f1", 0)
            vf1 = val_m.get(f"{name}_f1", 0)
            print(f"    {name:12s} | train_f1={tf1:.4f}  val_f1={vf1:.4f}")

    def _save_final_summary(self):
        summary = {
            "best_val_loss": float(self.best_val_loss),
            "best_val_acc": float(self.best_val_acc),
            "best_epoch": int(self.best_epoch),
            "total_epochs_trained": len(self.train_history),
            "config": self.cfg.to_dict(),
        }
        path = os.path.join(self.cfg.metrics_dir, "training_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, cls=EpochHistoryEncoder)


class _NoopContext:
    """No-op context manager for when EMA is not used."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
