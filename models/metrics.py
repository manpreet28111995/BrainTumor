import numpy as np
import json
import torch
from typing import Dict, Tuple
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)


class MetricTracker:
    def __init__(self, num_classes: int = 4, class_names: Tuple[str, ...] = None):
        self.num_classes = num_classes
        self.class_names = class_names or tuple(f"c{i}" for i in range(num_classes))
        self.reset()

    def reset(self):
        self.all_preds, self.all_targets, self.all_probs = [], [], []
        self.running_loss = 0.0
        self.num_samples = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor,
               loss: float, batch_size: int):
        # Always compute in float32 regardless of AMP dtype
        probs = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
        preds = logits.detach().float().argmax(dim=-1).cpu().numpy()
        tgt = targets.detach().cpu().numpy()

        # Guard NaN from logits (shouldn't happen after ssm/fusion fix, but safety net)
        probs = np.nan_to_num(probs, nan=1.0 / self.num_classes)

        # Explicitly re-normalize to sum=1.0 (float16→float32 drift can break sklearn)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        probs = probs / row_sums

        self.all_preds.extend(preds.tolist())
        self.all_targets.extend(tgt.tolist())
        self.all_probs.extend(probs.tolist())
        # Guard NaN loss before accumulating
        if np.isfinite(loss):
            self.running_loss += loss * batch_size
        self.num_samples += batch_size

    def compute(self) -> Dict[str, float]:
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        probs = np.array(self.all_probs)

        acc = accuracy_score(targets, preds)
        f1_w = f1_score(targets, preds, average="weighted", zero_division=0)
        f1_m = f1_score(targets, preds, average="macro", zero_division=0)
        prec_w = precision_score(targets, preds, average="weighted", zero_division=0)
        rec_w = recall_score(targets, preds, average="weighted", zero_division=0)

        m = {
            "loss": float(self.running_loss / max(self.num_samples, 1)),
            "accuracy": float(acc),
            "error_rate": float(1.0 - acc),
            "f1_weighted": float(f1_w),
            "f1_macro": float(f1_m),
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
        }

        per_f1 = f1_score(targets, preds, average=None, zero_division=0)
        per_prec = precision_score(targets, preds, average=None, zero_division=0)
        per_rec = recall_score(targets, preds, average=None, zero_division=0)

        for i, name in enumerate(self.class_names):
            if i < len(per_f1):
                m[f"{name}_f1"] = float(per_f1[i])
                m[f"{name}_precision"] = float(per_prec[i])
                m[f"{name}_recall"] = float(per_rec[i])

        # Robust AUC: only compute when all classes are present
        unique_targets = np.unique(targets)
        if len(unique_targets) == self.num_classes:
            try:
                # Final normalization check before passing to sklearn
                probs_safe = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                row_sums = probs_safe.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums <= 0, 1.0, row_sums)
                probs_safe = probs_safe / row_sums
                m["auc_roc"] = float(roc_auc_score(
                    targets, probs_safe, multi_class="ovr", average="macro"))
            except Exception as e:
                print(f"  [AUC] skipped: {e}")
                m["auc_roc"] = 0.0
        else:
            m["auc_roc"] = 0.0

        m["num_samples"] = int(self.num_samples)
        return m

    def get_confusion_matrix(self):
        cm = confusion_matrix(self.all_targets, self.all_preds,
                              labels=list(range(self.num_classes)))
        return cm.tolist()

    def get_classification_report(self):
        return classification_report(self.all_targets, self.all_preds,
                                     target_names=list(self.class_names),
                                     digits=4, zero_division=0)
