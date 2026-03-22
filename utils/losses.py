"""
Loss functions và metrics cho binary mammography classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import numpy as np
from typing import Dict


# ──────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss.

    FL(p) = -alpha * (1-p)^gamma * log(p)

    alpha: weight cho positive class (nên set ~0.75 vì imbalanced)
    gamma: focusing parameter (2.0 là standard)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B,) — raw logits (chưa qua sigmoid)
        targets: (B,) — float 0.0 hoặc 1.0
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────
# Metrics Calculator
# ──────────────────────────────────────────────
class MetricsCalculator:
    """
    Tính toán tất cả metrics cho binary classification.
    Dùng threshold 0.5 mặc định, nhưng optimal threshold được tính từ val set.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.all_logits  = []
        self.all_targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        self.all_logits.append(logits.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        logits  = torch.cat(self.all_logits).numpy()
        targets = torch.cat(self.all_targets).numpy().astype(int)
        probs   = 1 / (1 + np.exp(-logits))  # sigmoid

        preds = (probs >= self.threshold).astype(int)

        # AUC — không cần threshold
        try:
            auc = roc_auc_score(targets, probs)
        except ValueError:
            auc = 0.0

        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn + 1e-8)  # Recall (quan trọng nhất cho cancer)
        specificity = tn / (tn + fp + 1e-8)
        ppv         = tp / (tp + fp + 1e-8)  # Precision
        npv         = tn / (tn + fn + 1e-8)
        f1          = 2 * (ppv * sensitivity) / (ppv + sensitivity + 1e-8)
        accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        return {
            "auc":         round(float(auc), 4),
            "accuracy":    round(float(accuracy), 4),
            "sensitivity": round(float(sensitivity), 4),  # Recall / TPR
            "specificity": round(float(specificity), 4),  # TNR
            "ppv":         round(float(ppv), 4),          # Precision
            "npv":         round(float(npv), 4),
            "f1":          round(float(f1), 4),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }

    def find_optimal_threshold(self) -> float:
        """
        Tìm threshold tối ưu theo Youden's J (sensitivity + specificity - 1).
        Chạy trên validation set sau mỗi epoch.
        """
        from sklearn.metrics import roc_curve

        logits  = torch.cat(self.all_logits).numpy()
        targets = torch.cat(self.all_targets).numpy().astype(int)
        probs   = 1 / (1 + np.exp(-logits))

        fpr, tpr, thresholds = roc_curve(targets, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])

    def print_report(self, split: str = "Val"):
        metrics = self.compute()
        print(f"\n{'='*50}")
        print(f"  {split} Metrics")
        print(f"{'='*50}")
        print(f"  AUC:         {metrics['auc']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}  ← (TP/P, quan trọng nhất)")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  PPV:         {metrics['ppv']:.4f}")
        print(f"  NPV:         {metrics['npv']:.4f}")
        print(f"  F1:          {metrics['f1']:.4f}")
        print(f"  Confusion:   TP={metrics['tp']} TN={metrics['tn']} "
              f"FP={metrics['fp']} FN={metrics['fn']}")
        print(f"{'='*50}\n")
        return metrics
