"""
Loss functions và metrics cho binary mammography classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    multilabel_confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import numpy as np
from typing import Dict, Optional

from configs.config import Config
cfg = Config()

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

    def __init__(self, alpha: float, gamma: float, reduction: str):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B,) — raw logits (chưa qua sigmoid)
        targets: (B,) — float 0.0 hoặc 1.0
        """
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction = "none")
        p = torch.sigmoid(logits)
        p_t = p*targets + (1-p)*(1-targets)
        a = self.alpha.to(logits.device)
        a_t = a*targets + (1-a)*(1-targets)
        loss = a_t *(1-p_t).pow(self.gamma)*ce
        return loss.mean()


# ──────────────────────────────────────────────
# Metrics Calculator
# ──────────────────────────────────────────────
class MultiLabelMetricsCalculator:
    """
    Args:
        num_classes: số lượng class (vd 4: background, mass, calcification, asymmetry)
        class_names: tên hiển thị cho từng class, dùng khi in báo cáo
        threshold: threshold mặc định áp dụng cho MỌI class nếu không truyền
                    per-class threshold vào compute()
    """
 
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[list[str]] = None,
        threshold: float = 0.7,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.threshold = threshold
        self.reset()
 
    def reset(self):
        self.all_logits: list[torch.Tensor] = []
        self.all_targets: list[torch.Tensor] = []
 
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """logits, targets: (B, num_classes)"""
        self.all_logits.append(logits.detach().cpu())
        self.all_targets.append(targets.detach().cpu())
 
    def _get_probs_targets(self):
        logits = torch.cat(self.all_logits).numpy()            # (N, C)
        targets = torch.cat(self.all_targets).numpy().astype(int)  # (N, C)
        probs = 1 / (1 + np.exp(-logits))                      # sigmoid, (N, C)
        return probs, targets
 
    def compute(self, thresholds: Optional[list[float]] = None) -> Dict[str, float]:
        """
        thresholds: list threshold riêng cho từng class (độ dài = num_classes).
                    Nếu None, dùng self.threshold cho tất cả class.
        """
        probs, targets = self._get_probs_targets()
        thresholds = thresholds or [self.threshold] * self.num_classes
        preds = np.stack(
            [(probs[:, c] >= thresholds[c]).astype(int) for c in range(self.num_classes)],
            axis=1,
        )  # (N, C)
 
        # multilabel_confusion_matrix trả về (C, 2, 2): mỗi class 1 ma trận [[tn,fp],[fn,tp]]
        mcm = multilabel_confusion_matrix(targets, preds)
 
        per_class = {}
        for c in range(self.num_classes):
            tn, fp, fn, tp = mcm[c].ravel()
 
            sensitivity = tp / (tp + fn + 1e-8)   # Recall / TPR
            specificity = tn / (tn + fp + 1e-8)   # TNR
            ppv = tp / (tp + fp + 1e-8)           # Precision
            npv = tn / (tn + fn + 1e-8)
            f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity + 1e-8)
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
 
            # AUC-ROC và AUC-PR (Average Precision) — chỉ tính được nếu class có
            # cả positive lẫn negative trong tập hiện tại
            y_true_c = targets[:, c]
            y_prob_c = probs[:, c]
            if len(np.unique(y_true_c)) < 2:
                auc, ap = 0.0, 0.0
            else:
                auc = roc_auc_score(y_true_c, y_prob_c)
                ap = average_precision_score(y_true_c, y_prob_c)
 
            name = self.class_names[c]
            per_class[name] = {
                "auc": round(float(auc), 4),
                "ap": round(float(ap), 4),          # AUC-PR, quan trọng với class hiếm
                "accuracy": round(float(accuracy), 4),
                "sensitivity": round(float(sensitivity), 4),
                "specificity": round(float(specificity), 4),
                "ppv": round(float(ppv), 4),
                "npv": round(float(npv), 4),
                "f1": round(float(f1), 4),
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            }
 
        # Macro-average: trung bình cộng đơn giản qua các class (mọi class trọng số như nhau,
        # không bị class phổ biến lấn át — quan trọng khi dữ liệu mất cân bằng)
        macro = {}
        metric_keys = ["auc", "ap", "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1"]
        for key in metric_keys:
            macro[f"macro_{key}"] = round(
                float(np.mean([per_class[name][key] for name in self.class_names])), 4
            )
 
        return {"per_class": per_class, "macro": macro}
 
    def find_optimal_thresholds(self) -> list[float]:
        """
        Tìm threshold tối ưu RIÊNG cho từng class theo Youden's J
        (sensitivity + specificity - 1). Chạy trên validation set sau mỗi epoch.
        Trả về list threshold, độ dài = num_classes.
        """
        probs, targets = self._get_probs_targets()
        best_thresholds = []
        for c in range(self.num_classes):
            y_true_c = targets[:, c]
            if len(np.unique(y_true_c)) < 2:
                best_thresholds.append(self.threshold)  # fallback nếu class không đủ 2 lớp
                continue
            fpr, tpr, thr = roc_curve(y_true_c, probs[:, c])
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_thresholds.append(float(thr[best_idx]))
        return best_thresholds
 
    def print_report(self, split: str = "Val", thresholds: Optional[list[float]] = None):
        result = self.compute(thresholds)
        per_class, macro = result["per_class"], result["macro"]
 
        print(f"\n{'=' * 70}")
        print(f"  {split} Metrics (multi-label, {self.num_classes} classes)")
        print(f"{'=' * 70}")
        header = f"{'Class':<15}{'AUC':>8}{'AP':>8}{'Sens':>8}{'Spec':>8}{'PPV':>8}{'F1':>8}"
        print(header)
        print("-" * 70)
        for name in self.class_names:
            m = per_class[name]
            print(f"{name:<15}{m['auc']:>8.4f}{m['ap']:>8.4f}{m['sensitivity']:>8.4f}"
                  f"{m['specificity']:>8.4f}{m['ppv']:>8.4f}{m['f1']:>8.4f}")
        print("-" * 70)
        print(f"{'MACRO AVG':<15}{macro['macro_auc']:>8.4f}{macro['macro_ap']:>8.4f}"
              f"{macro['macro_sensitivity']:>8.4f}{macro['macro_specificity']:>8.4f}"
              f"{macro['macro_ppv']:>8.4f}{macro['macro_f1']:>8.4f}")
        print(f"{'=' * 70}\n")
        return result