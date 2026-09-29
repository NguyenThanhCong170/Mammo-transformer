from collections import Counter
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewNTXentLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.1,
        view_groups: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.temperature = temperature

        if view_groups is None:
            self.view_groups = None
            self.num_groups = 1
        else:
            raw = [int(g) for g in view_groups]
            # Chuẩn hoá nhãn nhóm về 0..G-1 để công thức label bên dưới luôn
            # đúng, bất kể người gọi truyền vào số gì.
            uniq = sorted(set(raw))
            self.view_groups = [uniq.index(g) for g in raw]
            self.num_groups = len(uniq)

            counts = Counter(self.view_groups)
            thieu = [g for g, c in counts.items() if c < 2]
            if thieu:
                raise ValueError(
                    f"view_groups={list(view_groups)}: nhom {thieu} chi co 1 view nen "
                    f"khong co positive nao. Moi nhom can it nhat 2 view."
                )

    def extra_repr(self) -> str:
        mode = "patient-level" if self.view_groups is None else f"ipsilateral{self.view_groups}"
        return f"temperature={self.temperature}, mode={mode}"

    def _build_labels(self, batch_size: int, num_views: int, device) -> torch.Tensor:
        """Pseudo-label cho từng hàng của ma trận (B*V, D)."""
        patient = torch.arange(batch_size, device=device).repeat_interleave(num_views)

        if self.view_groups is None:
            return patient                          # [0,0,0,0, 1,1,1,1, ...]

        if len(self.view_groups) != num_views:
            raise ValueError(
                f"view_groups co {len(self.view_groups)} phan tu nhung features co "
                f"{num_views} view. Hai cai phai khop va cung thu tu VIEW_KEYS."
            )
        group = torch.tensor(self.view_groups, device=device).repeat(batch_size)
        # (benh nhan, ben vu) → nhan duy nhat. Vd 4 view / 2 nhom:
        #   [0,0,1,1, 2,2,3,3, ...]
        return patient * self.num_groups + group

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (batch_size, num_views, feature_dim) — output của ProjectionHead.
        """
        device = features.device
        batch_size, num_views, feature_dim = features.shape

        if batch_size < 2:
            # Khong co negative tu benh nhan khac → tin hieu contrastive qua yeu.
            raise ValueError(
                f"Contrastive loss can batch_size >= 2, dang nhan {batch_size}. "
                "Kiem tra drop_last=True va batch_size_phase1."
            )

        # (B*V, D), các view cùng bệnh nhân nằm liền kề theo thứ tự VIEW_KEYS
        z = features.reshape(batch_size * num_views, feature_dim)
        z = F.normalize(z.float(), dim=1)      # fp32 — nhớ gọi loss NGOÀI autocast

        sim_matrix = torch.matmul(z, z.T) / self.temperature       # (B*V, B*V)

        labels = self._build_labels(batch_size, num_views, device)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Bỏ đường chéo (self-similarity)
        eye = torch.eye(batch_size * num_views, device=device)
        logits_mask = 1.0 - eye
        mask = mask * logits_mask

        # Numerical stability
        logits_max, _ = torch.max(sim_matrix - eye * 1e9, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_per_anchor = mask.sum(1).clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_anchor

        return -mean_log_prob_pos.mean()


# ──────────────────────────────────────────────────────────────────────────
# Helper: suy ra view_groups từ VIEW_KEYS
# ──────────────────────────────────────────────────────────────────────────
def build_view_groups(view_keys: Sequence[str], mode: str) -> Optional[list]:

    mode = str(mode).strip().lower()
    if mode == "patient":
        return None
    if mode != "ipsilateral":
        raise ValueError(f"mode phai la 'patient' hoac 'ipsilateral', nhan '{mode}'.")

    lats = [str(k).split("_")[0].upper() for k in view_keys]     # "L_MLO" → "L"
    uniq = sorted(set(lats))
    if len(uniq) < 2:
        raise ValueError(
            f"Chi tim thay 1 ben vu trong VIEW_KEYS={list(view_keys)} → "
            f"ipsilateral-only se giong het patient-level."
        )
    return [uniq.index(l) for l in lats]


# ══════════════════════════════════════════════════════════════════════════
# SupCon cho MULTI-LABEL — dùng ở train_phase2.py (train chung với FocalLoss)
# ══════════════════════════════════════════════════════════════════════════
class MultiLabelSupConLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.1,
        exclude_classes: Optional[Sequence[int]] = (0,),
        base_temperature: Optional[float] = None,
    ):
        super().__init__()
        self.temperature = temperature
        # Khosla et al. chia thêm cho base_temperature để độ lớn gradient không
        # đổi khi chỉnh temperature. Mặc định = temperature (tức hệ số 1).
        self.base_temperature = base_temperature or temperature
        self.exclude_classes = tuple(exclude_classes) if exclude_classes else ()

    def extra_repr(self) -> str:
        return (f"temperature={self.temperature}, "
                f"exclude_classes={self.exclude_classes}")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (N, D) — chưa cần chuẩn hoá, hàm tự normalize.
        labels  : (N, C) multi-hot float/bool.
        Trả về scalar. Batch không có cặp positive nào → trả 0 (có grad).
        """
        if features.ndim != 2:
            raise ValueError(f"features phải là (N, D), nhận {tuple(features.shape)}")
        if labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"Số mẫu lệch: features {features.shape[0]} vs labels {labels.shape[0]}")

        device = features.device
        N = features.shape[0]

        y = labels.float().clone()
        for c in self.exclude_classes:
            if 0 <= c < y.shape[1]:
                y[:, c] = 0.0

        # Jaccard: |giao| / |hợp|,  |hợp| = |y_i| + |y_j| - |giao|
        inter = y @ y.T                                     # (N, N)
        card = y.sum(1, keepdim=True)                       # (N, 1)
        union = card + card.T - inter
        w = torch.where(union > 0, inter / union.clamp(min=1e-12),
                        torch.zeros_like(inter))

        eye = torch.eye(N, device=device)
        w = w * (1.0 - eye)                                 # bỏ chính nó

        z = F.normalize(features.float(), dim=1)
        sim = (z @ z.T) / self.temperature

        # log-softmax trên mọi cột trừ đường chéo
        sim_masked = sim - eye * 1e9
        logits = sim - sim_masked.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits) * (1.0 - eye)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        w_sum = w.sum(1)                                    # (N,)
        valid = w_sum > 0                                   # anchor có positive
        if not valid.any():
            # Không cặp nào chung nhãn — trả 0 nhưng vẫn nối vào graph để
            # optimizer không vấp "grad is None" ở các step như vậy.
            return (features.float() * 0.0).sum()

        mean_log_prob_pos = (w * log_prob).sum(1)[valid] / w_sum[valid]
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()
