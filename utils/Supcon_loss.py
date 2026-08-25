from collections import Counter
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewNTXentLoss(nn.Module):
    """
    SupCon / NT-Xent trên nhiều view của cùng bệnh nhân.

    ── Hai chế độ ghép positive ──────────────────────────────────────────

    `view_groups=None`  →  PATIENT-LEVEL (hành vi cũ)
        Cả 4 view của một bệnh nhân là positive của nhau. 3 positive/anchor.

    `view_groups=[0,0,1,1]`  →  IPSILATERAL-ONLY
        Chỉ view cùng BÊN vú mới là positive: L_MLO ↔ L_CC, R_MLO ↔ R_CC.
        Vú đối bên của CHÍNH bệnh nhân đó trở thành NEGATIVE. 1 positive/anchor.

    ── Vì sao cần ipsilateral-only ───────────────────────────────────────
    Chế độ patient-level huấn luyện backbone làm biểu diễn của vú trái và vú
    phải cùng một người trở nên GIỐNG NHAU. Nhưng lớp 'asymmetry' được định
    nghĩa bằng đúng cái KHÁC NHAU giữa hai bên — pretext task đang tối ưu
    ngược lại với nhiệm vụ đích.

    Đo trên VinDr-Mammo (test, 1000 bệnh nhân): backbone patient-level thua cả
    khởi tạo ImageNet ở toàn bộ 4 lớp, và tụt mạnh nhất đúng ở asymmetry
    (AUC 0.5418 vs 0.6735 — về sát mức ngẫu nhiên).

    Ipsilateral-only giữ được tính nhất quán MLO↔CC cùng bên, đồng thời biến
    vú đối bên thành HARD NEGATIVE — ép mô hình mã hoá khác biệt trái/phải
    thay vì xoá nó đi.

    ── Lưu ý ─────────────────────────────────────────────────────────────
    Số negative tỉ lệ với batch_size. Ở chế độ ipsilateral mỗi anchor có
    (4B - 2) negative thay vì (4B - 4), và 2 trong số đó là hard negative đến
    từ chính bệnh nhân ấy. Vẫn nên tăng batch_size_phase1 tối đa theo VRAM.
    """

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
    """
    mode = "patient"      → None (mọi view cùng bệnh nhân là positive)
    mode = "ipsilateral"  → nhóm theo tiền tố bên vú lấy từ VIEW_KEYS

    Suy ra từ chính VIEW_KEYS thay vì hard-code [0,0,1,1]: nếu thứ tự view
    trong dataset đổi thì grouping tự đổi theo, không lệch âm thầm.
    """
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
