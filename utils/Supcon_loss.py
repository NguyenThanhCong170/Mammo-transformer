import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewNTXentLoss(nn.Module):
    """
    SupCon / NT-Xent với "nhãn" là ID bệnh nhân.

    4 view của cùng 1 bệnh nhân là positive của nhau;
    mọi view của bệnh nhân khác là negative.

    LƯU Ý: số negative tỉ lệ với batch_size. batch_size=2 chỉ cho 4 negative
    → tín hiệu contrastive rất yếu. Tăng batch_size_phase1 tối đa theo VRAM.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (batch_size, num_views, feature_dim) — output của ProjectionHead.
        """
        device = features.device
        batch_size, num_views, feature_dim = features.shape

        if batch_size < 2:
            # Không có negative nào → loss vô nghĩa (và log(0) → nan)
            raise ValueError(
                f"Contrastive loss cần batch_size >= 2, đang nhận {batch_size}. "
                "Kiểm tra drop_last=True và batch_size_phase1."
            )

        # (B*V, D), các view cùng bệnh nhân nằm liền kề
        z = features.reshape(batch_size * num_views, feature_dim)
        z = F.normalize(z.float(), dim=1)      # ép float32 — ổn định hơn dưới AMP

        sim_matrix = torch.matmul(z, z.T) / self.temperature       # (B*V, B*V)

        # Pseudo-label theo bệnh nhân: [0,0,0,0, 1,1,1,1, ...]
        labels = torch.arange(batch_size, device=device).repeat_interleave(num_views)
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
