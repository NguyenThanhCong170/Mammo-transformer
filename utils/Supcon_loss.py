import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewNTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        features: Tensor chứa các đặc trưng đầu ra từ Projection Head.
        Shape yêu cầu: [batch_size, num_views, feature_dim] 
        Ví dụ với VinDr-Mammo: [B, 4, 128] (B bệnh nhân, mỗi bệnh nhân 4 ảnh, vector 128 chiều)
        """
        device = features.device
        batch_size, num_views, feature_dim = features.shape
        
        # 1. Trải phẳng tensor thành [B * 4, feature_dim]
        # Các ảnh của cùng 1 bệnh nhân sẽ nằm liền kề nhau: P1_1, P1_2, P1_3, P1_4, P2_1, P2_2,...
        z = features.view(batch_size * num_views, feature_dim)
        z = F.normalize(z, dim=1)
        
        # 2. Tính ma trận cosine similarity: shape [B*4, B*4]
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # 3. Tạo nhãn giả (pseudo-labels) dựa trên ID bệnh nhân
        # labels sẽ có dạng: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, ...]
        labels = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, num_views).reshape(-1)
        
        # Tạo mask cho các cặp positive: mask[i, j] = 1 nếu i và j cùng chung nhãn bệnh nhân
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # 4. Loại bỏ self-similarity (đường chéo chính bằng 1)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * num_views, device=device)
        mask = mask * logits_mask # Chỉ giữ lại các positive thực sự (không tính chính nó)
        
        # 5. Trick ổn định tính toán (Numerical stability)
        # Trừ đi giá trị max trên mỗi hàng để tránh tràn số khi tính exp
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # 6. Tính xác suất (log_prob)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 7. Tính mean của log-likelihood trên các positive samples
        # mask.sum(1) ở đây luôn bằng 3 (vì 4 ảnh, trừ chính nó đi còn 3)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss là giá trị âm của log-likelihood trung bình
        loss = -mean_log_prob_pos.mean()
        
        return loss