"""
MammoTransformer — Multi-view Breast Cancer Classification

Kiến trúc:
    1. Shared Swin-V2-Base backbone (feature extractor)
    2. View Embedding (mỗi view có embedding riêng)
    3. Ipsilateral Cross-Attention (L-MLO ↔ L-CC, R-MLO ↔ R-CC)
    4. Bilateral Cross-Attention  (Left_fused ↔ Right_fused)
    5. MLP Classifier → Binary output
"""

import math
from typing import Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
cfg = Config()

# ──────────────────────────────────────────────
# 1. Swin-V2 Backbone Wrapper
# ──────────────────────────────────────────────
class SwinV2Backbone(nn.Module):
    """
    Wrapper quanh timm Swin-V2-Base.
    Output: CLS-like global feature vector (B, embed_dim)
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        img_width: int,
        img_height: int
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=(img_height, img_width),
            num_classes=0,        # Xóa head classifier
            global_pool="avg",    # Global average pool → (B, C)
        )
        self.backbone.set_grad_checkpointing(True)
        self.out_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, out_dim)"""
        return self.backbone(x)

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ──────────────────────────────────────────────
# 2. View Embedding
# ──────────────────────────────────────────────
class ViewEmbedding(nn.Module):
    """
    Học 4 embedding vectors riêng cho L_MLO, L_CC, R_MLO, R_CC.
    Cộng vào feature sau backbone để mô hình biết đang xử lý view nào.
    """

    VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(4, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        features: Dict[view_key → Tensor(B, D)]
        Trả về: cùng dict + view embedding
        """
        out = {}
        for i, key in enumerate(self.VIEW_KEYS):
            idx = torch.tensor(i, device=features[key].device)
            view_emb = self.embeddings(idx).unsqueeze(0)  # (1, D)
            out[key] = features[key] + view_emb
        return out


# ──────────────────────────────────────────────
# 3. Cross-Attention Block
# ──────────────────────────────────────────────
class CrossAttentionBlock(nn.Module):
    """
    Query từ view A, Key+Value từ view B.
    Output: view A được enriched bởi context của view B.

    Sau đó dùng residual + FFN (giống Transformer encoder layer).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,   # (B, D) — view cần được update
        context: torch.Tensor, # (B, D) — view cung cấp context
    ) -> torch.Tensor:         # (B, D)

        # Unsqueeze seq_len=1 cho MultiheadAttention
        q  = self.norm_q(query).unsqueeze(1)    # (B, 1, D)
        kv = self.norm_kv(context).unsqueeze(1) # (B, 1, D)

        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        attn_out = attn_out.squeeze(1)  # (B, D)

        # Residual
        query = query + attn_out

        # FFN + residual
        query = query + self.ffn(self.norm_ff(query))

        return query


# ──────────────────────────────────────────────
# 4. Ipsilateral Fusion Module
# ──────────────────────────────────────────────
class IpsilateralFusion(nn.Module):
    """
    Fuse MLO ↔ CC cho cùng bên (L hoặc R).
    MLO attends to CC, CC attends to MLO → concat → project.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()

        self.mlo_attn_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.cc_attn_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Merge MLO + CC → single representation
        self.merge = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        mlo: torch.Tensor,  # (B, D)
        cc:  torch.Tensor,  # (B, D)
    ) -> torch.Tensor:      # (B, D)

        for mlo_layer, cc_layer in zip(self.mlo_attn_layers, self.cc_attn_layers):
            mlo = mlo_layer(query=mlo, context=cc)
            cc  = cc_layer(query=cc,  context=mlo)

        fused = torch.cat([mlo, cc], dim=-1)  # (B, 2D)
        return self.merge(fused)              # (B, D)


# ──────────────────────────────────────────────
# 5. Bilateral Fusion Module
# ──────────────────────────────────────────────
class BilateralFusion(nn.Module):
    """
    So sánh Left ↔ Right (sau khi đã ipsilateral fused).
    Tương tự IpsilateralFusion nhưng ở tầng cao hơn.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()

        self.left_attn_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.right_attn_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Global feature = concat left + right
        self.merge = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        left:  torch.Tensor,  # (B, D)
        right: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:        # (B, D)

        for l_layer, r_layer in zip(self.left_attn_layers, self.right_attn_layers):
            left  = l_layer(query=left,  context=right)
            right = r_layer(query=right, context=left)

        fused = torch.cat([left, right], dim=-1)
        return self.merge(fused)


# ──────────────────────────────────────────────
# 6. MLP Classifier
# ──────────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, num_class: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2,num_class),  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,)


# ──────────────────────────────────────────────
# 7. Full MammoTransformer
# ──────────────────────────────────────────────
class MammoTransformer(nn.Module):
    """
    End-to-end multi-view mammography classifier.

    Forward input:
        images: Dict[str, Tensor(B,3,H,W)]
                keys = ["L_MLO", "L_CC", "R_MLO", "R_CC"]

    Forward output:
        logits: Tensor(B,)  — raw logit, dùng sigmoid để lấy probability
    """

    def __init__(
        self,
        backbone_name:       str,
        backbone_pretrained: bool,
        backbone_img_size:   Tuple[int,int],
        embed_dim:           int,
        num_heads:           int,
        attn_dropout:        float,
        ffn_dropout:         float,
        num_ipsi_layers:     int,
        num_bilateral_layers:int,
        mlp_hidden_dim:      int,
        mlp_dropout:         float,
    ):
        super().__init__()

        # ── Backbone
        self.backbone = SwinV2Backbone(
            model_name=backbone_name,
            pretrained=backbone_pretrained,
            img_size=backbone_img_size,
        )
        backbone_out_dim = self.backbone.out_dim

        # ── Project backbone output → embed_dim (nếu khác)
        self.input_proj = (
            nn.Linear(backbone_out_dim, embed_dim)
            if backbone_out_dim != embed_dim
            else nn.Identity()
        )

        # ── View Embedding
        self.view_embedding = ViewEmbedding(embed_dim)

        # ── Stage 1: Ipsilateral Fusion
        self.left_fusion  = IpsilateralFusion(embed_dim, num_heads, num_ipsi_layers, attn_dropout)
        self.right_fusion = IpsilateralFusion(embed_dim, num_heads, num_ipsi_layers, attn_dropout)

        # ── Stage 2: Bilateral Fusion
        self.bilateral_fusion = BilateralFusion(embed_dim, num_heads, num_bilateral_layers, attn_dropout)

        # ── Classifier
        self.classifier = MLPClassifier(embed_dim, mlp_hidden_dim, mlp_dropout)

        self._init_non_backbone_weights()

    def _init_non_backbone_weights(self):
        """Xavier init cho các layer ngoài backbone."""
        for name, module in self.named_modules():
            if "backbone" in name:
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        images: {"L_MLO": (B,3,H,W), "L_CC": ..., "R_MLO": ..., "R_CC": ...}
        → logits: (B,)
        """
        # ── Step 1: Extract features từng view qua shared backbone
        feats = {}
        for key in ["L_MLO", "L_CC", "R_MLO", "R_CC"]:
            f = self.backbone(images[key])     # (B, backbone_out_dim)
            f = self.input_proj(f)             # (B, embed_dim)
            feats[key] = f

        # ── Step 2: Cộng view embedding
        feats = self.view_embedding(feats)

        # ── Step 3: Ipsilateral Fusion
        left_feat  = self.left_fusion( mlo=feats["L_MLO"], cc=feats["L_CC"])   # (B, D)
        right_feat = self.right_fusion(mlo=feats["R_MLO"], cc=feats["R_CC"])   # (B, D)

        # ── Step 4: Bilateral Fusion
        global_feat = self.bilateral_fusion(left=left_feat, right=right_feat)   # (B, D)

        # ── Step 5: Classify
        logits = self.classifier(global_feat)  # (B,)

        return logits
    
    def freeze_backbone(self):
        self.backbone.freeze()
        print("[Model] Backbone frozen.")

    def unfreeze_backbone(self):
        self.backbone.unfreeze()
        print("[Model] Backbone unfrozen.")

    def get_param_groups(self, lr: float, backbone_lr_multiplier):
        """
        Trả về param groups để backbone có LR nhỏ hơn.
        """
        backbone_params = list(self.backbone.parameters())
        other_params = [p for n, p in self.named_parameters()
                        if not any(p is bp for bp in backbone_params)]
        return [
            {"params": backbone_params, "lr": 0},
            {"params": other_params,    "lr": lr},
        ]

    def count_parameters(self) -> Dict[str, int]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        total_params    = sum(p.numel() for p in self.parameters())
        return {
            "backbone": backbone_params,
            "other":    total_params - backbone_params,
            "total":    total_params,
        }
