"""
MammoTransformer — Multi-view Breast Cancer Classification

Kiến trúc:
    1. Shared Swin-V2 backbone → TOKEN MAP (B, N, D) cho mỗi view
       (KHÔNG global-pool trước attention — nếu pool thành 1 vector thì
        softmax của cross-attention chạy trên seq_len=1 và luôn trả về 1.0,
        tức là attention biến thành một phép linear vô nghĩa.)
    2. View Embedding + Positional Embedding
    3. Ipsilateral Cross-Attention (L-MLO ↔ L-CC, R-MLO ↔ R-CC)
    4. Bilateral Cross-Attention  (Left ↔ Right)
    5. Attention Pooling → MLP Classifier → multi-label logits (B, num_classes)
"""

from typing import Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]


# ──────────────────────────────────────────────
# 1. Swin-V2 Backbone Wrapper
# ──────────────────────────────────────────────
class SwinV2Backbone(nn.Module):
    """
    Wrapper quanh timm Swin-V2.

    - forward(x)         → (B, D)     : global-avg-pooled feature (dùng cho phase 1 contrastive)
    - forward_tokens(x)  → (B, N, D)  : token map đã pool về lưới token_grid (dùng cho attention)
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        img_size: Tuple[int, int],              # (H, W)
        token_grid: Optional[Tuple[int, int]] = (8, 4),
        grad_checkpointing: bool = True,
    ):
        super().__init__()

        # ── strict_img_size=False là BẮT BUỘC với ảnh mammo ──
        # timm precompute attn_mask ngay trong __init__ tại feat_size danh định,
        # và window_partition() yêu cầu feat_size chia hết cho window_size (=16).
        # Với (1856, 704): stage 2 cho 232x88 mà 232/16 = 14.5 → RuntimeError.
        # strict_img_size=False ⇒ dynamic_mask=True ⇒ timm bỏ precompute và
        # tính mask lúc runtime; _attn() tự pad H/W về bội của window_size.
        common = dict(
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
            global_pool="avg",
        )
        try:
            self.backbone = timm.create_model(model_name, strict_img_size=False, **common)
        except TypeError:
            # timm quá cũ, chưa có strict_img_size → yêu cầu size chia hết thủ công
            print("[Backbone] timm không hỗ trợ strict_img_size — hãy nâng cấp: pip install -U timm")
            self.backbone = timm.create_model(model_name, **common)

        self.token_grid = token_grid
        self.out_dim = self.backbone.num_features
        # alias cho tiện — nhiều đoạn code cũ gọi .num_features
        self.num_features = self.out_dim
        self.set_grad_checkpointing(grad_checkpointing)

    def set_grad_checkpointing(self, enable: bool = True):
        try:
            self.backbone.set_grad_checkpointing(enable)
        except Exception:
            pass

    # ── (B, 3, H, W) → (B, D)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ── (B, 3, H, W) → (B, N, D)
    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)

        if feats.ndim == 4:
            # timm Swin-V2 trả NHWC: (B, H', W', C). Một số model trả NCHW.
            if feats.shape[-1] == self.out_dim:
                feats = feats.permute(0, 3, 1, 2).contiguous()   # → (B, C, H', W')
            if self.token_grid is not None:
                feats = F.adaptive_avg_pool2d(feats, self.token_grid)
            return feats.flatten(2).transpose(1, 2)              # (B, N, C)

        if feats.ndim == 3:
            # (B, N, C) — không biết lưới không gian → pool 1D
            if self.token_grid is not None:
                n_tok = self.token_grid[0] * self.token_grid[1]
                feats = F.adaptive_avg_pool1d(feats.transpose(1, 2), n_tok).transpose(1, 2)
            return feats

        raise RuntimeError(f"Không hiểu output shape của backbone: {tuple(feats.shape)}")

    def num_tokens(self, img_size: Tuple[int, int]) -> int:
        """Số token sau khi pool. Chạy 1 forward giả để lấy chính xác."""
        if self.token_grid is not None:
            return self.token_grid[0] * self.token_grid[1]
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *img_size, device=next(self.parameters()).device)
            n = self.forward_tokens(dummy).shape[1]
        self.train(was_training)
        return n

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ──────────────────────────────────────────────
# 2. View + Positional Embedding
# ──────────────────────────────────────────────
class ViewEmbedding(nn.Module):
    """
    4 embedding riêng cho L_MLO, L_CC, R_MLO, R_CC (broadcast lên mọi token)
    + 1 positional embedding dùng chung cho các vị trí token trong lưới.
    """

    def __init__(self, embed_dim: int, num_tokens: int):
        super().__init__()
        self.view_emb = nn.Parameter(torch.zeros(len(VIEW_KEYS), 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.view_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for i, key in enumerate(VIEW_KEYS):
            x = features[key]                       # (B, N, D)
            pos = self.pos_emb[:, : x.shape[1], :]  # phòng khi N lệch
            out[key] = x + self.view_emb[i].unsqueeze(0) + pos
        return out


# ──────────────────────────────────────────────
# 3. Cross-Attention Block
# ──────────────────────────────────────────────
class CrossAttentionBlock(nn.Module):
    """
    Query từ view A (B, Nq, D), Key/Value từ view B (B, Nk, D).
    Pre-norm + residual + FFN, giống Transformer decoder layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        hidden = embed_dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        query = query + attn_out
        query = query + self.ffn(self.norm_ff(query))
        return query


# ──────────────────────────────────────────────
# 4. Attention Pooling (token seq → 1 vector)
# ──────────────────────────────────────────────
class AttentionPool(nn.Module):
    """Learnable query token attend lên toàn bộ token → (B, D)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(tokens)
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out.squeeze(1)


# ──────────────────────────────────────────────
# 5. Ipsilateral Fusion (MLO ↔ CC, cùng bên)
# ──────────────────────────────────────────────
class IpsilateralFusion(nn.Module):
    """
    MLO attends to CC, CC attends to MLO, lặp num_layers lần.
    Output: chuỗi token nối lại (B, 2N, D) — giữ nguyên token để tầng
    bilateral vẫn còn thông tin không gian để so sánh.
    """

    def __init__(self, embed_dim, num_heads, num_layers, attn_dropout, ffn_dropout, ffn_expansion=4):
        super().__init__()
        self.mlo_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, attn_dropout, ffn_dropout, ffn_expansion)
            for _ in range(num_layers)
        ])
        self.cc_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, attn_dropout, ffn_dropout, ffn_expansion)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, mlo: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
        for mlo_layer, cc_layer in zip(self.mlo_layers, self.cc_layers):
            # dùng bản cũ của mlo làm context cho cc → tránh phụ thuộc thứ tự
            mlo_prev = mlo
            mlo = mlo_layer(query=mlo, context=cc)
            cc = cc_layer(query=cc, context=mlo_prev)
        return self.norm(torch.cat([mlo, cc], dim=1))   # (B, 2N, D)


# ──────────────────────────────────────────────
# 6. Bilateral Fusion (Left ↔ Right)
# ──────────────────────────────────────────────
class BilateralFusion(nn.Module):
    """
    So sánh Left ↔ Right ở mức token, rồi attention-pool mỗi bên
    thành 1 vector và merge → (B, D).
    """

    def __init__(self, embed_dim, num_heads, num_layers, attn_dropout, ffn_dropout, ffn_expansion=4):
        super().__init__()
        self.left_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, attn_dropout, ffn_dropout, ffn_expansion)
            for _ in range(num_layers)
        ])
        self.right_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, attn_dropout, ffn_dropout, ffn_expansion)
            for _ in range(num_layers)
        ])
        self.pool_left = AttentionPool(embed_dim, num_heads)
        self.pool_right = AttentionPool(embed_dim, num_heads)

        self.merge = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        for l_layer, r_layer in zip(self.left_layers, self.right_layers):
            left_prev = left
            left = l_layer(query=left, context=right)
            right = r_layer(query=right, context=left_prev)

        l_vec = self.pool_left(left)     # (B, D)
        r_vec = self.pool_right(right)   # (B, D)
        return self.merge(torch.cat([l_vec, r_vec], dim=-1))   # (B, D)


# ──────────────────────────────────────────────
# 7. MLP Classifier
# ──────────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, num_classes)


# ──────────────────────────────────────────────
# 8. Full MammoTransformer
# ──────────────────────────────────────────────
class MammoTransformer(nn.Module):
    """
    Input : Dict[str, Tensor(B, 3, H, W)] với keys = L_MLO, L_CC, R_MLO, R_CC
    Output: logits (B, num_classes) — raw logits, dùng sigmoid cho multi-label
    """

    def __init__(
        self,
        backbone_name: str,
        backbone_pretrained: bool,
        backbone_img_size: Tuple[int, int],
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        ffn_dropout: float,
        num_ipsi_layers: int,
        num_bilateral_layers: int,
        mlp_hidden_dim: int,
        mlp_dropout: float,
        num_classes: int = 4,
        token_grid: Optional[Tuple[int, int]] = (8, 4),
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = SwinV2Backbone(
            model_name=backbone_name,
            pretrained=backbone_pretrained,
            img_size=backbone_img_size,
            token_grid=token_grid,
        )
        backbone_out_dim = self.backbone.out_dim

        self.input_proj = (
            nn.Linear(backbone_out_dim, embed_dim)
            if backbone_out_dim != embed_dim else nn.Identity()
        )

        num_tokens = self.backbone.num_tokens(backbone_img_size)
        self.view_embedding = ViewEmbedding(embed_dim, num_tokens)

        self.left_fusion = IpsilateralFusion(
            embed_dim, num_heads, num_ipsi_layers, attn_dropout, ffn_dropout, ffn_expansion)
        self.right_fusion = IpsilateralFusion(
            embed_dim, num_heads, num_ipsi_layers, attn_dropout, ffn_dropout, ffn_expansion)

        self.bilateral_fusion = BilateralFusion(
            embed_dim, num_heads, num_bilateral_layers, attn_dropout, ffn_dropout, ffn_expansion)

        self.classifier = MLPClassifier(embed_dim, mlp_hidden_dim, mlp_dropout, num_classes)

        self._init_non_backbone_weights()

    def _init_non_backbone_weights(self):
        for name, module in self.named_modules():
            if name.startswith("backbone"):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Step 1: token features từng view qua shared backbone
        feats = {}
        for key in VIEW_KEYS:
            t = self.backbone.forward_tokens(images[key])   # (B, N, D_bb)
            feats[key] = self.input_proj(t)                 # (B, N, D)

        # Step 2: view + positional embedding
        feats = self.view_embedding(feats)

        # Step 3: ipsilateral fusion → (B, 2N, D) mỗi bên
        left = self.left_fusion(mlo=feats["L_MLO"], cc=feats["L_CC"])
        right = self.right_fusion(mlo=feats["R_MLO"], cc=feats["R_CC"])

        # Step 4: bilateral fusion + attention pooling → (B, D)
        global_feat = self.bilateral_fusion(left=left, right=right)

        # Step 5: classify
        return self.classifier(global_feat)                 # (B, num_classes)

    # ── Backbone control ──
    def freeze_backbone(self):
        self.backbone.freeze()
        self.backbone.set_grad_checkpointing(False)   # vô nghĩa khi không có grad
        print("[Model] Backbone frozen (grad-checkpointing off).")

    def unfreeze_backbone(self):
        self.backbone.unfreeze()
        self.backbone.set_grad_checkpointing(True)
        print("[Model] Backbone unfrozen.")

    def load_backbone_weights(self, ckpt_path: str, device="cpu") -> None:
        """Nạp backbone đã contrastive-pretrain ở phase 1."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("backbone_state_dict", ckpt)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        print(f"[Model] Loaded phase-1 backbone từ {ckpt_path}")
        if missing:
            print(f"  missing keys   : {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")

    def train(self, mode: bool = True):
        """Nếu backbone bị freeze thì luôn giữ nó ở eval mode (tắt BN/dropout)."""
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def get_param_groups(self, lr: float, backbone_lr_multiplier: float = 0.1):
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        other_params = [p for p in self.parameters()
                        if id(p) not in backbone_ids and p.requires_grad]
        groups = [{"params": other_params, "lr": lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr * backbone_lr_multiplier})
        return groups

    def count_parameters(self) -> Dict[str, int]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone": backbone_params,
            "other": total_params - backbone_params,
            "total": total_params,
            "trainable": trainable,
        }
