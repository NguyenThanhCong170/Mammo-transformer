"""
Inference — dự đoán cho 1 bệnh nhân mới (4 ảnh PNG đã crop).
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from models.mammo_transformer import MammoTransformer
from data.augmentation import ValTransform
from configs.config import Config


VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]


# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
def load_model(checkpoint_path: str, cfg: Config, device: torch.device) -> MammoTransformer:
    model = MammoTransformer(
        backbone_name=cfg.model.backbone_name,
        backbone_pretrained=False,    # Không cần load ImageNet khi inference
        backbone_img_size=cfg.model.backbone_img_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        attn_dropout=0.0,             # Tắt dropout khi inference
        ffn_dropout=0.0,
        num_ipsi_layers=cfg.model.num_ipsi_layers,
        num_bilateral_layers=cfg.model.num_bilateral_layers,
        mlp_hidden_dim=cfg.model.mlp_hidden_dim,
        mlp_dropout=0.0,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[Inference] Loaded: {Path(checkpoint_path).name}")
    print(f"  Test AUC: {ckpt.get('test_metrics', {}).get('auc', '?')}")

    return model


# ──────────────────────────────────────────────
# PNG loader
# ──────────────────────────────────────────────
def _load_png_as_uint8(path: str) -> np.ndarray:
    """
    Đọc file PNG → numpy uint8 (H, W).
    """
    pixel = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if pixel is None:
        raise FileNotFoundError(f"Không thể đọc ảnh tại: {path}")
    
    return pixel.astype(np.uint8)


# ──────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_patient(
    model:       MammoTransformer,
    image_paths: Dict[str, str],   # {"L_MLO": path.png, "L_CC": path.png, ...}
    image_size:  int   = 256,
    threshold:   float = 0.5,
    device:      Optional[torch.device] = None,
) -> Dict:
    """
    Dự đoán cho 1 bệnh nhân.

    Args:
        image_paths : dict keys = L_MLO, L_CC, R_MLO, R_CC → path file .png
        threshold   : optimal threshold từ val set (lấy từ history.json)

    Returns:
        {
          "probability" : float  — P(BI-RADS 4/5)
          "prediction"  : str    — "Yes (BI-RADS 4/5)" hoặc "No (BI-RADS 1-3)"
          "label"       : int    — 1 hoặc 0
          "logit"       : float
          "threshold_used" : float
        }
    """
    if device is None:
        device = next(model.parameters()).device

    transform = ValTransform(image_size=image_size)  # resize + normalize, không augment
    images = {}

    for key in VIEW_KEYS:
        path = image_paths.get(key, "")
        if path and Path(path).exists():
            pixel = _load_png_as_uint8(path)        # (H, W) uint8
        else:
            print(f"  ⚠️  Missing view {key}, using blank image")
            pixel = np.zeros((image_size, image_size), dtype=np.uint8)

        tensor = transform(pixel)                   # → (3, H, W) float normalized
        images[key] = tensor.unsqueeze(0).to(device)  # → (1, 3, H, W)

    with torch.cuda.amp.autocast():
        logit = model(images).item()

    prob = torch.sigmoid(torch.tensor(logit)).item()
    pred = "Yes (BI-RADS 4/5)" if prob >= threshold else "No (BI-RADS 1-3)"

    return {
        "probability":    round(prob, 4),
        "prediction":     pred,
        "label":          int(prob >= threshold),
        "logit":          round(logit, 4),
        "threshold_used": threshold,
    }


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lưu ý: Thay đổi đường dẫn model cho phù hợp với máy của bạn
    model = load_model(
        checkpoint_path="outputs/mammo_transformer_v1/final_model.pt",
        cfg=cfg,
        device=device,
    )

    # Khai báo đường dẫn đến 4 file PNG của bệnh nhân
    result = predict_patient(
        model=model,
        image_paths={
            "L_MLO": "path/to/patient/L_MLO.png",
            "L_CC":  "path/to/patient/L_CC.png",
            "R_MLO": "path/to/patient/R_MLO.png",
            "R_CC":  "path/to/patient/R_CC.png",
        },
        image_size=cfg.model.backbone_img_size,
        threshold=0.45,
    )

    print("\n── Kết quả dự đoán ──")
    for k, v in result.items():
        print(f"  {k}: {v}")