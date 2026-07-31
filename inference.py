"""
Inference — dự đoán cho 1 bệnh nhân mới (4 ảnh DICOM đã crop).
"""
 
import torch
import numpy as np

from pathlib import Path
from typing import Dict, Optional,Tuple
 
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
 
    ckpt = torch.load(checkpoint_path, map_location=device,weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
 
    print(f"[Inference] Loaded: {Path(checkpoint_path).name}")
    print(f"  Test AUC: {ckpt.get('test_metrics', {}).get('auc', '?')}")
 
    return model
 
@torch.no_grad()
def predict_patient(
    model:       MammoTransformer,
    image_paths: Dict[str, str],   # {"L_MLO": path.dicom, "L_CC": path.dicom, ...}
    image_size: Tuple[int,int] = (1856, 704),
    threshold:   float = 0.7,
    device:      Optional[torch.device] = None,
):
    """
    Dự đoán cho 1 bệnh nhân.
 
    Args:
        image_paths : dict keys = L_MLO, L_CC, R_MLO, R_CC → path file .dicom
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
 
    transform = ValTransform()
    images = {}
 
    for key in VIEW_KEYS:
        path = image_paths.get(key, "")
        if not path:
            raise ValueError(f"Thiếu ảnh cho view '{key}' — inference cần đủ 4 view.")
        tensor = transform(path)                      # → (3, H, W) float normalized
        images[key] = tensor.unsqueeze(0).to(device)   # → (1, 3, H, W)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type):
        logit = model(images)
        probs = torch.sigmoid(logit).squeeze(0).cpu().numpy()
    return probs
 
# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = load_model(
        checkpoint_path="outputs/mammo_transformer_v1/final_model.pt",
        cfg=cfg,
        device=device,
    )
 
    result = predict_patient(
        model=model,
        image_paths={
            "L_MLO": "images_cropped/ff797ae566e0c252a105853faab6e7cd/323f8bf1926e56ec02e2bc2c5fe751da.dicom",
            "L_CC":  "images_cropped/ff797ae566e0c252a105853faab6e7cd/836071b1c4b0cc30f9b1bebcde81288f.dicom",
            "R_MLO": "images_cropped/ff797ae566e0c252a105853faab6e7cd/4fa150db6ca406124fe62200880a9ee9.dicom",
            "R_CC":  "images_cropped/ff797ae566e0c252a105853faab6e7cd/d9caaef549cd1ea35fed601ed7dc2304.dicom",
        },
        image_size=cfg.model.backbone_img_size,
        threshold=0.7,
    )
 
    print("\n── Kết quả dự đoán ──")
    for k, v in result.items():
        print(f"  {k}: {v}")
 