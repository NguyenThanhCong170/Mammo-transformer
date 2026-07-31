"""
Inference — dự đoán multi-label cho 1 bệnh nhân (4 ảnh PNG đã crop/resize).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from configs.config import Config
from data.augmentation import ValTransform
from models.mammo_transformer import MammoTransformer

VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]
DEFAULT_CLASS_NAMES = ["no_finding", "mass", "calcification", "asymmetry"]


# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
def load_model(checkpoint_path: str, cfg: Config, device: torch.device):
    model = MammoTransformer(
        backbone_name=cfg.model.backbone_name,
        backbone_pretrained=False,        # weight sẽ đến từ checkpoint
        backbone_img_size=cfg.model.backbone_img_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        num_ipsi_layers=cfg.model.num_ipsi_layers,
        num_bilateral_layers=cfg.model.num_bilateral_layers,
        mlp_hidden_dim=cfg.model.mlp_hidden_dim,
        mlp_dropout=0.0,
        num_classes=cfg.model.num_classes,
        token_grid=cfg.model.token_grid,
        ffn_expansion=cfg.model.ffn_expansion,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    thresholds = ckpt.get("thresholds") or [0.5] * cfg.model.num_classes
    class_names = ckpt.get("class_names") or DEFAULT_CLASS_NAMES

    print(f"[Inference] Đã nạp: {Path(checkpoint_path).name}")
    macro = (ckpt.get("test_metrics") or {}).get("macro", {})
    if macro:
        print(f"  Test macro-AP : {macro.get('macro_ap', '?')}")
        print(f"  Test macro-AUC: {macro.get('macro_auc', '?')}")
    print(f"  Thresholds     : {[round(t, 3) for t in thresholds]}")

    return model, thresholds, class_names


# ──────────────────────────────────────────────
# Load 1 ảnh → Tensor (3, H, W)
# ──────────────────────────────────────────────
def load_image(path: str, image_size: Tuple[int, int], transform: ValTransform) -> torch.Tensor:
    """image_size = (H, W)"""
    h, w = image_size
    img = Image.open(path).convert("L")
    if img.size != (w, h):                    # PIL dùng (W, H)
        img = img.resize((w, h), Image.BILINEAR)
    return transform(np.array(img, dtype=np.uint8))    # ← ValTransform cần np.ndarray, KHÔNG phải path


# ──────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_patient(
    model: MammoTransformer,
    image_paths: Dict[str, str],
    image_size: Tuple[int, int] = (1856, 704),
    thresholds: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    image_paths : {"L_MLO": ..., "L_CC": ..., "R_MLO": ..., "R_CC": ...} → path .png
    thresholds  : per-class threshold lấy từ val set (nằm trong checkpoint)

    Returns:
        {
          "probabilities": {class_name: float},
          "predictions"  : {class_name: 0/1},
          "positive"     : [class_name, ...],
          "thresholds"   : {class_name: float},
        }
    """
    if device is None:
        device = next(model.parameters()).device
    class_names = class_names or DEFAULT_CLASS_NAMES
    transform = ValTransform()

    images = {}
    for key in VIEW_KEYS:
        path = image_paths.get(key, "")
        if not path:
            raise ValueError(f"Thiếu ảnh cho view '{key}' — inference cần đủ 4 view.")
        images[key] = load_image(path, image_size, transform).unsqueeze(0).to(device)

    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(images)

    probs = torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()
    if thresholds is None:
        thresholds = [0.5] * len(probs)

    preds = (probs >= np.asarray(thresholds)).astype(int)

    return {
        "probabilities": {n: round(float(p), 4) for n, p in zip(class_names, probs)},
        "predictions": {n: int(v) for n, v in zip(class_names, preds)},
        "positive": [n for n, v in zip(class_names, preds) if v == 1],
        "thresholds": {n: round(float(t), 4) for n, t in zip(class_names, thresholds)},
    }


# ──────────────────────────────────────────────
# Example
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(cfg.train.output_dir) / cfg.train.experiment_name / "final_model.pt"
    model, thresholds, class_names = load_model(str(ckpt_path), cfg, device)

    pid = "ff797ae566e0c252a105853faab6e7cd"
    root = Path(cfg.data.data_root) / pid
    result = predict_patient(
        model=model,
        image_paths={
            "L_MLO": str(root / "323f8bf1926e56ec02e2bc2c5fe751da.png"),
            "L_CC":  str(root / "836071b1c4b0cc30f9b1bebcde81288f.png"),
            "R_MLO": str(root / "4fa150db6ca406124fe62200880a9ee9.png"),
            "R_CC":  str(root / "d9caaef549cd1ea35fed601ed7dc2304.png"),
        },
        image_size=cfg.data.image_size,
        thresholds=thresholds,
        class_names=class_names,
        device=device,
    )

    print("\n── Kết quả dự đoán ──")
    for name in class_names:
        p = result["probabilities"][name]
        t = result["thresholds"][name]
        mark = "✔" if result["predictions"][name] else " "
        print(f"  [{mark}] {name:<16} p={p:.4f}  (threshold {t:.3f})")
    print(f"\n  Kết luận: {result['positive'] or ['không phát hiện bất thường']}")
