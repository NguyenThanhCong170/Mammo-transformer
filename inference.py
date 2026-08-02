"""
INFERENCE / EVALUATION — Mammo-Transformer

Hai chế độ, dùng chung một file:

1) Đánh giá cả một split (mặc định = test) — chạy dataloader, in bảng metric,
   ghi JSON + CSV per-patient + biểu đồ ROC/PR:

       python inference.py
       python inference.py --split val
       python inference.py --ckpt outputs/mammo_transformer_v1/epoch012_macro_ap0.5123.pt
       python inference.py --tune-thresholds-on-val     # chốt threshold từ val rồi áp lên test

2) Dự đoán cho MỘT bệnh nhân (ảnh lấy tự động từ CSV theo patient_id):

       python inference.py --patient ff797ae566e0c252a105853faab6e7cd

Checkpoint: mặc định tự tìm `final_model.pt` trong outputs/<experiment_name>/,
nếu không có thì lấy file epoch***_macro_ap*** có điểm cao nhất. Ghi đè bằng --ckpt.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from configs.config import Config, resolve_path
from data.augmentation import ValTransform
from data.dataset import build_dataloaders
from models.mammo_transformer import MammoTransformer
from utils.losses import FocalLoss, MultiLabelMetricsCalculator

VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]
DEFAULT_CLASS_NAMES = ["no_finding", "mass", "calcification", "asymmetry"]


# ══════════════════════════════════════════════
# Checkpoint
# ══════════════════════════════════════════════
def find_checkpoint(cfg: Config, explicit: Optional[str] = None) -> Path:
    """Thứ tự ưu tiên: --ckpt > final_model.pt > epoch***_macro_ap*** cao điểm nhất."""
    if explicit:
        p = resolve_path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Không thấy checkpoint: {p}")
        return p

    out_dir = resolve_path(cfg.train.output_dir) / cfg.train.experiment_name
    final = out_dir / "final_model.pt"
    if final.exists():
        return final

    candidates = []
    for p in out_dir.glob("epoch*_macro_ap*.pt"):
        m = re.search(r"macro_ap([\d.]+)\.pt$", p.name)
        if m:
            candidates.append((float(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint nào trong {out_dir}.\n"
            f"Hãy chạy train_phase2.py trước, hoặc truyền --ckpt <đường_dẫn>."
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    print(f"[Inference] Không có final_model.pt → dùng checkpoint điểm cao nhất: {best.name}")
    return best


def load_model(checkpoint_path: str, cfg: Config, device: torch.device):
    """Dựng model theo config rồi nạp weight. Trả về (model, thresholds, class_names)."""
    model = MammoTransformer(
        backbone_name=cfg.model.backbone_name,
        backbone_pretrained=False,        # weight đến từ checkpoint, khỏi tải ImageNet
        backbone_img_size=cfg.model.backbone_img_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        attn_dropout=0.0,                 # eval → tắt hết dropout
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
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Inference] ⚠ Checkpoint thiếu {len(missing)} key, ví dụ: {missing[:3]}")
    if unexpected:
        print(f"[Inference] ⚠ Checkpoint thừa {len(unexpected)} key, ví dụ: {unexpected[:3]}")
    model.eval()

    thresholds = ckpt.get("thresholds") or [0.5] * cfg.model.num_classes
    class_names = ckpt.get("class_names") or DEFAULT_CLASS_NAMES

    print(f"\n[Inference] Checkpoint : {Path(checkpoint_path).name}")
    if "epoch" in ckpt:
        print(f"[Inference] Epoch      : {ckpt['epoch']}")
    macro = (ckpt.get("test_metrics") or ckpt.get("metrics") or {}).get("macro", {})
    if macro:
        print(f"[Inference] Đã ghi sẵn : macro-AP {macro.get('macro_ap', '?')} | "
              f"macro-AUC {macro.get('macro_auc', '?')}")
    print(f"[Inference] Thresholds : {[round(float(t), 3) for t in thresholds]}")
    return model, thresholds, class_names


# ══════════════════════════════════════════════
# Chạy 1 dataloader → prob + label + patient_id
# ══════════════════════════════════════════════
@torch.no_grad()
def run_loader(model, loader, device, criterion, amp_dtype, use_amp,
               num_classes: int, class_names: List[str], tag: str = "Test"):
    calc = MultiLabelMetricsCalculator(num_classes=num_classes, class_names=class_names)
    patient_ids: List[str] = []
    total_loss, n_batches = 0.0, 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1
        calc.update(logits.float(), labels)
        patient_ids.extend(batch["patient_id"])

        if (step + 1) % 10 == 0 or (step + 1) == len(loader):
            print(f"  [{tag}] batch {step + 1}/{len(loader)} | "
                  f"loss {total_loss / n_batches:.4f}   ", end="\r")

    print()
    elapsed = time.time() - t0
    probs, targets = calc._get_probs_targets()
    avg_loss = total_loss / max(1, n_batches)
    print(f"  [{tag}] Xong {len(patient_ids)} bệnh nhân trong {elapsed:.0f}s "
          f"({elapsed / max(1, len(patient_ids)) * 1000:.0f} ms/bệnh nhân)")
    return calc, probs, targets, patient_ids, avg_loss


# ══════════════════════════════════════════════
# Xuất kết quả
# ══════════════════════════════════════════════
def save_predictions_csv(path: Path, patient_ids, probs, targets, preds, class_names):
    """CSV per-patient, sắp xếp ca sai nhiều nhãn nhất lên đầu để soi lỗi."""
    data = {"patient_id": patient_ids}
    for c, name in enumerate(class_names):
        data[f"prob_{name}"] = np.round(probs[:, c], 5)
    for c, name in enumerate(class_names):
        data[f"pred_{name}"] = preds[:, c]
    for c, name in enumerate(class_names):
        data[f"true_{name}"] = targets[:, c]

    df = pd.DataFrame(data)
    df["n_wrong"] = (preds != targets).sum(axis=1)
    df["exact_match"] = (df["n_wrong"] == 0).astype(int)
    df.sort_values(["n_wrong", "patient_id"], ascending=[False, True], inplace=True)
    df.to_csv(path, index=False)
    return df


def plot_curves(probs, targets, class_names, out_dir: Path, split: str):
    """Vẽ ROC và Precision-Recall cho từng class. Trả về list file đã ghi."""
    try:
        import matplotlib
        matplotlib.use("Agg")             # không cần màn hình — chạy được trên server
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Inference] ⚠ Chưa cài matplotlib → bỏ qua biểu đồ. "
              "Cài bằng: pip install matplotlib")
        return []

    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    written = []
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)
        for c, name in enumerate(class_names):
            y_true, y_prob = targets[:, c], probs[:, c]
            if len(np.unique(y_true)) < 2:          # class không đủ 2 lớp → bỏ qua
                continue
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                ax.plot(fpr, tpr, lw=1.8,
                        label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.3f})")
            else:
                prec, rec, _ = precision_recall_curve(y_true, y_prob)
                ax.plot(rec, prec, lw=1.8,
                        label=f"{name} (AP={average_precision_score(y_true, y_prob):.3f})")

        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Ngẫu nhiên")
            ax.set_xlabel("False Positive Rate (1 - Specificity)")
            ax.set_ylabel("True Positive Rate (Sensitivity)")
            ax.set_title(f"ROC — {split} set")
        else:
            # Baseline của PR curve = tỉ lệ dương tính của class đó, không phải 0.5
            for c in range(len(class_names)):
                ax.axhline(float(targets[:, c].mean()), ls=":", lw=0.8, alpha=0.35)
            ax.set_xlabel("Recall (Sensitivity)")
            ax.set_ylabel("Precision (PPV)")
            ax.set_title(f"Precision-Recall — {split} set\n"
                         f"(đường chấm = tỉ lệ dương tính nền của mỗi class)")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right" if kind == "roc" else "upper right", fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{kind}_curves_{split.lower()}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


# ══════════════════════════════════════════════
# Chế độ 1 — đánh giá cả split
# ══════════════════════════════════════════════
def evaluate(cfg: Config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32
    print(f"[Inference] Device: {device} | AMP: {use_amp}")

    ckpt_path = find_checkpoint(cfg, args.ckpt)
    model, thresholds, class_names = load_model(str(ckpt_path), cfg, device)
    num_classes = cfg.data.num_classes

    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=str(resolve_path(cfg.data.csv_path)),
        image_size=cfg.data.image_size,
        batch_size=args.batch_size or cfg.train.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else cfg.data.num_workers,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,          # PHẢI trùng seed lúc train, nếu không val/train lẫn vào nhau
        aug_level=cfg.data.aug_level,
        num_classes=num_classes,
        persistent_workers=False,    # chạy một lần rồi thoát, không cần giữ worker
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    criterion = FocalLoss(
        alpha=cfg.train.focal_alpha,
        gamma=cfg.train.focal_gamma,
        reduction=cfg.train.focal_reduction,
    ).to(device)

    # ── Threshold: lấy từ checkpoint / cố định / dò lại trên val (không đụng test → không leakage)
    thr_source = "checkpoint"
    if args.threshold is not None:
        thresholds = [args.threshold] * num_classes
        thr_source = f"cố định {args.threshold}"
    elif args.tune_thresholds_on_val and args.split != "val":
        print("\n[Inference] Dò lại threshold trên VAL set...")
        val_calc, *_ = run_loader(model, val_loader, device, criterion, amp_dtype, use_amp,
                                  num_classes, class_names, tag="Val")
        thresholds = val_calc.find_optimal_thresholds()
        thr_source = "dò lại trên val (Youden's J)"
        print(f"[Inference] Threshold mới: {[round(t, 3) for t in thresholds]}")

    print(f"\n[Inference] Chạy {args.split.upper()} set...")
    calc, probs, targets, patient_ids, avg_loss = run_loader(
        model, loader, device, criterion, amp_dtype, use_amp,
        num_classes, class_names, tag=args.split.capitalize())

    if args.split == "val" and args.tune_thresholds_on_val and args.threshold is None:
        thresholds = calc.find_optimal_thresholds()
        thr_source = "dò trực tiếp trên val (Youden's J)"

    # ── Metric
    result = calc.print_report(args.split.capitalize(), thresholds=thresholds)

    preds = np.stack([(probs[:, c] >= thresholds[c]).astype(int)
                      for c in range(num_classes)], axis=1)
    exact = float((preds == targets).all(axis=1).mean())
    hamming = float((preds != targets).mean())

    result.update({
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "n_patients": int(len(patient_ids)),
        "loss": round(float(avg_loss), 6),
        "exact_match_ratio": round(exact, 4),
        "hamming_loss": round(hamming, 4),
        "threshold_source": thr_source,
        "thresholds": {n: round(float(t), 4) for n, t in zip(class_names, thresholds)},
        "class_prevalence": {n: round(float(targets[:, c].mean()), 4)
                             for c, n in enumerate(class_names)},
    })

    print(f"  Loss (focal)      : {avg_loss:.4f}")
    print(f"  Exact-match ratio : {exact:.4f}   (đúng CẢ {num_classes} nhãn cùng lúc)")
    print(f"  Hamming loss      : {hamming:.4f}   (tỉ lệ nhãn bị sai)")
    print(f"  Threshold lấy từ  : {thr_source}\n")

    # ── Ghi file
    if args.no_save:
        return result

    out_dir = (resolve_path(args.out_dir) if args.out_dir else
               resolve_path(cfg.train.output_dir) / cfg.train.experiment_name / f"eval_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"metrics_{args.split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / f"predictions_{args.split}.csv"
    save_predictions_csv(csv_path, patient_ids, probs, targets, preds, class_names)

    plots = plot_curves(probs, targets, class_names, out_dir, args.split)

    print("[Inference] Đã ghi:")
    print(f"  {json_path}")
    print(f"  {csv_path}   (sắp theo số nhãn sai giảm dần — ca khó nằm trên đầu)")
    for p in plots:
        print(f"  {p}")
    return result


# ══════════════════════════════════════════════
# Chế độ 2 — dự đoán cho 1 bệnh nhân
# ══════════════════════════════════════════════
def load_image(path: str, image_size: Tuple[int, int], transform: ValTransform) -> torch.Tensor:
    """image_size = (H, W)"""
    h, w = image_size
    img = Image.open(resolve_path(path)).convert("L")
    if img.size != (w, h):                     # PIL dùng (W, H)
        img = img.resize((w, h), Image.BILINEAR)
    return transform(np.array(img, dtype=np.uint8))    # ValTransform nhận np.ndarray, KHÔNG phải path


@torch.no_grad()
def predict_patient(
    model: MammoTransformer,
    image_paths: Dict[str, str],
    image_size: Tuple[int, int] = (928, 352),
    thresholds: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    image_paths : {"L_MLO": ..., "L_CC": ..., "R_MLO": ..., "R_CC": ...} → path .png
    thresholds  : per-class threshold chốt từ val set (nằm sẵn trong checkpoint)

    Returns:
        {"probabilities": {...}, "predictions": {...}, "positive": [...], "thresholds": {...}}
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


def lookup_patient_images(cfg: Config, patient_id: str) -> Dict[str, str]:
    """Tra CSV lấy đủ 4 view của 1 bệnh nhân — khỏi gõ tay từng đường dẫn."""
    df = pd.read_csv(resolve_path(cfg.data.csv_path))
    rows = df[df["patient_id"] == patient_id]
    if rows.empty:
        raise ValueError(f"Không thấy patient_id '{patient_id}' trong {cfg.data.csv_path}")

    paths = {}
    for _, r in rows.iterrows():
        key = f"{str(r['laterality']).strip().upper()}_{str(r['view_position']).strip().upper()}"
        paths[key] = str(r["image_path"])
    missing = [k for k in VIEW_KEYS if k not in paths]
    if missing:
        raise ValueError(f"Bệnh nhân {patient_id} thiếu view: {missing}")
    return {k: paths[k] for k in VIEW_KEYS}


def predict_single(cfg: Config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = find_checkpoint(cfg, args.ckpt)
    model, thresholds, class_names = load_model(str(ckpt_path), cfg, device)

    result = predict_patient(
        model=model,
        image_paths=lookup_patient_images(cfg, args.patient),
        image_size=cfg.data.image_size,
        thresholds=thresholds,
        class_names=class_names,
        device=device,
    )

    print(f"\n── Bệnh nhân {args.patient} ──")
    for name in class_names:
        p = result["probabilities"][name]
        t = result["thresholds"][name]
        mark = "✔" if result["predictions"][name] else " "
        print(f"  [{mark}] {name:<16} p={p:.4f}  (threshold {t:.3f})")
    print(f"\n  Kết luận: {result['positive'] or ['không phát hiện bất thường']}\n")
    return result


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Đánh giá Mammo-Transformer trên một split, hoặc predict cho 1 bệnh nhân.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Đường dẫn checkpoint. Bỏ trống = tự tìm final_model.pt / epoch tốt nhất.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                   help="Split để đánh giá (mặc định: test).")
    p.add_argument("--batch-size", type=int, default=None, help="Mặc định lấy từ config.")
    p.add_argument("--num-workers", type=int, default=None, help="Mặc định lấy từ config.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Dùng CHUNG một threshold cho mọi class, ghi đè threshold trong checkpoint.")
    p.add_argument("--tune-thresholds-on-val", action="store_true",
                   help="Dò lại threshold tối ưu trên val rồi áp lên split đang đánh giá.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Thư mục ghi kết quả. Mặc định: outputs/<experiment_name>/eval_<split>/")
    p.add_argument("--no-save", action="store_true", help="Chỉ in ra console, không ghi file.")
    p.add_argument("--patient", type=str, default=None,
                   help="patient_id → chạy chế độ dự đoán 1 bệnh nhân thay vì đánh giá cả split.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.patient:
        predict_single(cfg, args)
    else:
        evaluate(cfg, args)


# num_workers > 0 trên Windows cần entry point được bảo vệ như thế này
if __name__ == "__main__":
    main()
