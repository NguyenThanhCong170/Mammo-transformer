"""
MammoTransformer (Swin-V2 + ipsilateral/bilateral cross-attention) + Focal Loss, nhị phân normal/abnormal.

Mỗi sample = 1 EXAM gồm 4 ảnh: L_MLO, L_CC, R_MLO, R_CC. Output: 1 logit (abnormal).
Chia train/val/test THEO BỆNH NHÂN. Nhãn ảnh: finding_birads NaN → 0, 3/4/5 → 1.
Nhãn exam = 1 nếu BẤT KỲ ảnh nào trong exam là abnormal (max).

Giả định về CSV (đổi bằng tham số nếu tên cột khác):
    image_path   : đường dẫn ảnh
    patient_id   : mã bệnh nhân                  (--patient-col)
    study_id     : mã lần chụp, tùy chọn         (--study-col, mặc định bỏ qua → 1 bệnh nhân = 1 exam)
    laterality   : L / R (hoặc LEFT / RIGHT)     (--laterality-col)
    view_position: CC / MLO                      (--view-col)
Exam thiếu bất kỳ view nào trong 4 view sẽ bị loại (có in số lượng).

Mixed precision: bf16 autocast (không cần GradScaler). GPU không hỗ trợ bf16 → chạy fp32.

Chạy:
    python Mammo-resnet101/train_mammo_transformer.py
    python Mammo-resnet101/train_mammo_transformer.py --unfreeze-from 0          # train toàn bộ backbone
    python Mammo-resnet101/train_mammo_transformer.py --backbone-ckpt checkpoints/phase1.pt
    python Mammo-resnet101/train_mammo_transformer.py --eval-only

Sau khi train xong, script đánh giá checkpoint tốt nhất trên val và test:
AUC, AP và Sens, Spec, PPV, F1 (ở ngưỡng 0.5 và ngưỡng tối ưu F1 chọn trên val),
đồng thời lưu kết quả ra file .metrics.json cạnh checkpoint.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from configs.config import Config, resolve_path
from data.augmentation import build_transforms
from data.dataset import birads_to_label, split_patients
# ↓ đổi đường dẫn import này cho đúng với vị trí file chứa MammoTransformer của bạn
from mammo_transformer import MammoTransformer, VIEW_KEYS

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bf16 autocast: cần GPU Ampere trở lên (RTX 30xx, A100...). Không hỗ trợ thì chạy fp32.
use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16
print(f"[AMP] {'bf16' if use_amp else 'tắt (fp32)'}")


# ──────────────────────────────────────────────
# Dataset: 1 mẫu = 1 exam = 4 ảnh
# ──────────────────────────────────────────────
def build_exams(df, patient_col, study_col, lat_col, view_col):
    """Gom các dòng ảnh thành exam: trả về (list[dict view_key → path], list[label])."""
    for c in [patient_col, lat_col, view_col, "image_path", "label"] + ([study_col] if study_col else []):
        assert c in df.columns, f"Không có cột '{c}' trong CSV. Các cột hiện có: {list(df.columns)}"

    d = df.copy()
    lat = d[lat_col].astype(str).str.strip().str.upper().str[0]
    vw = d[view_col].astype(str).str.upper()
    view = pd.Series(np.where(vw.str.contains("MLO"), "MLO", np.where(vw.str.contains("CC"), "CC", "")),
                     index=d.index)
    d["_key"] = lat + "_" + view
    d = d[d["_key"].isin(VIEW_KEYS)]

    group_cols = [patient_col] + ([study_col] if study_col else [])
    exams, labels, n_incomplete = [], [], 0
    for _, g in d.groupby(group_cols, sort=False):
        paths = g.drop_duplicates("_key").set_index("_key")["image_path"].to_dict()
        if not all(k in paths for k in VIEW_KEYS):
            n_incomplete += 1
            continue
        exams.append({k: paths[k] for k in VIEW_KEYS})
        labels.append(int(g["label"].max()))
    return exams, labels, n_incomplete


class MammoExamDataset(Dataset):
    def __init__(self, exams, labels, image_size=(512, 256), is_train=True, aug_level=3):
        self.exams, self.labels = exams, labels
        self.h, self.w = image_size
        self.transform = build_transforms(is_train, aug_level)

    def __len__(self):
        return len(self.exams)

    def _load(self, path):
        img = Image.open(resolve_path(path)).convert("L")
        if img.size != (self.w, self.h):                    # PIL dùng (W, H)
            img = img.resize((self.w, self.h), Image.BILINEAR)
        return self.transform(np.array(img, dtype=np.uint8))   # → Tensor(3, H, W)

    def __getitem__(self, i):
        x = {k: self._load(self.exams[i][k]) for k in VIEW_KEYS}
        y = torch.tensor([float(self.labels[i])], dtype=torch.float32)
        return x, y

    def class_counts(self):
        pos = sum(self.labels)
        return len(self.labels) - pos, pos


def build_loaders(args):
    df = pd.read_csv(resolve_path(args.csv))
    df["label"] = df[args.birads_col].map(birads_to_label).astype(int)
    splits = split_patients(df, val_ratio=args.val_ratio, seed=args.seed)   # chia theo bệnh nhân

    size = (args.img_h, args.img_w)
    loaders = []
    for name, sdf, is_train in zip(("train", "val", "test"), splits, (True, False, False)):
        exams, labels, n_inc = build_exams(sdf, args.patient_col, args.study_col,
                                           args.laterality_col, args.view_col)
        ds = MammoExamDataset(exams, labels, size, is_train=is_train, aug_level=args.aug_level)
        neg, pos = ds.class_counts()
        print(f"[Dataset] {name}: {len(ds)} exam (normal {neg}, abnormal {pos}), "
              f"bỏ {n_inc} exam thiếu view")
        loaders.append(DataLoader(
            ds, batch_size=args.batch_size, shuffle=is_train, drop_last=is_train,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0))
    return loaders


# ──────────────────────────────────────────────
# Focal Loss (nhị phân, nhận logits)
# ──────────────────────────────────────────────
class BinaryFocalLoss(nn.Module):
    """FL = -alpha_t * (1 - p_t)^gamma * log(p_t); alpha là trọng số lớp abnormal."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()   # luôn tính loss ở fp32
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ──────────────────────────────────────────────
# Train / Eval
# ──────────────────────────────────────────────
def to_device(x):
    return {k: v.to(device, non_blocking=True) for k, v in x.items()}


def train_one_epoch(model, loader, criterion, optimizer, grad_clip=1.0):
    model.train()
    running, total = 0.0, 0
    for x, y in loader:
        x, y = to_device(x), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)                       # (B, 1)
        loss = criterion(logits, y)                 # criterion tự cast logits sang float32
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        running += loss.item() * y.size(0)
        total += y.size(0)
    return running / max(total, 1)


@torch.no_grad()
def predict(model, loader, criterion):
    model.eval()
    probs, ys, running, total = [], [], 0.0, 0
    for x, y in loader:
        x, y = to_device(x), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
        running += criterion(logits, y).item() * y.size(0)
        total += y.size(0)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
        ys.append(y.cpu().numpy().ravel())
    return running / max(total, 1), np.concatenate(probs), np.concatenate(ys)


def metrics_at(probs, ys, thr):
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (ys == 1)).sum()); tn = int(((pred == 0) & (ys == 0)).sum())
    fp = int(((pred == 1) & (ys == 0)).sum()); fn = int(((pred == 0) & (ys == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    return dict(sens=sens, spec=spec, ppv=prec, f1=f1)


def best_threshold(probs, ys):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([metrics_at(probs, ys, t)["f1"] for t in grid]))])


def report(tag, probs, ys, thr):
    """In và trả về dict: AUC, AP, Sens, Spec, PPV, F1 (+ ngưỡng)."""
    if len(np.unique(ys)) < 2:
        auc = ap = float("nan")          # tập chỉ có 1 lớp → AUC không xác định
    else:
        auc, ap = roc_auc_score(ys, probs), average_precision_score(ys, probs)
    m = metrics_at(probs, ys, thr)
    print(f"  [{tag}] AUC {auc:.4f} | AP {ap:.4f} | thr {thr:.2f} → "
          f"Sens {m['sens']:.4f} Spec {m['spec']:.4f} PPV {m['ppv']:.4f} F1 {m['f1']:.4f}")
    return dict(auc=float(auc), ap=float(ap), thr=float(thr), **{k: float(v) for k, v in m.items()})


def evaluate_final(model, val_loader, test_loader, criterion, epoch, out_path):
    """Đánh giá checkpoint hiện tại trên val và test, in kết quả và lưu JSON."""
    _, va_p, va_y = predict(model, val_loader, criterion)
    thr = best_threshold(va_p, va_y)          # ngưỡng chọn trên val, áp dụng cho test
    _, te_p, te_y = predict(model, test_loader, criterion)

    print(f"\n{'=' * 70}\n  ĐÁNH GIÁ CUỐI — checkpoint epoch {epoch} (val-thr = {thr:.2f})\n{'=' * 70}")
    results = {
        "epoch": epoch,
        "val @0.5": report("val  @0.5    ", va_p, va_y, 0.5),
        "val @val-thr": report("val  @val-thr ", va_p, va_y, thr),
        "test @0.5": report("test @0.5    ", te_p, te_y, 0.5),
        "test @val-thr": report("test @val-thr ", te_p, te_y, thr),
    }
    metrics_path = Path(out_path).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"\n  ✔ Đã lưu metrics: {metrics_path}")
    return results


def build_scheduler(optimizer, epochs, warmup_epochs):
    """Warmup tuyến tính (theo epoch) rồi cosine."""
    warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
    if warmup_epochs <= 0:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def build_model(args):
    model = MammoTransformer(
        backbone_name=args.backbone,
        backbone_pretrained=not args.no_pretrained,
        backbone_img_size=(args.img_h, args.img_w),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        ffn_dropout=args.ffn_dropout,
        num_ipsi_layers=args.ipsi_layers,
        num_bilateral_layers=args.bilateral_layers,
        mlp_hidden_dim=args.mlp_hidden,
        mlp_dropout=args.mlp_dropout,
        num_classes=1,                                 # 1 logit nhị phân (abnormal)
        token_grid=(args.token_h, args.token_w),
    )
    if args.backbone_ckpt:
        model.load_backbone_weights(args.backbone_ckpt, device="cpu")
    if args.unfreeze_from >= 0:
        model.unfreeze_backbone_from_stage(args.unfreeze_from)   # 0 = mở hết, 2 = chỉ stage 3+4
    else:
        model.unfreeze_backbone_from_stage(None)                 # đóng băng toàn bộ backbone
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    # dữ liệu
    ap.add_argument("--csv", default=cfg.data.csv_path)
    ap.add_argument("--birads-col", default="finding_birads")
    ap.add_argument("--patient-col", default="patient_id")
    ap.add_argument("--study-col", default=None, help="cột mã lần chụp; bỏ trống → 1 bệnh nhân = 1 exam")
    ap.add_argument("--laterality-col", default="laterality")
    ap.add_argument("--view-col", default="view_position")
    ap.add_argument("--img-h", type=int, default=256)
    ap.add_argument("--img-w", type=int, default=256)
    ap.add_argument("--val-ratio", type=float, default=cfg.data.val_ratio)
    ap.add_argument("--aug-level", type=int, default=cfg.data.aug_level)
    ap.add_argument("--num-workers", type=int, default=cfg.data.num_workers)
    # backbone
    ap.add_argument("--backbone", default="swinv2_base_window8_256.ms_in1k")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--backbone-ckpt", default=None, help="backbone đã contrastive-pretrain ở phase 1")
    ap.add_argument("--unfreeze-from", type=int, default=2,
                    help="mở backbone từ stage này (đánh số từ 0): 2 = chỉ stage 3+4, 0 = toàn bộ, -1 = đóng băng hết")
    ap.add_argument("--token-h", type=int, default=8)
    ap.add_argument("--token-w", type=int, default=8)
    # fusion + classifier
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=2)
    ap.add_argument("--attn-dropout", type=float, default=0.1)
    ap.add_argument("--ffn-dropout", type=float, default=0.1)
    ap.add_argument("--ipsi-layers", type=int, default=1)
    ap.add_argument("--bilateral-layers", type=int, default=1)
    ap.add_argument("--mlp-hidden", type=int, default=256)
    ap.add_argument("--mlp-dropout", type=float, default=0.5)
    # tối ưu
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32, help="số EXAM mỗi batch (mỗi exam 4 ảnh)")
    ap.add_argument("--lr", type=float, default=1e-4, help="lr của phần fusion/classifier")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1, help="lr backbone = lr × hệ số này")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0, help="0 để tắt")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=None,
                    help="trọng số lớp abnormal; mặc định = tỉ lệ normal trong train")
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--out", default="checkpoints/mammo_transformer_baseline.pt")
    ap.add_argument("--eval-only", action="store_true", help="bỏ qua train, chỉ đánh giá checkpoint ở --out")
    args = ap.parse_args()
    args.out = str(resolve_path(args.out))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader = build_loaders(args)

    neg, pos = train_loader.dataset.class_counts()
    alpha = args.alpha if args.alpha is not None else neg / max(neg + pos, 1)
    print(f"[Train] normal={neg}, abnormal={pos} → focal alpha={alpha:.3f}, gamma={args.gamma}")
    criterion = BinaryFocalLoss(alpha=alpha, gamma=args.gamma)

    model = build_model(args)
    cnt = model.count_parameters()
    print(f"[Model] tổng {cnt['total'] / 1e6:.1f}M tham số "
          f"(backbone {cnt['backbone'] / 1e6:.1f}M, còn lại {cnt['other'] / 1e6:.1f}M) "
          f"— trainable {cnt['trainable'] / 1e6:.1f}M")

    if args.eval_only:
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["model"])
        evaluate_final(model, val_loader, test_loader, criterion, ckpt.get("epoch", -1), args.out)
        return

    # 2 nhóm lr: fusion/classifier (lr) và backbone (lr × mult). Weight decay áp dụng đồng đều
    # cho cả hai nhóm, trừ tham số 1 chiều (bias, LayerNorm, embedding...) thì không decay.
    groups = []
    for g in model.get_param_groups(args.lr, args.backbone_lr_mult):
        dec = [p for p in g["params"] if p.ndim > 1]
        nodec = [p for p in g["params"] if p.ndim <= 1]
        groups.append({"params": dec, "lr": g["lr"], "weight_decay": args.weight_decay})
        groups.append({"params": nodec, "lr": g["lr"], "weight_decay": 0.0})
    optimizer = optim.AdamW([g for g in groups if g["params"]])
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch = -float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, args.grad_clip)
        va_loss, va_p, va_y = predict(model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | train loss {tr_loss:.4f} | val loss {va_loss:.4f}")
        auc = report("val", va_p, va_y, 0.5)["auc"]

        score = auc if not np.isnan(auc) else -va_loss
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, args.out)
            print(f"  ✔ lưu checkpoint tốt nhất (epoch {epoch})")

    # Train xong → nạp lại checkpoint tốt nhất rồi đánh giá đầy đủ trên val + test
    model.load_state_dict(torch.load(args.out, map_location=device)["model"])
    evaluate_final(model, val_loader, test_loader, criterion, best_epoch, args.out)


if __name__ == "__main__":
    main()