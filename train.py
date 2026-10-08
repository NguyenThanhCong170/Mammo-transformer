"""
Baseline: ConvNeXt-Base (timm, ImageNet pretrained) + Focal Loss, nhị phân normal/abnormal.
Mỗi sample = 1 ảnh. Chia train/val/test THEO BỆNH NHÂN. Nhãn: finding_birads NaN → 0, 3/4/5 → 1.

Cần: pip install timm

Model ConvNeXt (timm):
    convnext_base.fb_in22k_ft_in1k        (mặc định, ~89M, pretrain IN-22k → fine-tune IN-1k, ảnh 224)
    convnext_base.fb_in1k                 (chỉ IN-1k)
    convnext_base.fb_in22k_ft_in1k_384    (fine-tune ở 384 → dùng với --img-size 384)
    convnext_tiny/small/large.fb_in22k_ft_in1k   (các cỡ khác)
Thêm --no-pretrained để train từ đầu.

Chạy:
    python Mammo-resnet101/train_convnext.py
    python Mammo-resnet101/train_convnext.py --cnx-name convnext_base.fb_in22k_ft_in1k_384 --img-size 384 --batch-size 16
    python Mammo-resnet101/train_convnext.py --eval-only                    # chỉ đánh giá checkpoint đã lưu

ConvNeXt là mạng tích chập thuần (dùng LayerNorm, không BatchNorm) nên chạy được ở mọi kích thước ảnh
chia hết cho 32, không cần nội suy position embedding.

Sau khi train xong, script đánh giá checkpoint tốt nhất trên val và test:
AUC, AP (không phụ thuộc ngưỡng) và Sens, Spec, PPV, F1 (ở ngưỡng 0.5 và ngưỡng tối ưu F1 chọn trên val),
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

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Dataset: 1 dòng CSV = 1 ảnh
# ──────────────────────────────────────────────
class MammoImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size=(224, 224), is_train=True, aug_level=3):
        self.paths = df["image_path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.h, self.w = image_size
        self.transform = build_transforms(is_train, aug_level)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(resolve_path(self.paths[i])).convert("L")
        if img.size != (self.w, self.h):                    # PIL dùng (W, H)
            img = img.resize((self.w, self.h), Image.BILINEAR)
        x = self.transform(np.array(img, dtype=np.uint8))   # → Tensor(3, H, W)
        y = torch.tensor([float(self.labels[i])], dtype=torch.float32)
        return x, y

    def class_counts(self):
        pos = sum(self.labels)
        return len(self.labels) - pos, pos


def build_loaders(csv, birads_col, batch_size, num_workers, val_ratio, seed, aug_level, img_size):
    df = pd.read_csv(resolve_path(csv))   # tương đối theo PROJECT_ROOT, không theo cwd
    df["label"] = df[birads_col].map(birads_to_label).astype(int)
    train_df, val_df, test_df = split_patients(df, val_ratio=val_ratio, seed=seed)

    size = (img_size, img_size)
    train_ds = MammoImageDataset(train_df, size, is_train=True, aug_level=aug_level)
    val_ds = MammoImageDataset(val_df, size, is_train=False, aug_level=aug_level)
    test_ds = MammoImageDataset(test_df, size, is_train=False, aug_level=aug_level)

    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    print(f"[Dataset] ảnh — train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# Focal Loss (nhị phân, nhận logits)
# ──────────────────────────────────────────────
class BinaryFocalLoss(nn.Module):
    """FL = -alpha_t * (1 - p_t)^gamma * log(p_t); alpha là trọng số lớp abnormal."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ──────────────────────────────────────────────
# Model: ConvNeXt-Base → 1 logit
# ──────────────────────────────────────────────
class ConvNeXtWrapper(nn.Module):
    """Backbone ConvNeXt (timm) pretrained, head phân loại thay bằng 1 logit (abnormal)."""

    def __init__(self, name, pretrained=True, dropout=0.0, drop_path=0.1):
        super().__init__()
        import timm
        self.net = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=1,            # thay head mới (khởi tạo ngẫu nhiên) → 1 logit
            drop_rate=dropout,        # dropout trước head
            drop_path_rate=drop_path,  # stochastic depth
        )
        print(f"[Model] {name} — {'PRETRAINED' if pretrained else 'khởi tạo NGẪU NHIÊN'}")

    def forward(self, x):
        return self.net(x)            # (B, 1) logits


# ──────────────────────────────────────────────
# Train / Eval
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, grad_clip=1.0):
    model.train()
    running, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(x)
        loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
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
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
    """Warmup tuyến tính (theo epoch) rồi cosine — ConvNeXt fine-tune cần warmup để ổn định."""
    warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
    if warmup_epochs <= 0:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnx-name", default="convnext_base.fb_in22k_ft_in1k",
                    help="tên model timm ConvNeXt, vd convnext_base.fb_in1k")
    ap.add_argument("--no-pretrained", action="store_true", help="train từ đầu, không dùng pretrain")
    ap.add_argument("--img-size", type=int, default=224, help="kích thước ảnh vuông đưa vào ConvNeXt (chia hết cho 32)")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout trước head")
    ap.add_argument("--drop-path", type=float, default=0.1, help="stochastic depth rate")
    ap.add_argument("--csv", default=cfg.data.csv_path)
    ap.add_argument("--birads-col", default="finding_birads")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=cfg.data.num_workers)
    ap.add_argument("--val-ratio", type=float, default=cfg.data.val_ratio)
    ap.add_argument("--aug-level", type=int, default=cfg.data.aug_level)
    ap.add_argument("--lr", type=float, default=5e-5, help="ConvNeXt fine-tune nên dùng lr nhỏ")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0, help="0 để tắt")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=None,
                    help="trọng số lớp abnormal; mặc định = tỉ lệ normal trong train")
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--out", default=None, help="mặc định checkpoints/<tên model>_baseline.pt")
    ap.add_argument("--eval-only", action="store_true",
                    help="bỏ qua train, chỉ đánh giá checkpoint ở --out")
    args = ap.parse_args()
    args.out = str(resolve_path(args.out or f"checkpoints/{args.cnx_name.split('.')[0]}_baseline.pt"))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader = build_loaders(
        args.csv, args.birads_col, args.batch_size, args.num_workers,
        args.val_ratio, args.seed, args.aug_level, args.img_size)

    neg, pos = train_loader.dataset.class_counts()
    alpha = args.alpha if args.alpha is not None else neg / max(neg + pos, 1)
    print(f"[Train] normal={neg}, abnormal={pos} → focal alpha={alpha:.3f}, gamma={args.gamma}")
    criterion = BinaryFocalLoss(alpha=alpha, gamma=args.gamma)

    model = ConvNeXtWrapper(args.cnx_name, pretrained=not args.no_pretrained,
                            dropout=args.dropout, drop_path=args.drop_path).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {args.cnx_name} — {n_params / 1e6:.1f}M tham số")

    if args.eval_only:
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["model"])
        evaluate_final(model, val_loader, test_loader, criterion, ckpt.get("epoch", -1), args.out)
        return

    # AdamW + không weight-decay cho bias / LayerNorm / layer-scale gamma (đều là tham số 1 chiều)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch = -float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, args.grad_clip)
        va_loss, va_p, va_y = predict(model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"train loss {tr_loss:.4f} | val loss {va_loss:.4f}")
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