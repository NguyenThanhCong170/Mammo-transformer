"""
Baseline: ResNet-101 (ImageNet pretrained) + Focal Loss, nhị phân normal/abnormal.
Mỗi sample = 1 ảnh (không multi-view). Chia train/val/test THEO BỆNH NHÂN để không rò rỉ dữ liệu.
Nhãn ảnh: finding_birads NaN → 0 (normal), 3/4/5 → 1 (abnormal).
"""

import argparse
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
from torchvision import models

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


def build_loaders(csv, birads_col, batch_size, num_workers, val_ratio, seed, aug_level):
    df = pd.read_csv(resolve_path(csv))   # tương đối theo PROJECT_ROOT, không theo cwd
    df["label"] = df[birads_col].map(birads_to_label).astype(int)
    train_df, val_df, test_df = split_patients(df, val_ratio=val_ratio, seed=seed)

    train_ds = MammoImageDataset(train_df, is_train=True, aug_level=aug_level)
    val_ds = MammoImageDataset(val_df, is_train=False, aug_level=aug_level)
    test_ds = MammoImageDataset(test_df, is_train=False, aug_level=aug_level)

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
# Train / Eval
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler):
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
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
    if len(np.unique(ys)) < 2:
        auc = ap = float("nan")          # tập chỉ có 1 lớp → AUC không xác định
    else:
        auc, ap = roc_auc_score(ys, probs), average_precision_score(ys, probs)
    m = metrics_at(probs, ys, thr)
    print(f"  [{tag}] AUC {auc:.4f} | AP {ap:.4f} | thr {thr:.2f} → "
          f"Sens {m['sens']:.4f} Spec {m['spec']:.4f} PPV {m['ppv']:.4f} F1 {m['f1']:.4f}")
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=cfg.data.csv_path)
    ap.add_argument("--birads-col", default="finding_birads")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=cfg.data.num_workers)
    ap.add_argument("--val-ratio", type=float, default=cfg.data.val_ratio)
    ap.add_argument("--aug-level", type=int, default=cfg.data.aug_level)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=None,
                    help="trọng số lớp abnormal; mặc định = tỉ lệ normal trong train")
    ap.add_argument("--seed", type=int, default=cfg.data.seed)
    ap.add_argument("--out", default="checkpoints/resnet101_baseline.pt")
    args = ap.parse_args()
    args.out = str(resolve_path(args.out))   # checkpoint luôn nằm trong PROJECT_ROOT

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader = build_loaders(
        args.csv, args.birads_col, args.batch_size, args.num_workers,
        args.val_ratio, args.seed, args.aug_level)

    neg, pos = train_loader.dataset.class_counts()
    alpha = args.alpha if args.alpha is not None else neg / max(neg + pos, 1)
    print(f"[Train] normal={neg}, abnormal={pos} → focal alpha={alpha:.3f}, gamma={args.gamma}")
    criterion = BinaryFocalLoss(alpha=alpha, gamma=args.gamma)

    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)       # 1 logit: abnormal
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch = -float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        va_loss, va_p, va_y = predict(model, val_loader, criterion)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | train loss {tr_loss:.4f} | val loss {va_loss:.4f}")
        auc = report("val", va_p, va_y, 0.5)

        score = auc if not np.isnan(auc) else -va_loss
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, args.out)
            print(f"  ✔ lưu checkpoint tốt nhất (epoch {epoch})")

    print(f"\n[Test] dùng checkpoint epoch {best_epoch}")
    model.load_state_dict(torch.load(args.out, map_location=device)["model"])
    _, va_p, va_y = predict(model, val_loader, criterion)
    thr = best_threshold(va_p, va_y)
    _, te_p, te_y = predict(model, test_loader, criterion)
    report("test @0.5", te_p, te_y, 0.5)
    report("test @val-thr", te_p, te_y, thr)


if __name__ == "__main__":
    main()