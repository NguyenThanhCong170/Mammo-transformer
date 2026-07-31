"""
PHASE 2 — Freeze backbone (đã contrastive-pretrain ở phase 1),
train cross-attention + MLP classifier cho multi-label classification.

Chạy:
    python train_phase1.py     # trước
    python train_phase2.py     # sau
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from configs.config import Config
from data.dataset import build_dataloaders
from models.mammo_transformer import MammoTransformer
from utils.losses import FocalLoss, MultiLabelMetricsCalculator
from utils.wandb_utils import WandbLogger, flatten_metrics

CLASS_NAMES = ["no_finding", "mass", "calcification", "asymmetry"]


# ──────────────────────────────────────────────
# Checkpoint Manager
# ──────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, output_dir: str, save_top_k: int = 3, monitor: str = "macro_ap"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.top_checkpoints = []          # List[(score, path)]

    def save(self, model, optimizer, epoch: int, metrics: dict):
        score = float(metrics.get(self.monitor, 0.0))
        path = self.output_dir / f"epoch{epoch:03d}_{self.monitor}{score:.4f}.pt"

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {k: v for k, v in metrics.items() if k != "thresholds"},
            "thresholds": metrics.get("thresholds"),
        }, path)

        self.top_checkpoints.append((score, path))
        self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)

        while len(self.top_checkpoints) > self.save_top_k:
            _, old_path = self.top_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

        print(f"  [Checkpoint] Đã lưu: {path.name} (giữ top-{self.save_top_k})")
        return path

    def best_checkpoint(self) -> Optional[Path]:
        return self.top_checkpoints[0][1] if self.top_checkpoints else None


# ──────────────────────────────────────────────
# Train 1 epoch
# ──────────────────────────────────────────────
def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, scaler,
    device, cfg, epoch, logger, global_step, amp_dtype,
):
    model.train()          # override trong MammoTransformer giữ backbone ở eval nếu frozen
    total_loss = 0.0
    accumulate_steps = cfg.train.accumulate_grad_steps

    metrics_calc = MultiLabelMetricsCalculator(
        num_classes=cfg.data.num_classes,
        class_names=CLASS_NAMES,
    )
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for step, batch in enumerate(loader):
        images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels) / accumulate_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulate_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.train.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()               # ← step theo OPTIMIZER STEP, không theo epoch
            global_step += 1

        batch_loss = loss.item() * accumulate_steps
        total_loss += batch_loss
        metrics_calc.update(logits.detach().float(), labels)

        if (step + 1) % cfg.train.log_every_n_steps == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [P2] Epoch {epoch} | Step {step+1}/{len(loader)} | "
                  f"loss {batch_loss:.4f} | avg {total_loss/(step+1):.4f} | lr {lr_now:.2e}")
            logger.log({"train/step_loss": batch_loss, "train/lr": lr_now}, step=global_step)

    metrics = metrics_calc.compute()
    metrics["loss"] = total_loss / max(1, len(loader))
    metrics["time"] = time.time() - t0
    return metrics, global_step


# ──────────────────────────────────────────────
# Validate / Test
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, cfg, scaler, amp_dtype,
             split: str = "Val", thresholds: Optional[list] = None) -> dict:
    model.eval()
    total_loss = 0.0
    metrics_calc = MultiLabelMetricsCalculator(
        num_classes=cfg.data.num_classes,
        class_names=CLASS_NAMES,
    )

    for batch in loader:
        images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        metrics_calc.update(logits.float(), labels)

    # Val: tự tìm threshold. Test: dùng threshold đã chốt từ val → tránh leakage.
    if thresholds is None:
        thresholds = metrics_calc.find_optimal_thresholds()

    metrics = metrics_calc.print_report(split, thresholds=thresholds)
    metrics.update(metrics["macro"])          # đẩy macro_ap / macro_auc... lên top-level
    metrics["loss"] = total_loss / max(1, len(loader))
    metrics["thresholds"] = thresholds
    return metrics


# ──────────────────────────────────────────────
# Optimizer + Scheduler (chỉ train non-backbone)
# ──────────────────────────────────────────────
def build_optimizer_scheduler(model, lr: float, cfg: Config, total_optim_steps: int):
    params = [p for n, p in model.named_parameters()
              if not n.startswith("backbone") and p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=cfg.train.weight_decay)

    effective_warmup = max(1, min(cfg.train.warmup_steps, total_optim_steps // 2))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=effective_warmup)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_optim_steps - effective_warmup), eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[effective_warmup])
    return optimizer, scheduler


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32
    print(f"\n[Phase 2] Device: {device} | AMP: {use_amp}")

    out_dir = Path(cfg.train.output_dir) / cfg.train.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(cfg, phase="phase2")

    # ── Data
    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=cfg.data.csv_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
        aug_level=cfg.data.aug_level,
        num_classes=cfg.data.num_classes,
        persistent_workers=cfg.data.persistent_workers,
    )

    # ── Model
    model = MammoTransformer(
        backbone_name=cfg.model.backbone_name,
        backbone_pretrained=cfg.model.backbone_pretrained,
        backbone_img_size=cfg.model.backbone_img_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        attn_dropout=cfg.model.attn_dropout,
        ffn_dropout=cfg.model.ffn_dropout,
        num_ipsi_layers=cfg.model.num_ipsi_layers,
        num_bilateral_layers=cfg.model.num_bilateral_layers,
        mlp_hidden_dim=cfg.model.mlp_hidden_dim,
        mlp_dropout=cfg.model.mlp_dropout,
        num_classes=cfg.model.num_classes,
        token_grid=cfg.model.token_grid,
        ffn_expansion=cfg.model.ffn_expansion,
    ).to(device)

    # ── Nạp backbone từ phase 1 (bước này trước đây BỊ THIẾU
    #    → toàn bộ contrastive learning của phase 1 bị vứt bỏ)
    phase1_ckpt = out_dir / cfg.train.phase1_ckpt_name
    if cfg.train.load_phase1_backbone:
        if phase1_ckpt.exists():
            model.load_backbone_weights(str(phase1_ckpt), device=device)
        else:
            print(f"[Phase 2] ⚠ Không tìm thấy {phase1_ckpt}. "
                  f"Đang dùng backbone ImageNet — hãy chạy train_phase1.py trước.")
    model.to(device)

    # ── Freeze backbone
    print("\n" + "=" * 60)
    print("  Frozen backbone — train cross-attention + classifier")
    print("=" * 60)
    model.freeze_backbone()

    counts = model.count_parameters()
    print(f"[Model] Backbone: {counts['backbone']:,} | Other: {counts['other']:,} | "
          f"Total: {counts['total']:,} | Trainable: {counts['trainable']:,}")
    if cfg.wandb.watch_model:
        logger.watch(model, log_freq=cfg.wandb.log_every_n_steps)

    # ── Loss / Optim / AMP
    criterion = FocalLoss(
        alpha=cfg.train.focal_alpha,
        gamma=cfg.train.focal_gamma,
        reduction=cfg.train.focal_reduction,
    ).to(device)

    steps_per_epoch = max(1, len(train_loader) // cfg.train.accumulate_grad_steps)
    total_optim_steps = steps_per_epoch * cfg.train.epochs_phase2
    optimizer, scheduler = build_optimizer_scheduler(
        model, cfg.train.lr_phase2, cfg, total_optim_steps)
    scaler = GradScaler(device=device.type, enabled=use_amp)

    ckpt_manager = CheckpointManager(
        output_dir=str(out_dir),
        save_top_k=cfg.train.save_top_k,
        monitor="macro_ap",
    )

    best_score, best_thresholds, no_improve, global_step = 0.0, None, 0, 0
    history = {"phase2": []}

    # ── Loop
    for epoch in range(1, cfg.train.epochs_phase2 + 1):
        print(f"\n── [Phase 2] Epoch {epoch}/{cfg.train.epochs_phase2}")

        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler,
            device, cfg, epoch, logger, global_step, amp_dtype,
        )
        val_metrics = validate(model, val_loader, criterion, device, cfg,
                               scaler, amp_dtype, split="Val")

        lr_now = optimizer.param_groups[0]["lr"]
        epoch_log = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
                     "time": round(train_metrics["time"], 1)}
        history["phase2"].append(epoch_log)

        logger.log_epoch(epoch, {
            **flatten_metrics(train_metrics, "train"),
            **flatten_metrics(val_metrics, "val"),
            "train/epoch_lr": lr_now,
        })

        if val_metrics["macro_ap"] > best_score:
            best_score = val_metrics["macro_ap"]
            best_thresholds = val_metrics["thresholds"]
            ckpt_manager.save(model, optimizer, epoch, val_metrics)
            no_improve = 0
            logger.set_summary({"best_val_macro_ap": best_score, "best_epoch": epoch})
        else:
            no_improve += 1

        print(f"  [P2] Epoch {epoch} | train loss {train_metrics['loss']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} | macro-AP {val_metrics['macro_ap']:.4f} | "
              f"macro-AUC {val_metrics['macro_auc']:.4f} | best AP {best_score:.4f} | "
              f"{epoch_log['time']}s")

        if no_improve >= cfg.train.early_stopping_patience:
            print(f"  [P2] Early stop — không cải thiện {no_improve} epoch.")
            break

    # ── Test
    print("\n" + "=" * 60)
    print("  FINAL TEST EVALUATION")
    print("=" * 60)

    best_ckpt = ckpt_manager.best_checkpoint()
    if best_ckpt:
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Nạp checkpoint tốt nhất: {best_ckpt.name}")

    test_metrics = validate(model, test_loader, criterion, device, cfg,
                            scaler, amp_dtype, split="Test", thresholds=best_thresholds)
    logger.log_epoch(cfg.train.epochs_phase2 + 1, flatten_metrics(test_metrics, "test"))
    logger.set_summary({f"test_{k}": v for k, v in test_metrics["macro"].items()})

    # ── Save
    final_model_path = out_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "thresholds"},
        "thresholds": best_thresholds,
        "class_names": CLASS_NAMES,
    }, final_model_path)
    print(f"[Done] Final model → {final_model_path}")

    history_path = out_dir / "history_phase2.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[Done] History → {history_path}")

    logger.log_artifact(final_model_path, name=f"{cfg.train.experiment_name}-final")
    logger.finish()

    return model, history, test_metrics


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train(Config())
