"""
Training loop — 2-phase strategy:
  Phase 1: Freeze backbone, train attention + classifier
  Phase 2: Unfreeze backbone, full fine-tune với LR nhỏ hơn
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import wandb

from configs.config import Config
from data.dataset import build_dataloaders
from models.mammo_transformer import MammoTransformer
from utils.losses import FocalLoss, MetricsCalculator


# ──────────────────────────────────────────────
# Checkpoint Manager
# ──────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, output_dir: str, save_top_k: int = 3, monitor: str = "auc"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.top_checkpoints = []  # List of (score, path)

    def save(self, model, optimizer, epoch: int, metrics: dict, phase: str):
        score = metrics.get(self.monitor, 0.0)
        path = self.output_dir / f"epoch{epoch:03d}_{phase}_{self.monitor}{score:.4f}.pt"

        torch.save({
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }, path)

        self.top_checkpoints.append((score, path))
        self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Xóa checkpoint dư
        while len(self.top_checkpoints) > self.save_top_k:
            _, old_path = self.top_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

        print(f"  [Checkpoint] Saved: {path.name}  (top-{self.save_top_k})")
        return path

    def best_checkpoint(self) -> Optional[Path]:
        if self.top_checkpoints:
            return self.top_checkpoints[0][1]
        return None


# ──────────────────────────────────────────────
# One epoch train
# ──────────────────────────────────────────────
def train_one_epoch(
    model, loader, optimizer, criterion, scaler,
    device, accumulate_steps: int, epoch: int
) -> dict:

    model.train()
    total_loss = 0.0
    metrics_calc = MetricsCalculator()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        images = {k: v.to(device) for k, v in batch["images"].items()}
        labels = batch["label"].to(device)

        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels) / accumulate_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulate_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulate_steps
        metrics_calc.update(logits, labels)

        if (step + 1) % 20 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | "
                  f"Loss: {total_loss / (step+1):.4f}")

    metrics = metrics_calc.compute()
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    metrics_calc = MetricsCalculator()

    for batch in loader:
        images = {k: v.to(device) for k, v in batch["images"].items()}
        labels = batch["label"].to(device)

        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        metrics_calc.update(logits, labels)

    # Update threshold từ val set
    optimal_thresh = metrics_calc.find_optimal_threshold()
    metrics_calc.threshold = optimal_thresh

    metrics = metrics_calc.print_report("Val")
    metrics["loss"] = total_loss / len(loader)
    metrics["optimal_threshold"] = optimal_thresh
    return metrics


# ──────────────────────────────────────────────
# Build Optimizer + Scheduler
# ──────────────────────────────────────────────
def build_optimizer_scheduler(model, lr: float, cfg: Config, total_steps: int, phase: int):
    if phase == 1:
        # Phase 1: Chỉ optimize non-backbone parameters
        params = [p for n, p in model.named_parameters()
                  if "backbone" not in n and p.requires_grad]
        optimizer = AdamW(params, lr=lr, weight_decay=cfg.train.weight_decay)
    else:
        # Phase 2: Full model với backbone LR nhỏ hơn
        param_groups = model.get_param_groups(
            lr=lr,
            backbone_lr_multiplier=cfg.train.backbone_lr_multiplier
        )
        optimizer = AdamW(param_groups, weight_decay=cfg.train.weight_decay)

    # Warmup không được vượt quá total_steps
    effective_warmup = min(cfg.train.warmup_steps, max(1, total_steps // 2))

    # Warmup → Cosine
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=effective_warmup,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max= max(1,total_steps - effective_warmup),
        eta_min=1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[effective_warmup],
    )

    return optimizer, scheduler


# ──────────────────────────────────────────────
# Main Train Function
# ──────────────────────────────────────────────
def train(cfg: Config):
    config_dict = {k: vars(v) if hasattr(v, '__dict__') else v for k, v in vars(cfg).items()}
    wandb.init(
        project = "mammo-transformer",
        name = cfg.train.experiment_name,
        config = config_dict,
        tags = ["phase1","phase2"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}")

    # ── Data
    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=cfg.data.csv_path,
        image_size=cfg.model.backbone_img_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
        positive_birads=cfg.data.positive_birads,
        aug_level=cfg.data.aug_level,
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
    ).to(device)

    # WANDB: Theo dõi gradients và weights của model
    wandb.watch(model, log="all", log_freq=100)

    param_counts = model.count_parameters()
    print(f"[Model] Backbone: {param_counts['backbone']:,} params | "
          f"Other: {param_counts['other']:,} params | "
          f"Total: {param_counts['total']:,} params")

    # ── Loss
    criterion = FocalLoss(
        alpha=cfg.train.focal_alpha,
        gamma=cfg.train.focal_gamma,
    )

    # ── AMP Scaler
    scaler = GradScaler(enabled=cfg.train.mixed_precision)

    # ── Checkpoint Manager
    ckpt_manager = CheckpointManager(
        output_dir=os.path.join(cfg.train.output_dir, cfg.train.experiment_name),
        save_top_k=cfg.train.save_top_k,
    )

    # ── History
    history = {"phase1": [], "phase2": []}
    best_auc = 0.0
    no_improve = 0
    global_epoch = 0

    # ════════════════════════════════
    # PHASE 1: Freeze backbone
    # ════════════════════════════════
    print("\n" + "="*60)
    print("  PHASE 1 — Frozen backbone, training attention + classifier")
    print("="*60)

    model.freeze_backbone()

    steps_p1 = len(train_loader) * cfg.train.epochs_phase1
    optimizer1, scheduler1 = build_optimizer_scheduler(model, cfg.train.lr_phase1, cfg, steps_p1, phase=1)

    for epoch in range(1, cfg.train.epochs_phase1 + 1):
        global_epoch += 1
        t0 = time.time()
        print(f"\n── Epoch {epoch}/{cfg.train.epochs_phase1} [Phase 1]")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer1, criterion, scaler,
            device, cfg.train.accumulate_grad_steps, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device)

        current_lr = optimizer1.param_groups[0]["lr"]
        scheduler1.step()

        epoch_log = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
                     "time": round(time.time() - t0, 1)}
        history["phase1"].append(epoch_log)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            ckpt_manager.save(model, optimizer1, epoch, val_metrics, "p1")
            no_improve = 0
        else:
            no_improve += 1

        print(f"  [Phase 1] Epoch {epoch} | Val AUC: {val_metrics['auc']:.4f} | "
              f"Best: {best_auc:.4f} | Time: {epoch_log['time']}s")

        wandb.log({
            "epoch": global_epoch,
            "phase": 1,
            "lr": current_lr,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()}
        })

    # ════════════════════════════════
    # PHASE 2: Unfreeze backbone
    # ════════════════════════════════
    print("\n" + "="*60)
    print("  PHASE 2 — Full fine-tune (backbone LR × 0.1)")
    print("="*60)

    model.unfreeze_backbone()
    no_improve = 0

    steps_p2 = len(train_loader) * cfg.train.epochs_phase2
    optimizer2, scheduler2 = build_optimizer_scheduler(model, cfg.train.lr_phase2, cfg, steps_p2, phase=2)

    for epoch in range(1, cfg.train.epochs_phase2 + 1):
        global_epoch += 1
        t0 = time.time()
        print(f"\n── Epoch {epoch}/{cfg.train.epochs_phase2} [Phase 2]")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer2, criterion, scaler,
            device, cfg.train.accumulate_grad_steps, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device)
        current_lr = optimizer2.param_groups[0]["lr"]
        scheduler2.step()

        epoch_log = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
                     "time": round(time.time() - t0, 1)}
        history["phase2"].append(epoch_log)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            ckpt_manager.save(model, optimizer2, epoch, val_metrics, "p2")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.train.early_stopping_patience:
                print(f"\n[Early Stopping] No improvement for {no_improve} epochs.")
                break

        print(f"  [Phase 2] Epoch {epoch} | Val AUC: {val_metrics['auc']:.4f} | "
              f"Best: {best_auc:.4f} | Time: {epoch_log['time']}s")

        wandb.log({
            "epoch": global_epoch,
            "phase": 2,
            "lr": current_lr,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()}
        })

    # ════════════════════════════════
    # FINAL TEST EVALUATION
    # ════════════════════════════════
    print("\n" + "="*60)
    print("  FINAL TEST EVALUATION")
    print("="*60)

    best_ckpt = ckpt_manager.best_checkpoint()
    if best_ckpt:
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best checkpoint: {best_ckpt.name}")

    test_metrics = validate(model, test_loader, criterion, device)
    print("\n  ===== TEST SET FINAL RESULTS =====")
    for k, v in test_metrics.items():
        print(f"    {k}: {v}")

    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    
    # Save final model
    final_model_path = Path(cfg.train.output_dir) / cfg.train.experiment_name / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "test_metrics": test_metrics,
    }, final_model_path)
    print(f"[Done] Final model saved to {final_model_path}")

    # Save history
    history_path = Path(cfg.train.output_dir) / cfg.train.experiment_name / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Done] History saved to {history_path}")

    wandb.finish()

    return model, history, test_metrics


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from configs.config import Config
    cfg = Config()

    # Override nếu cần
    cfg.data.csv_path    = "labels.csv"
    cfg.data.data_root   = "images_cropped"
    cfg.train.output_dir = "./outputs"

    train(cfg)
