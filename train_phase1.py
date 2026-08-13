"""
PHASE 1 — Contrastive pretraining cho Swin-V2 backbone.

4 view của cùng 1 bệnh nhân = positive pair; view của bệnh nhân khác = negative.
Backbone được TRAIN (unfrozen) ở phase này. Kết thúc, weight backbone tốt nhất
được lưu ra `outputs/<exp>/phase1_backbone.pt` để train_phase2.py nạp vào.

Chạy:
    python train_phase1.py
"""

import json
import os
import time
from pathlib import Path

# PHẢI đặt TRƯỚC khi import torch — biến này chỉ có tác dụng lúc CUDA khởi tạo.
# expandable_segments giảm phân mảnh của caching allocator: reserved bám sát
# allocated hơn, thay vì giữ thừa 30-40%. Rất quan trọng khi dùng chung GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from configs.config import Config
from data.dataset import build_dataloaders, VIEW_KEYS
from models.mammo_transformer import SwinV2Backbone
from models.projection_head import ProjectionHead
from utils.Supcon_loss import MultiViewNTXentLoss
from utils.wandb_utils import WandbLogger


# ──────────────────────────────────────────────
# Scheduler: warmup tuyến tính → cosine, tính theo STEP
# ──────────────────────────────────────────────
def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    effective_warmup = max(1, min(warmup_steps, total_steps // 2))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=effective_warmup)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - effective_warmup), eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[effective_warmup])


# ──────────────────────────────────────────────
# Gộp dict 4 view → tensor (B*V, 3, H, W)
# ──────────────────────────────────────────────
def stack_views(images_dict, device):
    images = torch.stack([images_dict[k] for k in VIEW_KEYS], dim=1)   # (B, V, 3, H, W)
    B, V, C, H, W = images.shape
    return images.reshape(B * V, C, H, W).to(device, non_blocking=True), B, V


# ──────────────────────────────────────────────
# Train 1 epoch
# ──────────────────────────────────────────────
def train_one_epoch(
    backbone, proj_head, loader, optimizer, scheduler, criterion, scaler,
    device, epoch, cfg, logger, global_step, amp_dtype,
):
    backbone.train()
    proj_head.train()

    total_loss, n_batches = 0.0, 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        images, B, V = stack_views(batch["images"], device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
            features = backbone(images)          # (B*V, D) — global-pooled
            z = proj_head(features)              # (B*V, out_dim)
            z = z.reshape(B, V, -1)
            loss = criterion(z)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(proj_head.parameters()),
            max_norm=cfg.train.grad_clip,
        )
        # GradScaler BỎ QUA optimizer.step() khi gradient inf/nan (hay xảy ra ở
        # vài step đầu khi scale còn cao). Nếu vẫn gọi scheduler.step() thì LR
        # schedule lệch pha so với số bước thật — và PyTorch cảnh báo.
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        
        if scaler.get_scale() >= scale_before:    # step thật sự đã chạy
            scheduler.step()                      # ← per-STEP, khớp total_steps

        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1
        global_step += 1

        if (step + 1) % cfg.train.log_every_n_steps == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [P1] Epoch {epoch} | Step {step+1}/{len(loader)} | "
                  f"loss {loss_val:.4f} | avg {total_loss/n_batches:.4f} | lr {lr_now:.2e}")
            logger.log({"train/step_loss": loss_val, "train/lr": lr_now, "train/amp_scale": scale_after,       
            "train/scale_dropped": scale_after < scale_before,}, step=global_step)

    return total_loss / max(1, n_batches), global_step, time.time() - t0


# ──────────────────────────────────────────────
# Validate (contrastive loss trên val set, không augment)
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(backbone, proj_head, loader, criterion, device, scaler, amp_dtype):
    backbone.eval()
    proj_head.eval()

    total_loss, n_batches = 0.0, 0
    for batch in loader:
        images, B, V = stack_views(batch["images"], device)
        if B < 2:
            continue                              # cần ít nhất 2 bệnh nhân để có negative
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
            z = proj_head(backbone(images)).reshape(B, V, -1)
            loss = criterion(z)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32
    print(f"\n[Phase 1] Device: {device} | AMP: {use_amp}")

    out_dir = Path(cfg.train.output_dir) / cfg.train.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(cfg, phase="phase1")

    # ── Data
    train_loader, val_loader, _ = build_dataloaders(
        csv_path=cfg.data.csv_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.train.batch_size_phase1,
        num_workers=cfg.data.num_workers,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
        aug_level=cfg.data.aug_level,
        num_classes=cfg.data.num_classes,
        persistent_workers=cfg.data.persistent_workers,
    )

    # ── Model
    backbone = SwinV2Backbone(
        model_name=cfg.model.backbone_name,
        pretrained=cfg.train.pretrained_phase1,
        img_size=cfg.model.backbone_img_size,     # (H, W)
        token_grid=cfg.model.token_grid,
        grad_checkpointing=True,                  # cần thiết vì backbone unfrozen
    ).to(device)
    backbone.unfreeze()

    proj_head = ProjectionHead(
        input_dim=backbone.out_dim,
        hidden_dim=cfg.train.proj_hidden_dim,
        out_dim=cfg.train.proj_out_dim,
    ).to(device)

    n_bb = sum(p.numel() for p in backbone.parameters())
    n_ph = sum(p.numel() for p in proj_head.parameters())
    print(f"[Phase 1] Backbone: {n_bb:,} params | ProjectionHead: {n_ph:,} params")
    if cfg.wandb.watch_model:
        logger.watch(backbone, log_freq=cfg.wandb.log_every_n_steps)

    # ── Loss / Optim / Sched / AMP
    criterion = MultiViewNTXentLoss(temperature=cfg.train.temperature)
    optimizer = AdamW(
        list(backbone.parameters()) + list(proj_head.parameters()),
        lr=cfg.train.lr_phase1,
        weight_decay=cfg.train.weight_decay,
    )
    total_steps = max(1, len(train_loader) * cfg.train.epochs_phase1)
    scheduler = build_scheduler(optimizer, cfg.train.warmup_steps, total_steps)
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # ── Loop
    ckpt_path = out_dir / cfg.train.phase1_ckpt_name
    best_val = float("inf")
    no_improve = 0
    global_step = 0
    history = []

    for epoch in range(1, cfg.train.epochs_phase1 + 1):
        print(f"\n── [Phase 1] Epoch {epoch}/{cfg.train.epochs_phase1}")

        train_loss, global_step, elapsed = train_one_epoch(
            backbone, proj_head, train_loader, optimizer, scheduler,
            criterion, scaler, device, epoch, cfg, logger, global_step, amp_dtype,
        )
        val_loss = validate(backbone, proj_head, val_loader, criterion, device, scaler, amp_dtype)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"  [P1] Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr_now:.2e} | {elapsed:.0f}s")

        logger.log_epoch(epoch, {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/epoch_lr": lr_now,
            "train/epoch_time_s": elapsed,
        })
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "time": round(elapsed, 1)})

        # ── Lưu backbone tốt nhất (đây là cầu nối sang phase 2)
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),
                "projection_head_state_dict": proj_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "backbone_name": cfg.model.backbone_name,
                "img_size": cfg.model.backbone_img_size,
            }, ckpt_path)
            print(f"  [P1] ✔ Lưu backbone tốt nhất → {ckpt_path.name} (val {val_loss:.4f})")
            logger.set_summary({"best_val_loss": best_val, "best_epoch": epoch})
        else:
            no_improve += 1
            if no_improve >= cfg.train.early_stopping_patience:
                print(f"  [P1] Early stop — không cải thiện {no_improve} epoch.")
                break

    # ── Kết
    with open(out_dir / "history_phase1.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n[Phase 1] Xong. Best val loss: {best_val:.4f}")
    print(f"[Phase 1] Backbone lưu tại: {ckpt_path}")
    logger.log_artifact(ckpt_path, name=f"{cfg.train.experiment_name}-phase1-backbone")
    logger.finish()

    return ckpt_path


# ──────────────────────────────────────────────
# Entry point — BẮT BUỘC bọc trong __main__ vì num_workers > 0 trên Windows
# dùng spawn, sẽ re-import module này ở mỗi worker.
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main(Config())
