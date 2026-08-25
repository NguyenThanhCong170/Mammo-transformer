"""
PHASE 1 — Contrastive pretraining cho Swin-V2 backbone.

4 view của cùng 1 bệnh nhân = positive pair; view của bệnh nhân khác = negative.
Backbone được TRAIN (unfrozen) ở phase này. Kết thúc, weight backbone tốt nhất
được lưu ra `outputs/<exp>/phase1_backbone.pt` để train_phase2.py nạp vào.

Chạy:
    python train_phase1.py
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

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
from utils.Supcon_loss import MultiViewNTXentLoss, build_view_groups
from utils.wandb_utils import WandbLogger


# ──────────────────────────────────────────────
# Seed — bắt buộc nếu muốn so sánh các nhánh ablation
# ──────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    device, epoch, cfg, logger, global_step, amp_dtype, use_amp, params,
):
    backbone.train()
    proj_head.train()

    total_loss, n_batches = 0.0, 0
    n_skipped, n_nonfinite = 0, 0          # chẩn đoán sức khoẻ AMP
    t0 = time.time()

    for step, batch in enumerate(loader):
        images, B, V = stack_views(batch["images"], device)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward: backbone + head chạy ở half precision
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            features = backbone(images)          # (B*V, D) — global-pooled
            z = proj_head(features)              # (B*V, out_dim)

        # ── Loss LUÔN tính ở fp32.
        # KHÔNG đủ nếu chỉ gọi .float() bên trong autocast: torch.matmul nằm
        # trong autocast list nên PyTorch ép ngược nó về half. Phải tắt autocast
        # tường minh. NT-Xent chia cho temperature=0.1 (khuếch đại 10x) nên đây
        # là chỗ dễ tràn số nhất trong toàn bộ pipeline.
        with autocast(device_type=device.type, enabled=False):
            loss = criterion(z.float().reshape(B, V, -1))

        # Lưới an toàn: nếu loss đã NaN/Inf thì backward chỉ làm hỏng weight.
        if not torch.isfinite(loss):
            n_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            if n_nonfinite <= 5:
                print(f"  [P1] ⚠ loss không hữu hạn ở step {step+1} → bỏ qua batch này.")
            continue

        scaler.scale(loss).backward()

        # unscale_ chỉ có nghĩa khi đang dùng fp16 + GradScaler.
        # Với bf16 scaler bị tắt, gradient vốn đã ở scale thật.
        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.train.grad_clip)

        # GradScaler BỎ QUA optimizer.step() khi gradient inf/nan. Nếu vẫn gọi
        # scheduler.step() thì LR schedule lệch pha so với số bước thật.
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        stepped = (not scaler.is_enabled()) or (scale_after >= scale_before)
        if stepped:
            scheduler.step()                      # ← per-STEP, khớp total_steps
        else:
            n_skipped += 1

        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1
        global_step += 1

        if (step + 1) % cfg.train.log_every_n_steps == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            gn = float(grad_norm)
            print(f"  [P1] Epoch {epoch} | Step {step+1}/{len(loader)} | "
                  f"loss {loss_val:.4f} | avg {total_loss/n_batches:.4f} | "
                  f"lr {lr_now:.2e} | grad_norm {gn:.2f}")
            logger.log({
                "train/step_loss": loss_val,
                "train/lr": lr_now,
                "train/grad_norm": gn if math.isfinite(gn) else -1.0,
                "train/amp_scale": scale_after,
                "train/scale_dropped": float(not stepped),
            }, step=global_step)

    stats = {
        "skipped": n_skipped,
        "nonfinite": n_nonfinite,
        "total": n_batches + n_nonfinite,
    }
    return total_loss / max(1, n_batches), global_step, time.time() - t0, stats


# ──────────────────────────────────────────────
# Validate (contrastive loss trên val set, không augment)
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(backbone, proj_head, loader, criterion, device, amp_dtype, use_amp):
    backbone.eval()
    proj_head.eval()

    total_loss, n_batches = 0.0, 0
    for batch in loader:
        images, B, V = stack_views(batch["images"], device)
        if B < 2:
            continue                              # cần ít nhất 2 bệnh nhân để có negative
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            z = proj_head(backbone(images))
        with autocast(device_type=device.type, enabled=False):
            loss = criterion(z.float().reshape(B, V, -1))
        if not torch.isfinite(loss):
            continue
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.mixed_precision and device.type == "cuda"

    # ── Chọn dtype cho AMP ────────────────────────────────────────────────
    # bfloat16 có DẢI MŨ y hệt fp32 (8 bit exponent) nên không bao giờ tràn số,
    # và vì thế KHÔNG cần loss scaling → GradScaler bị tắt hoàn toàn.
    # fp16 chỉ biểu diễn tới ~65504; cosine-attention của Swin-V2 (logit_scale
    # exp tới 100) vượt ngưỡng đó → gradient inf → GradScaler tụt về 2^-11 →
    # gradient underflow về 0 → model không học được gì. Đó là lỗi của run cũ.
    use_bf16 = use_amp and getattr(cfg.train, "prefer_bf16", True) and torch.cuda.is_bf16_supported()
    if use_bf16:
        amp_dtype = torch.bfloat16
    elif use_amp:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    # fp16 mới cần scaler; bf16 và fp32 thì không.
    scaler_enabled = use_amp and not use_bf16

    print(f"\n[Phase 1] Device: {device} | AMP: {use_amp} | dtype: {amp_dtype}")
    if use_amp and not use_bf16:
        print("[Phase 1] ⚠ GPU không hỗ trợ bfloat16 → dùng fp16 + GradScaler(init_scale=1024). "
              "Theo dõi train/amp_scale: nếu tụt xuống dưới 1.0 thì AMP đang hỏng, "
              "hãy đặt cfg.train.mixed_precision = False.")

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
    # ── Cách ghép positive — đây là trục ablation của phase 1.
    # "patient"     : cả 4 view cùng bệnh nhân là positive (bản gốc)
    # "ipsilateral" : chỉ cùng bên vú; vú đối bên thành hard negative
    # view_groups suy ra từ VIEW_KEYS của dataset nên luôn khớp thứ tự
    # mà stack_views() xếp tensor.
    mode = getattr(cfg.train, "contrastive_positives", "patient")
    view_groups = build_view_groups(VIEW_KEYS, mode)
    criterion = MultiViewNTXentLoss(
        temperature=cfg.train.temperature, view_groups=view_groups)
    print(f"[Phase 1] Positive pairing: {mode}  |  VIEW_KEYS={VIEW_KEYS}  "
          f"|  view_groups={view_groups}")
    if view_groups is None:
        print("[Phase 1] CANH BAO: che do 'patient' keo vu trai va vu phai lai gan nhau, "
              "pha tin hieu cua lop asymmetry. Dung --positives ipsilateral.")
    # Danh sách param dựng 1 lần — trước đây build lại mỗi step trong clip_grad_norm_.
    params = list(backbone.parameters()) + list(proj_head.parameters())
    optimizer = AdamW(
        params,
        lr=cfg.train.lr_phase1,
        weight_decay=cfg.train.weight_decay,
    )
    total_steps = max(1, len(train_loader) * cfg.train.epochs_phase1)
    scheduler = build_scheduler(optimizer, cfg.train.warmup_steps, total_steps)
    # init_scale=1024 thay vì mặc định 65536: khởi đầu thấp hơn nên không phải
    # đốt hàng chục step đầu để halve dần xuống vùng an toàn.
    scaler = GradScaler(device=device.type, enabled=scaler_enabled, init_scale=1024.0)
    print(f"[Phase 1] GradScaler: {'BẬT (fp16)' if scaler_enabled else 'TẮT (không cần với bf16/fp32)'}")
    print(f"[Phase 1] LR: {cfg.train.lr_phase1:.1e} | warmup: {cfg.train.warmup_steps} step "
          f"| total: {total_steps} step ({len(train_loader)} step/epoch)")

    # ── Loop
    ckpt_path = out_dir / cfg.train.phase1_ckpt_name
    best_val = float("inf")
    no_improve = 0
    global_step = 0
    history = []

    for epoch in range(1, cfg.train.epochs_phase1 + 1):
        print(f"\n── [Phase 1] Epoch {epoch}/{cfg.train.epochs_phase1}")

        train_loss, global_step, elapsed, stats = train_one_epoch(
            backbone, proj_head, train_loader, optimizer, scheduler,
            criterion, scaler, device, epoch, cfg, logger, global_step,
            amp_dtype, use_amp, params,
        )
        val_loss = validate(backbone, proj_head, val_loader, criterion, device, amp_dtype, use_amp)
        lr_now = optimizer.param_groups[0]["lr"]

        skip_rate = (stats["skipped"] + stats["nonfinite"]) / max(1, stats["total"])
        print(f"  [P1] Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"lr {lr_now:.2e} | {elapsed:.0f}s")
        print(f"  [P1] AMP: bỏ qua {stats['skipped']} step (inf/nan grad), "
              f"{stats['nonfinite']} batch loss không hữu hạn "
              f"→ {skip_rate*100:.1f}% | scale hiện tại {scaler.get_scale():.4g}")
        if skip_rate > 0.05:
            print("  [P1] ⚠ Trên 5% step bị bỏ qua — AMP đang không ổn định. "
                  "Cân nhắc tắt mixed_precision hoặc hạ LR thêm.")

        logger.log_epoch(epoch, {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/epoch_lr": lr_now,
            "train/epoch_time_s": elapsed,
            "train/epoch_skip_rate": skip_rate,
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
# CLI
# ──────────────────────────────────────────────
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1 — contrastive pretrain backbone. Moi co deu ghi de config.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUAN TRONG: --exp-name quyet dinh thu muc ghi phase1_backbone.pt.
Neu khong doi, run moi se DE LEN checkpoint phase 1 cu.

  # Nhanh moi: chi ghep positive cung ben vu
  python train_phase1.py --exp-name p1_ipsi --positives ipsilateral --seed 42

  # Nhanh cu (de tai lap): ca 4 view cung benh nhan
  python train_phase1.py --exp-name p1_patient --positives patient --seed 42

Sau do danh gia bang phase 2, dung cung mot cau hinh head cho ca ba nhanh:

  python train_phase2.py --exp-name abl_ipsi --phase1 \\
      --phase1-ckpt outputs/p1_ipsi/phase1_backbone.pt \\
      --embed-dim 256 --ffn-expansion 2 --lr 1e-4 --grad-clip 0.5 --epochs 30 --seed 42
""")
    p.add_argument("--positives", choices=["patient", "ipsilateral"], default=None,
                   help="Cach ghep positive. ipsilateral = chi cung ben vu, "
                        "vu doi ben thanh hard negative.")
    p.add_argument("--exp-name", type=str, default=None,
                   help="Ten thi nghiem — quyet dinh thu muc output. DAT KHAC NHAU "
                        "cho moi nhanh, neu khong se de len checkpoint cu.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="lr_phase1")
    p.add_argument("--batch-size", type=int, default=None, help="batch_size_phase1")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    t, d, w = cfg.train, cfg.data, cfg.wandb
    if args.positives is not None:     t.contrastive_positives = args.positives
    if args.exp_name is not None:      t.experiment_name = args.exp_name
    if args.seed is not None:          d.seed = args.seed
    if args.epochs is not None:        t.epochs_phase1 = args.epochs
    if args.lr is not None:            t.lr_phase1 = args.lr
    if args.batch_size is not None:    t.batch_size_phase1 = args.batch_size
    if args.temperature is not None:   t.temperature = args.temperature
    if args.warmup_steps is not None:  t.warmup_steps = args.warmup_steps
    if args.grad_clip is not None:     t.grad_clip = args.grad_clip
    if args.patience is not None:      t.early_stopping_patience = args.patience
    if args.no_wandb:                  w.enabled = False
    return cfg


# ──────────────────────────────────────────────
# Entry point — BẮT BUỘC bọc trong __main__ vì num_workers > 0 trên Windows
# dùng spawn, sẽ re-import module này ở mỗi worker.
# ──────────────────────────────────────────────
if __name__ == "__main__":
    _args = build_argparser().parse_args()
    _cfg = apply_overrides(Config(), _args)
    set_seed(_cfg.data.seed)

    _ckpt = Path(_cfg.train.output_dir) / _cfg.train.experiment_name / _cfg.train.phase1_ckpt_name
    print("=" * 60)
    print(f"  exp_name   : {_cfg.train.experiment_name}")
    print(f"  positives  : {_cfg.train.contrastive_positives}")
    print(f"  lr / epochs: {_cfg.train.lr_phase1:.1e} / {_cfg.train.epochs_phase1}")
    print(f"  batch / T  : {_cfg.train.batch_size_phase1} / {_cfg.train.temperature}")
    print(f"  seed       : {_cfg.data.seed}")
    print(f"  se ghi ra  : {_ckpt}")
    if _ckpt.exists():
        print(f"  CANH BAO: file nay DA TON TAI va se bi ghi de. "
              f"Doi --exp-name neu muon giu lai.")
    print("=" * 60)

    main(_cfg)
