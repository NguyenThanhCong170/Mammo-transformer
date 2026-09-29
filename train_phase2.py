

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

# PHẢI đặt TRƯỚC khi import torch (xem giải thích trong train_phase1.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from configs.config import Config
from data.dataset import build_dataloaders, VIEW_KEYS as DATA_VIEW_KEYS
from models.mammo_transformer import MammoTransformer, VIEW_KEYS as MODEL_VIEW_KEYS
from models.projection_head import ProjectionHead

# VIEW_KEYS bị khai báo trùng ở hai file. SupCon ghép z[i] với view_labels[i]
# theo đúng thứ tự này, nên nếu hai bản lệch nhau thì nhãn sẽ gán sai ảnh —
# một lỗi câm, không crash, chỉ làm kết quả sai. Chặn ngay lúc import.
assert list(DATA_VIEW_KEYS) == list(MODEL_VIEW_KEYS), (
    f"VIEW_KEYS lech nhau!\n  data/dataset.py        : {list(DATA_VIEW_KEYS)}\n"
    f"  models/mammo_transformer.py: {list(MODEL_VIEW_KEYS)}"
)
from utils.losses import FocalLoss, MultiLabelMetricsCalculator
from utils.Supcon_loss import MultiLabelSupConLoss
from utils.wandb_utils import WandbLogger, flatten_metrics

CLASS_NAMES = ["no_finding", "mass", "calcification"]


# ──────────────────────────────────────────────
# Seed — BẮT BUỘC cho ablation
# ──────────────────────────────────────────────
def set_seed(seed: int):
    """
    Cố định seed cho khởi tạo trọng số, augmentation và shuffle.

    Với ablation A/B (có phase-1 vs ImageNet), nếu KHÔNG seed thì hai run có
    head khởi tạo khác nhau và thứ tự batch khác nhau — một phần chênh lệch
    macro-AP sẽ là nhiễu, không phải hiệu ứng của phase 1. Trước đây file này
    không seed gì cả.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    device, cfg, epoch, logger, global_step, amp_dtype, use_amp, trainable_params,
    supcon=None, proj_head=None,
):
    model.train()          # override trong MammoTransformer giữ backbone ở eval nếu frozen
    if proj_head is not None:
        proj_head.train()
    total_loss = 0.0
    sum_cls, sum_con, n_con = 0.0, 0.0, 0
    accumulate_steps = cfg.train.accumulate_grad_steps
    n_skipped, n_nonfinite = 0, 0
    grad_norm = torch.tensor(0.0)
    use_supcon = supcon is not None and cfg.train.supcon_weight > 0

    metrics_calc = MultiLabelMetricsCalculator(
        num_classes=cfg.data.num_classes,
        class_names=CLASS_NAMES,
    )
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for step, batch in enumerate(loader):
        images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
        labels = batch["label"].to(device, non_blocking=True)

        # Forward ở half precision; loss LUÔN ở fp32.
        # FocalLoss có log/exp và sigmoid — dưới fp16 rất dễ mất chính xác ở
        # đuôi phân phối, mà focal loss thì sống bằng đúng cái đuôi đó.
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if use_supcon:
                logits, view_feats, _ = model(images, return_features=True)
                z = proj_head(view_feats.reshape(-1, view_feats.shape[-1]))
            else:
                logits = model(images)

        with autocast(device_type=device.type, enabled=False):
            loss_cls = criterion(logits.float(), labels)
            if use_supcon:
                vl = batch["view_labels"].to(device, non_blocking=True)
                loss_con = supcon(z.float(), vl.reshape(-1, vl.shape[-1]).float())
                loss = loss_cls + cfg.train.supcon_weight * loss_con
                sum_con += float(loss_con.detach()); n_con += 1
            else:
                loss_con = None
                loss = loss_cls
            sum_cls += float(loss_cls.detach())
            loss = loss / accumulate_steps

        if not torch.isfinite(loss):
            n_nonfinite += 1
            if n_nonfinite <= 5:
                print(f"  [P2] ⚠ loss không hữu hạn ở step {step+1} → bỏ qua batch này.")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (step + 1) % accumulate_steps == 0 or (step + 1) == len(loader):
            # unscale_ chỉ có nghĩa với fp16 + GradScaler; bf16 thì scaler tắt.
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=cfg.train.grad_clip)
            # Xem giải thích trong train_phase1.py: scaler có thể bỏ qua step.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            stepped = (not scaler.is_enabled()) or (scaler.get_scale() >= scale_before)
            if stepped:
                scheduler.step()           # ← theo OPTIMIZER STEP, không theo epoch
            else:
                n_skipped += 1
            global_step += 1

        batch_loss = loss.item() * accumulate_steps
        total_loss += batch_loss
        metrics_calc.update(logits.detach().float(), labels)

        if (step + 1) % cfg.train.log_every_n_steps == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            gn = float(grad_norm)
            extra = ""
            payload = {
                "train/step_loss": batch_loss,
                "train/lr": lr_now,
                "train/grad_norm": gn if math.isfinite(gn) else -1.0,
                "train/amp_scale": scaler.get_scale(),
            }
            if use_supcon and loss_con is not None:
                lc, lo = float(loss_cls.detach()), float(loss_con.detach())
                w = cfg.train.supcon_weight
                # In cả tỉ lệ đóng góp — đây là con số dùng để chỉnh
                # --supcon-weight. Nên nằm quanh 0.2–0.5.
                extra = f" | cls {lc:.4f} | con {lo:.3f} (x{w:g} = {w*lo:.4f}, {w*lo/max(lc,1e-9):.2f}x cls)"
                payload.update({
                    "train/step_loss_cls": lc,
                    "train/step_loss_con": lo,
                    "train/step_con_ratio": w * lo / max(lc, 1e-9),
                })
            print(f"  [P2] Epoch {epoch} | Step {step+1}/{len(loader)} | "
                  f"loss {batch_loss:.4f} | avg {total_loss/(step+1):.4f} | "
                  f"lr {lr_now:.2e} | grad_norm {gn:.2f}{extra}")
            logger.log(payload, step=global_step)

    metrics = metrics_calc.compute()
    metrics["loss"] = total_loss / max(1, len(loader))
    metrics["loss_cls"] = sum_cls / max(1, len(loader))
    if n_con:
        metrics["loss_con"] = sum_con / n_con
    metrics["time"] = time.time() - t0
    metrics["amp_skipped"] = n_skipped
    metrics["amp_nonfinite"] = n_nonfinite
    return metrics, global_step


# ──────────────────────────────────────────────
# Validate / Test
# ──────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, cfg, amp_dtype, use_amp,
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

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
        with autocast(device_type=device.type, enabled=False):
            loss = criterion(logits.float(), labels)

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
def build_optimizer_scheduler(model, lr: float, cfg: Config, total_optim_steps: int,
                              extra_module=None):
    backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
    if backbone_trainable:
        # LR phân tầng: backbone (đã pretrain) đi chậm hơn head (ngẫu nhiên).
        # get_param_groups() đã có sẵn trong mammo_transformer.py nhưng trước
        # đây không ai gọi — hàm này lọc bỏ hẳn param backbone.
        groups = model.get_param_groups(
            lr, backbone_lr_multiplier=cfg.train.backbone_lr_multiplier)
        # ProjectionHead của SupCon đi cùng LR với head (nó cũng khởi tạo ngẫu nhiên)
        if extra_module is not None:
            groups[0]["params"] = list(groups[0]["params"]) + list(extra_module.parameters())
        optimizer = AdamW(groups, weight_decay=cfg.train.weight_decay)
        n_g = [(len(g["params"]), g["lr"]) for g in groups]
        print(f"[Optim] Param groups: " + " | ".join(f"{n} tensor @ lr {l:.1e}" for n, l in n_g))
    else:
        params = [p for n, p in model.named_parameters()
                  if not n.startswith("backbone") and p.requires_grad]
        if extra_module is not None:
            params += [p for p in extra_module.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=lr, weight_decay=cfg.train.weight_decay)
        print(f"[Optim] Backbone dong bang — train {len(params)} tensor @ lr {lr:.1e}")

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

    # ── Chọn dtype cho AMP (giống hệt train_phase1.py) ────────────────────
    # bfloat16 có dải mũ bằng fp32 → không tràn số → KHÔNG cần GradScaler.
    # fp16 chỉ tới ~65504; cosine-attention của Swin-V2 vượt ngưỡng đó và làm
    # GradScaler tụt dần tới mức gradient underflow về 0.
    use_bf16 = use_amp and getattr(cfg.train, "prefer_bf16", True) and torch.cuda.is_bf16_supported()
    if use_bf16:
        amp_dtype = torch.bfloat16
    elif use_amp:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    scaler_enabled = use_amp and not use_bf16

    print(f"\n[Phase 2] Device: {device} | AMP: {use_amp} | dtype: {amp_dtype}")
    if use_amp and not use_bf16:
        print("[Phase 2] CANH BAO: GPU khong ho tro bfloat16 -> dung fp16 + GradScaler. "
              "Theo doi train/amp_scale: tut xuong duoi 1.0 la AMP dang hong.")

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

    # ── Khởi tạo backbone: phase-1 contrastive HAY ImageNet
    # Đây chính là trục ablation. --no-phase1 đi nhánh else.
    if cfg.train.load_phase1_backbone:
        phase1_ckpt = Path(cfg.train.phase1_ckpt_path) if cfg.train.phase1_ckpt_path \
            else Path(cfg.train.output_dir) / cfg.train.experiment_name / cfg.train.phase1_ckpt_name
        if phase1_ckpt.exists():
            model.load_backbone_weights(str(phase1_ckpt), device=device)
            backbone_init = f"phase1 ({phase1_ckpt.name})"
        else:
            # DỪNG chứ không âm thầm rơi về ImageNet: nếu không, một run
            # "có phase-1" thực chất lại là run ImageNet và ablation vô nghĩa.
            raise FileNotFoundError(
                f"Khong tim thay checkpoint phase 1: {phase1_ckpt}\n"
                f"  - Chay train_phase1.py truoc, HOAC\n"
                f"  - Chi ro duong dan:  --phase1-ckpt <path>, HOAC\n"
                f"  - Co y dung ImageNet: --no-phase1"
            )
    else:
        backbone_init = "ImageNet" if cfg.model.backbone_pretrained else "random"
        print(f"[Phase 2] Bo qua checkpoint phase 1 — backbone khoi tao tu {backbone_init}.")
    model.to(device)

    # ── Freeze / unfreeze backbone
    # Ba chế độ, ưu tiên --unfreeze-from-stage nếu được đặt:
    #   None (mặc định)        → backbone đóng băng hoàn toàn
    #   --unfreeze-from-stage 2 → chỉ train stage 3+4 (khuyến nghị)
    #   --unfreeze-backbone     → mở toàn bộ (tương đương stage 0)
    print("\n" + "=" * 60)
    stage = cfg.train.unfreeze_from_stage
    if stage is None and cfg.train.unfreeze_backbone:
        stage = 0
    if stage is not None:
        label = "TOAN BO" if stage == 0 else f"tu stage {stage + 1}"
        print(f"  Backbone init: {backbone_init}  |  FINE-TUNE {label}")
        print("=" * 60)
        model.unfreeze_backbone_from_stage(stage)
    else:
        print(f"  Backbone init: {backbone_init}  |  FROZEN")
        print("  Train cross-attention + classifier")
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

    # ── SupCon (tuỳ chọn). supcon_weight = 0 → tắt hoàn toàn, không tạo head.
    supcon, proj_head = None, None
    if cfg.train.supcon_weight > 0:
        supcon = MultiLabelSupConLoss(
            temperature=cfg.train.supcon_temperature,
            exclude_classes=(0,) if cfg.train.supcon_exclude_no_finding else (),
        ).to(device)
        proj_head = ProjectionHead(
            input_dim=model.backbone.out_dim,
            hidden_dim=cfg.train.supcon_proj_hidden,
            out_dim=cfg.train.supcon_proj_dim,
        ).to(device)
        n_anchor = cfg.train.batch_size * 4
        print(f"[Phase 2] SupCon: BAT | weight {cfg.train.supcon_weight:g} "
              f"| T {cfg.train.supcon_temperature} "
              f"| loai no_finding khoi positive: {cfg.train.supcon_exclude_no_finding} "
              f"| {n_anchor} anchor/batch")
        print(f"[Phase 2] CHINH --supcon-weight theo cot 'con_ratio' trong log: "
              f"nen quanh 0.2-0.5. FocalLoss ~0.04 con SupCon ~1-4 nen weight "
              f"hop ly thuong la 0.005-0.02, KHONG phai 1.0.")
    else:
        print("[Phase 2] SupCon: TAT (--supcon-weight 0) — chi FocalLoss.")

    steps_per_epoch = max(1, len(train_loader) // cfg.train.accumulate_grad_steps)
    total_optim_steps = steps_per_epoch * cfg.train.epochs_phase2
    optimizer, scheduler = build_optimizer_scheduler(
        model, cfg.train.lr_phase2, cfg, total_optim_steps, extra_module=proj_head)
    scaler = GradScaler(device=device.type, enabled=scaler_enabled, init_scale=1024.0)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if proj_head is not None:
        trainable_params += [p for p in proj_head.parameters() if p.requires_grad]
    print(f"[Phase 2] GradScaler: {'BAT (fp16)' if scaler_enabled else 'TAT (khong can voi bf16/fp32)'}")
    print(f"[Phase 2] LR: {cfg.train.lr_phase2:.1e} | warmup: {cfg.train.warmup_steps} step "
          f"| total: {total_optim_steps} optim-step ({steps_per_epoch} step/epoch)")

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
            device, cfg, epoch, logger, global_step, amp_dtype, use_amp, trainable_params,
            supcon=supcon, proj_head=proj_head,
        )
        val_metrics = validate(model, val_loader, criterion, device, cfg,
                               amp_dtype, use_amp, split="Val")

        if train_metrics["amp_skipped"] or train_metrics["amp_nonfinite"]:
            print(f"  [P2] AMP: bo qua {train_metrics['amp_skipped']} optim-step, "
                  f"{train_metrics['amp_nonfinite']} batch loss khong huu han.")

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

        # LỖI CŨ: `break` bị comment out nên early stopping không bao giờ chạy —
        # vòng lặp luôn chạy đủ 100 epoch dù val đã ngừng cải thiện từ lâu.
        # Nhánh else cũng in "có cải thiện" sai: nó là nhánh "chưa hết kiên nhẫn",
        # không phải nhánh "val tốt lên".
        # Muốn tắt early stopping thì đặt cfg.train.early_stopping_patience rất lớn.
        if no_improve >= cfg.train.early_stopping_patience:
            print(f"  [P2] Early stop — khong cai thien {no_improve} epoch.")
            break
        elif no_improve > 0:
            print(f"  [P2] Chua cai thien {no_improve}/{cfg.train.early_stopping_patience} epoch.")

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
                            amp_dtype, use_amp, split="Test", thresholds=best_thresholds)
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
# CLI
# ──────────────────────────────────────────────
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 2 — multi-view classification. Moi co deu ghi de config.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du — ablation "phase 1 co ich khong":

  # A: backbone tu phase 1 (contrastive)
  python train_phase2.py --exp-name abl_phase1  --phase1 \\
      --phase1-ckpt outputs/mammo_transformer_v1/phase1_backbone.pt

  # B: backbone tu ImageNet  <-- cai ban dang muon chay
  python train_phase2.py --exp-name abl_imagenet --no-phase1

Hai run PHAI dung cung --seed va cung moi sieu tham so khac,
neu khong thi chenh lech macro-AP la nhieu chu khong phai hieu ung phase 1.

  # Head nho hon + LR thap hon (khuyen nghi sau khi thay run 162M param sup)
  python train_phase2.py --exp-name abl_imagenet --no-phase1 \\
      --embed-dim 256 --ffn-expansion 2 --lr 1e-4 --grad-clip 0.5 --epochs 30

Hai model de so sanh (fine-tune stage 3+4, ImageNet init):

  # M1 — ImageNet thuan, chi FocalLoss
  python train_phase2.py --exp-name ft_pure --no-phase1 --unfreeze-from-stage 2 --embed-dim 256 --ffn-expansion 2 --lr 1e-4 --backbone-lr-mult 0.05 --grad-clip 0.5 --epochs 30 --seed 42

  # M2 — nhu tren, cong them SupCon
  python train_phase2.py --exp-name ft_supcon --no-phase1 --unfreeze-from-stage 2 --embed-dim 256 --ffn-expansion 2 --lr 1e-4 --backbone-lr-mult 0.05 --grad-clip 0.5 --epochs 30 --seed 42 --supcon-weight 0.01""")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--phase1", dest="load_phase1", action="store_true", default=None,
                   help="Nap backbone tu checkpoint phase 1 (mac dinh theo config).")
    g.add_argument("--no-phase1", dest="load_phase1", action="store_false",
                   help="BO QUA phase 1 — backbone khoi tao tu ImageNet.")
    p.add_argument("--phase1-ckpt", type=str, default=None,
                   help="Duong dan toi phase1_backbone.pt. Mac dinh: "
                        "<output_dir>/<experiment_name>/phase1_backbone.pt")

    p.add_argument("--exp-name", type=str, default=None,
                   help="Ten thi nghiem — quyet dinh thu muc output va ten run wandb. "
                        "DAT KHAC NHAU cho moi nhanh ablation de checkpoint khong de len nhau.")
    p.add_argument("--seed", type=int, default=None, help="Seed (mac dinh: data.seed).")

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="lr_phase2")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--patience", type=int, default=None, help="early_stopping_patience")

    p.add_argument("--embed-dim", type=int, default=None,
                   help="Chieu cua khoi fusion. 1024 -> 162M param; 256 -> 7.5M.")
    p.add_argument("--ffn-expansion", type=int, default=None)
    p.add_argument("--ipsi-layers", type=int, default=None)
    p.add_argument("--bilateral-layers", type=int, default=None)
    p.add_argument("--token-grid", type=str, default=None,
                   help='Vi du "8,4" hoac "16,6". "none" = giu nguyen 319 token.')

    p.add_argument("--unfreeze-backbone", action="store_true", default=None,
                   help="Fine-tune TOAN BO backbone (= --unfreeze-from-stage 0).")
    p.add_argument("--unfreeze-from-stage", type=int, default=None, metavar="N",
                   help="Chi mo backbone tu stage N (danh so tu 0). "
                        "2 = train stage 3+4 — KHUYEN NGHI: ~9GB VRAM thay vi ~17GB, "
                        "va stage 1+2 chi giu 2.3%% tham so.")
    p.add_argument("--backbone-lr-mult", type=float, default=None,
                   help="LR cua backbone = lr * he so nay. Mac dinh 0.05.")

    p.add_argument("--supcon-weight", type=float, default=None, metavar="W",
                   help="Cong them W * MultiLabelSupConLoss vao FocalLoss. "
                        "0 = tat (chi FocalLoss). Dai hop ly 0.005-0.02 — "
                        "KHONG phai 1.0, vi FocalLoss o day chi ~0.04.")
    p.add_argument("--supcon-temperature", type=float, default=None)
    p.add_argument("--supcon-proj-dim", type=int, default=None)
    p.add_argument("--supcon-keep-no-finding", action="store_true",
                   help="Giu no_finding trong positive set (mac dinh loai bo).")

    p.add_argument("--no-wandb", action="store_true", help="Tat wandb logging.")
    return p


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Ghi de config bang cac co CLI. None = khong dung toi, giu nguyen config.py."""
    t, m, d, w = cfg.train, cfg.model, cfg.data, cfg.wandb

    if args.load_phase1 is not None:   t.load_phase1_backbone = args.load_phase1
    if args.phase1_ckpt is not None:   t.phase1_ckpt_path = args.phase1_ckpt
    if args.exp_name is not None:      t.experiment_name = args.exp_name
    if args.seed is not None:          d.seed = args.seed

    if args.epochs is not None:        t.epochs_phase2 = args.epochs
    if args.lr is not None:            t.lr_phase2 = args.lr
    if args.batch_size is not None:    t.batch_size = args.batch_size
    if args.warmup_steps is not None:  t.warmup_steps = args.warmup_steps
    if args.grad_clip is not None:     t.grad_clip = args.grad_clip
    if args.patience is not None:      t.early_stopping_patience = args.patience

    if args.embed_dim is not None:         m.embed_dim = args.embed_dim
    if args.ffn_expansion is not None:     m.ffn_expansion = args.ffn_expansion
    if args.ipsi_layers is not None:       m.num_ipsi_layers = args.ipsi_layers
    if args.bilateral_layers is not None:  m.num_bilateral_layers = args.bilateral_layers
    if args.token_grid is not None:
        tg = args.token_grid.strip().lower()
        m.token_grid = None if tg in ("none", "null", "") else \
            tuple(int(x) for x in tg.replace("x", ",").split(","))

    if args.unfreeze_backbone is not None:   t.unfreeze_backbone = args.unfreeze_backbone
    if args.unfreeze_from_stage is not None: t.unfreeze_from_stage = args.unfreeze_from_stage
    if args.backbone_lr_mult is not None:    t.backbone_lr_multiplier = args.backbone_lr_mult

    if args.supcon_weight is not None:       t.supcon_weight = args.supcon_weight
    if args.supcon_temperature is not None:  t.supcon_temperature = args.supcon_temperature
    if args.supcon_proj_dim is not None:     t.supcon_proj_dim = args.supcon_proj_dim
    if args.supcon_keep_no_finding:          t.supcon_exclude_no_finding = False
    if args.no_wandb:                       w.enabled = False

    return cfg


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = build_argparser().parse_args()
    cfg = apply_overrides(Config(), args)
    set_seed(cfg.data.seed)

    print("=" * 60)
    print(f"  exp_name        : {cfg.train.experiment_name}")
    print(f"  backbone init   : {'phase1' if cfg.train.load_phase1_backbone else 'ImageNet'}")
    _st = cfg.train.unfreeze_from_stage
    if _st is None and cfg.train.unfreeze_backbone:
        _st = 0
    print(f"  backbone        : " + ("frozen" if _st is None else
          ("fine-tune TOAN BO" if _st == 0 else f"fine-tune tu stage {_st + 1}")))
    print(f"  supcon          : " + ("TAT" if cfg.train.supcon_weight <= 0 else
          f"weight {cfg.train.supcon_weight:g}, T {cfg.train.supcon_temperature}"))
    print(f"  embed_dim       : {cfg.model.embed_dim} | ffn_exp {cfg.model.ffn_expansion} "
          f"| ipsi {cfg.model.num_ipsi_layers} | bila {cfg.model.num_bilateral_layers}")
    print(f"  token_grid      : {cfg.model.token_grid}")
    print(f"  lr / epochs     : {cfg.train.lr_phase2:.1e} / {cfg.train.epochs_phase2}")
    print(f"  batch / clip    : {cfg.train.batch_size} / {cfg.train.grad_clip}")
    print(f"  seed            : {cfg.data.seed}")
    print("=" * 60)

    train(cfg)
