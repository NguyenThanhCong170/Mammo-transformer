"""
Dò batch_size lớn nhất chạy được cho phase 1 / phase 2 trên GPU của bạn.

    python find_batch_size.py                # phase 1 (backbone + projection head)
    python find_batch_size.py --phase 2      # phase 2 (full model, backbone frozen)

Mô phỏng đúng 1 bước train thật: forward + backward + optimizer.step()
(bước optimizer rất quan trọng — AdamW cấp phát thêm 2 buffer bằng size model
ở lần step đầu tiên, nhiều người OOM đúng ở chỗ này sau khi forward đã qua.)
"""

import argparse
import gc

import torch

from configs.config import Config
from models.mammo_transformer import MammoTransformer, SwinV2Backbone
from models.projection_head import ProjectionHead
from utils.losses import FocalLoss
from utils.Supcon_loss import MultiViewNTXentLoss

VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]


def _clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def try_phase1(cfg, batch_size, device, amp_dtype):
    backbone = SwinV2Backbone(
        cfg.model.backbone_name, pretrained=False,
        img_size=cfg.model.backbone_img_size,
        token_grid=cfg.model.token_grid, grad_checkpointing=True,
    ).to(device)
    backbone.unfreeze()
    head = ProjectionHead(backbone.out_dim, cfg.train.proj_hidden_dim,
                          cfg.train.proj_out_dim).to(device)
    opt = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=1e-4)
    scaler = torch.amp.GradScaler(device="cuda")
    crit = MultiViewNTXentLoss(cfg.train.temperature)

    n_img = batch_size * len(VIEW_KEYS)
    for _ in range(2):                        # 2 step: step 2 mới thấy peak thật của AdamW
        x = torch.randn(n_img, 3, *cfg.model.backbone_img_size, device=device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            z = head(backbone(x)).reshape(batch_size, len(VIEW_KEYS), -1)
            loss = crit(z)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    del backbone, head, opt, x, z, loss


def try_phase2(cfg, batch_size, device, amp_dtype):
    model = MammoTransformer(
        backbone_name=cfg.model.backbone_name,
        backbone_pretrained=False,
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
    model.freeze_backbone()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4)
    scaler = torch.amp.GradScaler(device="cuda")
    crit = FocalLoss(cfg.train.focal_alpha, cfg.train.focal_gamma, "mean").to(device)

    for _ in range(2):
        images = {k: torch.randn(batch_size, 3, *cfg.model.backbone_img_size, device=device)
                  for k in VIEW_KEYS}
        labels = torch.randint(0, 2, (batch_size, cfg.model.num_classes),
                               device=device).float()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            loss = crit(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    del model, opt, images, labels, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2])
    ap.add_argument("--max", type=int, default=128, help="trần trên khi dò")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Cần GPU để chạy script này.")

    cfg = Config()
    device = torch.device("cuda")
    amp_dtype = torch.float16
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU        : {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")
    print(f"Phase      : {args.phase}")
    print(f"image_size : {cfg.model.backbone_img_size} (H, W)")
    print(f"backbone   : {cfg.model.backbone_name}\n")

    runner = try_phase1 if args.phase == 1 else try_phase2

    # Phase 1 dùng contrastive loss → cần ít nhất 2 bệnh nhân mới có negative,
    # MultiViewNTXentLoss raise ValueError nếu batch_size < 2.
    min_bs = 2 if args.phase == 1 else 1
    peaks = {}

    def probe(bs: int) -> bool:
        """True nếu chạy được. Chỉ nuốt OOM — lỗi khác phải nổi lên."""
        _clear()
        try:
            runner(cfg, bs, device, amp_dtype)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            peaks[bs] = (peak, reserved)
            print(f"  batch_size={bs:>3} ({bs*4:>3} ảnh) → peak {peak:5.2f} GB | "
                  f"reserved {reserved:5.2f} GB  ✔")
            return True
        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={bs:>3} ({bs*4:>3} ảnh) → OOM  ✘")
            return False
        except RuntimeError as e:                      # torch cũ ném RuntimeError
            if "out of memory" in str(e).lower():
                print(f"  batch_size={bs:>3} ({bs*4:>3} ảnh) → OOM  ✘")
                return False
            raise
        finally:
            _clear()

    # Giai đoạn 1: nhân đôi cho tới khi OOM (nhanh hơn nhiều so với dò tuyến tính)
    print(f"Dò từ batch_size={min_bs}, nhân đôi cho tới khi OOM...\n")
    last_ok, first_oom, bs = 0, None, min_bs
    while bs <= args.max:
        if probe(bs):
            last_ok = bs
            bs *= 2
        else:
            first_oom = bs
            break

    # Giai đoạn 2: nhị phân giữa (chạy được) và (OOM) để tìm ranh giới chính xác
    if last_ok and first_oom and first_oom - last_ok > 1:
        print(f"\nTinh chỉnh trong khoảng ({last_ok}, {first_oom})...\n")
        lo, hi = last_ok, first_oom
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if probe(mid):
                lo = mid
            else:
                hi = mid
        last_ok = lo

    best = last_ok
    results = [(b, *peaks[b]) for b in sorted(peaks)]

    print("\n" + "=" * 60)
    key = "batch_size_phase1" if args.phase == 1 else "batch_size"
    if best == 0:
        print(f"  Ngay cả batch_size={min_bs} cũng OOM.")
        print("  → Giảm image_size, hoặc đổi sang swinv2_tiny/small.")
    else:
        # Chừa ~15% headroom: đo bằng tensor random, dữ liệu thật + fragmentation
        # sau vài trăm step sẽ tốn hơn con số đo được ở đây.
        safe = max(min_bs, int(best * 0.85))
        peak_at_best = peaks[best][0]
        print(f"  Lớn nhất chạy được : {best}  ({best*4} ảnh, peak {peak_at_best:.2f} GB)")
        print(f"  KHUYẾN NGHỊ        : {safe}   ← chừa 15% headroom")
        if best >= args.max:
            print(f"  ⚠ Chạm trần --max={args.max}, thực tế còn cao hơn. Thử --max {args.max*2}")
        print(f"\n  cfg.train.{key} = {safe}")
        if args.phase == 1:
            print(f"  → {safe} bệnh nhân = {safe*4} ảnh/step, "
                  f"{(safe-1)*4} negative cho mỗi anchor")
    print("=" * 60)
    print("\nMẹo giảm fragmentation (đặt TRƯỚC khi chạy train):")
    print("  Windows PowerShell : $env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'")
    print("  Linux/macOS        : export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


if __name__ == "__main__":
    main()
