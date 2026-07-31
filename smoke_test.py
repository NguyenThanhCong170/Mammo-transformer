"""
Smoke test — chạy TRƯỚC khi train để bắt lỗi shape/API trong ~1 phút,
thay vì đợi 20 phút load data rồi mới crash.

    python smoke_test.py                  # tự dùng GPU nếu có
    python smoke_test.py --device cpu     # ép chạy CPU
    python smoke_test.py --data           # test thêm dataloader (cần CSV + ảnh)
    python smoke_test.py --full-size      # dùng đúng image_size thật

Trên server có GPU cứ để mặc định — nhanh hơn CPU hàng chục lần.
"""

import argparse
import time
import traceback

import torch

from configs.config import Config
from models.mammo_transformer import MammoTransformer, SwinV2Backbone
from models.projection_head import ProjectionHead
from utils.losses import FocalLoss, MultiLabelMetricsCalculator
from utils.Supcon_loss import MultiViewNTXentLoss

VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]
OK, FAIL = "  ✔", "  ✘"


def _sync():
    """CUDA chạy bất đồng bộ — phải sync trước khi đo thời gian, nếu không
    con số đo được chỉ là thời gian xếp lệnh vào queue, không phải thời gian chạy."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _report(name, fn):
    print(f"\n▶ {name}")
    _sync()
    t0 = time.time()
    try:
        fn()
        _sync()
        print(f"{OK} PASS  ({time.time()-t0:.1f}s)")
        return True
    except Exception:
        _sync()
        print(f"{FAIL} FAIL  ({time.time()-t0:.1f}s)")
        traceback.print_exc()
        return False


def test_backbone_tokens(cfg, img_size, device):
    bb = SwinV2Backbone(cfg.model.backbone_name, pretrained=False,
                        img_size=img_size, token_grid=cfg.model.token_grid).to(device)
    x = torch.randn(2, 3, *img_size, device=device)
    pooled = bb(x)
    tokens = bb.forward_tokens(x)
    print(f"    pooled : {tuple(pooled.shape)}  (kỳ vọng (2, {bb.out_dim}))")
    print(f"    tokens : {tuple(tokens.shape)}  (kỳ vọng (2, N>1, {bb.out_dim}))")
    assert pooled.shape == (2, bb.out_dim)
    assert tokens.ndim == 3 and tokens.shape[0] == 2 and tokens.shape[2] == bb.out_dim
    # ĐÂY là điều kiện sống còn: N phải > 1, nếu không cross-attention vô nghĩa
    assert tokens.shape[1] > 1, "N == 1 → softmax của attention luôn = 1.0, attention vô dụng!"


def test_model_forward(cfg, img_size, device):
    model = MammoTransformer(
        backbone_name=cfg.model.backbone_name,
        backbone_pretrained=False,
        backbone_img_size=img_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        attn_dropout=cfg.model.attn_dropout,
        ffn_dropout=cfg.model.ffn_dropout,
        num_ipsi_layers=1,
        num_bilateral_layers=1,
        mlp_hidden_dim=cfg.model.mlp_hidden_dim,
        mlp_dropout=cfg.model.mlp_dropout,
        num_classes=cfg.data.num_classes,
        token_grid=cfg.model.token_grid,
    ).to(device)
    images = {k: torch.randn(2, 3, *img_size, device=device) for k in VIEW_KEYS}
    logits = model(images)
    print(f"    logits: {tuple(logits.shape)}  (kỳ vọng (2, {cfg.data.num_classes}))")
    assert logits.shape == (2, cfg.data.num_classes)

    # freeze → backbone phải ở eval mode và không còn param trainable
    model.freeze_backbone()
    model.train()
    assert not model.backbone.training, "Backbone bị freeze nhưng vẫn ở train mode!"
    counts = model.count_parameters()
    print(f"    params: total {counts['total']:,} | trainable {counts['trainable']:,}")
    assert counts["trainable"] < counts["total"]

    # backward chạy được
    targets = torch.randint(0, 2, (2, cfg.data.num_classes), device=device).float()
    loss = FocalLoss(cfg.train.focal_alpha, cfg.train.focal_gamma, "mean").to(device)(
        model(images), targets)
    loss.backward()
    print(f"    focal loss = {loss.item():.4f}, backward OK")

    if device.type == "cuda":
        print(f"    VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


def test_contrastive(cfg, device):
    bb_dim, B, V = 1024, 3, 4
    head = ProjectionHead(bb_dim, cfg.train.proj_hidden_dim,
                          cfg.train.proj_out_dim).to(device)
    feats = torch.randn(B * V, bb_dim, device=device)
    z = head(feats).reshape(B, V, -1)
    loss = MultiViewNTXentLoss(cfg.train.temperature)(z)
    print(f"    z: {tuple(z.shape)} | NT-Xent loss = {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss = nan/inf"
    loss.backward()

    # batch_size=1 phải báo lỗi rõ ràng thay vì trả nan
    try:
        MultiViewNTXentLoss()(torch.randn(1, 4, 128, device=device))
        raise AssertionError("Đáng lẽ phải raise ValueError khi batch_size=1")
    except ValueError:
        print("    batch_size=1 → raise ValueError đúng như mong đợi")


def test_metrics(cfg):
    calc = MultiLabelMetricsCalculator(
        num_classes=cfg.data.num_classes,
        class_names=["no_finding", "mass", "calcification", "asymmetry"])
    for _ in range(5):
        calc.update(torch.randn(8, cfg.data.num_classes),
                    torch.randint(0, 2, (8, cfg.data.num_classes)).float())
    thr = calc.find_optimal_thresholds()
    res = calc.compute(thr)
    print(f"    thresholds: {[round(t, 3) for t in thr]}")
    print(f"    macro_ap  : {res['macro']['macro_ap']}")
    assert len(thr) == cfg.data.num_classes
    assert "macro_ap" in res["macro"]


def test_dataloader(cfg, device):
    from data.dataset import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=cfg.data.csv_path,
        image_size=cfg.data.image_size,
        batch_size=2,
        num_workers=0,                       # 0 để traceback rõ ràng
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
        aug_level=cfg.data.aug_level,
        num_classes=cfg.data.num_classes,
        persistent_workers=False,
    )
    batch = next(iter(train_loader))
    h, w = cfg.data.image_size
    for k in VIEW_KEYS:
        shape = tuple(batch["images"][k].shape)
        print(f"    {k}: {shape}")
        assert shape[1:] == (3, h, w), f"Sai shape! kỳ vọng (3, {h}, {w})"
    print(f"    label: {tuple(batch['label'].shape)} | ví dụ: {batch['label'][0].tolist()}")
    assert batch["label"].shape[1] == cfg.data.num_classes

    # chuyển sang device đúng như lúc train
    images = {k: v.to(device) for k, v in batch["images"].items()}
    assert images["L_MLO"].device.type == device.type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true", help="test cả dataloader (cần labels.csv)")
    ap.add_argument("--full-size", action="store_true",
                    help="dùng đúng image_size thật (chậm trên CPU)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto = dùng GPU nếu có")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Yêu cầu --device cuda nhưng torch.cuda.is_available() = False")

    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"Device: cuda — {p.name} ({p.total_memory/1024**3:.1f} GB)")
        torch.cuda.reset_peak_memory_stats()
    else:
        print(f"Device: cpu — {torch.get_num_threads()} thread "
              f"(đặt OMP_NUM_THREADS nếu server bị oversubscribe)")

    cfg = Config()
    # 448x224: chọn cho NHANH trên CPU, tỉ lệ dọc gần giống ảnh mammo thật.
    #
    # Lưu ý: size này KHÔNG chia hết cho window_size ở mọi stage (112/16, 56/16...),
    # và kích thước thật (1856, 704) cũng vậy. Đó là chủ ý — nó ép đường
    # dynamic_mask + padding của timm phải chạy, đúng như lúc train thật.
    # Ràng buộc "native" của Swin window-16 là chia hết cho 512, không phải 32.
    img_size = cfg.model.backbone_img_size if args.full_size else (448, 224)
    print(f"Smoke test — img_size (H, W) = {img_size}, token_grid = {cfg.model.token_grid}")

    results = [
        _report("Backbone → token map", lambda: test_backbone_tokens(cfg, img_size, device)),
        _report("MammoTransformer forward + freeze + backward",
                lambda: test_model_forward(cfg, img_size, device)),
        _report("ProjectionHead + NT-Xent (phase 1)", lambda: test_contrastive(cfg, device)),
        _report("Metrics + threshold search", lambda: test_metrics(cfg)),
    ]
    if args.data:
        results.append(_report("DataLoader", lambda: test_dataloader(cfg, device)))

    print("\n" + "=" * 50)
    print(f"  KẾT QUẢ: {sum(results)}/{len(results)} PASS")
    print("=" * 50)
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
