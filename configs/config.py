from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union

# Thư mục gốc của project (nơi chứa configs/, data/, models/...).
# Dùng cái này thay vì cwd để `python -m data.prepare_csv` chạy được
# từ bất kỳ thư mục nào, không phụ thuộc bạn đang đứng ở đâu.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: Union[str, Path]) -> Path:
    """Đường dẫn tương đối → tính từ PROJECT_ROOT. Tuyệt đối → giữ nguyên."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class DataConfig:
    # ── PIPELINE: crop DICOM → raw_images_dir
    #              → prepare_csv.py  → csv_raw
    #              → resize_images.py → data_root + csv_path
    #
    # raw_images_dir : ảnh gốc sau crop, ĐẦU VÀO của prepare_csv.py
    # data_root      : ảnh dùng để TRAIN (sau resize). Hai cái có thể khác nhau!
    raw_images_dir: str = "images_cropped"      # prepare_csv.py quét thư mục này
    data_root: str = "images_928x352"           # training đọc thư mục này

    raw_annotations_csv: str = "finding_annotations.csv"   # file gốc VinDr
    csv_raw: str = "labels.csv"                 # prepare_csv.py ghi ra file này
    csv_path: str = "labels_352x928.csv"        # resize_images.py ghi ra, training đọc
    image_ext: str = ".png"

    # QUY ƯỚC TOÀN DỰ ÁN: image_size LUÔN là (H, W) — giống PyTorch/timm.
    # Gốc 1856x704 → resize 0.5x → 928x352 (giữ nguyên tỉ lệ 2.6364).
    image_size: Tuple[int, int] = (928, 352)

    # Trên Windows nên để 0 hoặc 2. num_workers>0 cần code bọc trong main().
    num_workers: int = 4
    persistent_workers: bool = True

    # Split — VinDr-Mammo đã có sẵn cột 'split' (training/test)
    val_ratio: float = 0.15
    seed: int = 42

    # Augmentation
    aug_level: int = 3                 # MedAugment level ∈ {1,2,3,4,5}

    num_classes: int = 4

    label_mapping: dict = field(default_factory=lambda: {
        "no finding": 0,
        "Mass": 1,
        "Suspicious Calcification": 2,
        "Asymmetry": 3,
        "Global Asymmetry": 3,
        "Focal Asymmetry": 3,
    })


@dataclass
class ModelConfig:
    backbone_name: str = "swinv2_base_window12to16_192to256"
    backbone_pretrained: bool = True
    backbone_img_size: Tuple[int, int] = (928, 352)    # (H, W) — phải khớp data.image_size
    embed_dim: int = 1024                              # Swin-V2-Base num_features

    # Token grid sau backbone.
    # Ở 928x352, Swin cho ra 29x11 = 319 token → pool xuống lưới nhỏ để
    # cross-attention vừa có nghĩa (nhiều hơn 1 token) vừa không nổ VRAM.
    # Ảnh nhẹ đi 4x nên giờ có thể thử (16, 6) = 96 token nếu VRAM còn dư.
    # Đặt None = giữ nguyên toàn bộ token.
    token_grid: Tuple[int, int] = (8, 4)               # (H_tok, W_tok) → 32 token/view

    # Cross-Attention
    num_heads: int = 8
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    ffn_expansion: int = 4
    num_ipsi_layers: int = 2
    num_bilateral_layers: int = 2

    # MLP Classifier
    mlp_hidden_dim: int = 512
    mlp_dropout: float = 0.3
    num_classes: int = 4


@dataclass
class TrainConfig:
    # Paths
    output_dir: str = "./outputs"
    experiment_name: str = "mammo_transformer_v1"

    # ── Phase 1: contrastive pretrain backbone
    epochs_phase1: int = 50
    pretrained_phase1: bool = True          # khởi tạo từ ImageNet
    # Contrastive: batch CÀNG LỚN CÀNG TỐT (negative = 4*(B-1) mỗi anchor).
    # Đo trên A40 dùng chung ~22 GB. Chạy lại find_batch_size.py nếu GPU trống hơn.
    batch_size_phase1: int = 8              # 8 bệnh nhân = 32 ảnh/forward, 28 negative
    lr_phase1: float = 1e-4
    temperature: float = 0.1
    proj_hidden_dim: int = 2048
    proj_out_dim: int = 128
    # File checkpoint backbone mà phase 1 ghi ra và phase 2 đọc vào
    phase1_ckpt_name: str = "phase1_backbone.pt"

    # ── Phase 2: freeze backbone, train attention + MLP
    epochs_phase2: int = 50
    # Supervised: batch lớn KHÔNG tốt hơn — nó làm GIẢM số bước cập nhật gradient.
    # Với ~3400 bệnh nhân train: eff.batch 16 → 212 step/epoch (10.600 step tổng).
    # Nếu để eff.batch 256 thì chỉ còn 13 step/epoch (650 step) — quá ít để hội tụ.
    # VRAM cho phép tới batch 64, nhưng ta chỉ dùng phần dư để BỎ accumulation
    # (nhanh hơn), chứ không tăng effective batch.
    batch_size: int = 16
    accumulate_grad_steps: int = 1          # effective batch = 16
    lr_phase2: float = 3e-4
    # True = nạp backbone từ phase 1. Đặt False để train phase 2 từ ImageNet.
    load_phase1_backbone: bool = True

    # Optimizer
    weight_decay: float = 1e-2
    warmup_steps: int = 100                 # tính theo STEP, không phải epoch
    grad_clip: float = 1.0

    # Loss
    focal_alpha: list = field(default_factory=lambda: [0.25, 0.80, 0.85, 0.88])
    focal_gamma: float = 2.0
    focal_reduction: str = "mean"

    # Misc
    mixed_precision: bool = True
    save_top_k: int = 3
    early_stopping_patience: int = 8
    log_every_n_steps: int = 20


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "mammo-transformer"
    entity: str = None          # None = dùng account mặc định của bạn
    run_name_phase1: str = None # None = wandb tự sinh tên
    run_name_phase2: str = None
    mode: str = "online"        # "online" | "offline" | "disabled"
    log_every_n_steps: int = 20
    watch_model: bool = False   # True = log gradient/param histogram (chậm hơn)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict:
        """Flatten config → dict phẳng để log lên wandb."""
        from dataclasses import asdict
        out = {}
        for section in ("data", "model", "train"):
            for k, v in asdict(getattr(self, section)).items():
                out[f"{section}.{k}"] = v
        return out
