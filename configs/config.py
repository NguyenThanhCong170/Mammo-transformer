from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    data_root: str = "images_png"          # root chứa ảnh đã crop
    csv_path: str = "labels.csv" # cột: patient_id, laterality, view, image_path, birads
    image_size: Tuple[int, int] = (256, 256)
    num_workers: int = 4

    # Split — VinDr-Mammo đã có sẵn cột 'split' (training/test)
    # Chỉ cần tách thêm val từ training
    val_ratio: float = 0.15          # Tỉ lệ val tách từ training set gốc
    seed: int = 42

    # Augmentation
    aug_level: int = 3                 # MedAugment level ∈ {1,2,3,4,5}
    # PA = 0.2*level: level=3 → PA=0.6, balanced cho 5000 patients

    # Class definition
    # Yes = BI-RADS 3,4, 5  |  No = BI-RADS 1, 2
    positive_birads: Tuple[int, ...] = (3,4, 5)


@dataclass
class ModelConfig:
    backbone_name: str = "swinv2_base_window12to16_192to256"  # timm model name
    backbone_pretrained: bool = True
    backbone_img_size: int = 256           # Swin-V2-Base input size
    embed_dim: int = 1024                  # Swin-V2-Base output dim

    # Cross-Attention
    num_heads: int = 8
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    num_ipsi_layers: int = 2               # Ipsilateral attention layers
    num_bilateral_layers: int = 2          # Bilateral attention layers

    # MLP Classifier
    mlp_hidden_dim: int = 512
    mlp_dropout: float = 0.3
    num_classes: int = 1                   # Binary → sigmoid


@dataclass
class TrainConfig:
    # Paths
    output_dir: str = "./outputs"
    experiment_name: str = "mammo_transformer_v1"

    # Training
    epochs_phase1: int = 10            # Freeze backbone
    epochs_phase2: int = 100               # Full fine-tune
    batch_size: int = 4                   # 4 ảnh/patient → memory nặng
    accumulate_grad_steps: int = 4        # Effective batch = 16

    # Optimizer
    lr_phase1: float = 3e-4               # Chỉ train attention + MLP
    lr_phase2: float = 5e-5               # Full fine-tune
    backbone_lr_multiplier: float = 0.1   # Backbone LR = lr * 0.1
    weight_decay: float = 1e-2
    warmup_steps: int = 100

    # Loss
    focal_alpha: float = 0.75             # Weight cho positive class
    focal_gamma: float = 2.0

    # Misc
    mixed_precision: bool = True
    save_top_k: int = 3
    early_stopping_patience: int = 8
    log_every_n_steps: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
