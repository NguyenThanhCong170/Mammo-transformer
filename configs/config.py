from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    data_root: str = "images_cropped"          # root chứa ảnh đã crop
    csv_path: str = "finding_annotations.csv" 
    image_ext = ".png"
    image_size: Tuple[int, int] = (704, 1856)
    num_workers: int = 4

    # Split — VinDr-Mammo đã có sẵn cột 'split' (training/test)
    # Chỉ cần tách thêm val từ training
    val_ratio: float = 0.15          # Tỉ lệ val tách từ training set gốc
    seed: int = 42

    # Augmentation
    aug_level: int = 3                 # MedAugment level ∈ {1,2,3,4,5}

    num_classes = 4

    label_mapping = {
    "no finding": 0,       # Background / Ảnh bình thường
    "Mass": 1,
    "Suspicious Calcification": 2,
    "Asymmetry": 3,
    "Global Asymmetry": 3,
    "Focal Asymmetry": 3
    }


@dataclass
class ModelConfig:
    backbone_name: str = "swinv2_base_window12to16_192to256"  # timm model name
    backbone_pretrained: bool = True
    backbone_img_size: Tuple[int, int] = (1856, 704)        # Swin-V2-Base input size
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
    num_classes: int = 4


@dataclass
class TrainConfig:
    # Paths
    output_dir: str = "./outputs"
    experiment_name: str = "mammo_transformer_v1"

    # Projection_head
    hidden_dim = 2048
    out_dim = 128

    # Training      
    epochs_phase1: int = 50
    pretrained_phase1: bool = True
    
    epochs_phase2: int = 50               # Full fine-tune  # Freeze backbone
    pretrained_phase2: bool = False
    batch_size: int = 4                   # 4 ảnh/patient → memory nặng
    accumulate_grad_steps: int = 4        # Effective batch = 16

    # Optimizer
    lr_phase1: float = 1e-4
    lr_phase2: float = 3e-4               # Chỉ train attention + MLP
    backbone_lr_multiplier: float = 0.1   # Backbone LR = lr * 0.1
    weight_decay: float = 1e-2
    warmup_steps: int = 100

    # Loss
    focal_alpha: list[float] = [0.25, 0.25, 0.5, 0.25]           # Weight cho positive class
    focal_gamma: float = 2.0
    focal_reduction: str = "mean"

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
