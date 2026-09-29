from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class DataConfig:
    # ── PIPELINE: crop DICOM → raw_images_dir
    #              → prepare_csv.py  → csv_raw
    #              → resize_images.py → data_root + csv_path
    
    raw_images_dir: str = "images_cropped"      # image after crop dicom
    data_root: str = "images_928x352"           # image after resize

    raw_annotations_csv: str = "finding_annotations.csv"   
    csv_raw: str = "labels.csv"                 # csv of image after crop dicom
    csv_path: str = "labels_352x928.csv"        # csv of image after resize
    image_ext: str = ".png"

    image_size: Tuple[int, int] = (928, 352)

    num_workers: int = 4
    persistent_workers: bool = True

    val_ratio: float = 0.15
    seed: int = 42

    # Augmentation
    aug_level: int = 3                 # MedAugment level ∈ {1,2,3,4,5}

    num_classes: int = 3

    label_mapping: dict = field(default_factory=lambda: {
        "no finding": 0,
        "Mass": 1,
        "Suspicious Calcification": 2,
    })


@dataclass
class ModelConfig:
    backbone_name: str = "swinv2_base_window12to16_192to256"
    backbone_pretrained: bool = True
    backbone_img_size: Tuple[int, int] = (928, 352) 
    embed_dim: int = 1024                              # Swin-V2-Base num_features

    # Token grid 
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
    num_classes: int = 3


@dataclass
class TrainConfig:
    # Paths
    output_dir: str = "./outputs"
    experiment_name: str = "mammo_transformer"

    # ── Phase 1: contrastive pretrain backbone
    epochs_phase1: int = 50
    pretrained_phase1: bool = True          
    batch_size_phase1: int = 12            
    lr_phase1: float = 2.5e-5
    temperature: float = 0.1
    #   "patient"     : cả 4 view cùng bệnh nhân là positive (bản gốc)
    #   "ipsilateral" : chỉ cùng bên vú (L_MLO↔L_CC, R_MLO↔R_CC);
    #                   vú đối bên của chính bệnh nhân đó là HARD NEGATIVE

    contrastive_positives: str = "ipsilateral"
    proj_hidden_dim: int = 2048
    proj_out_dim: int = 128
    phase1_ckpt_name: str = "phase1_backbone.pt"

    # ── Phase 2: freeze backbone, train attention + MLP
    epochs_phase2: int = 100
    batch_size: int = 16
    accumulate_grad_steps: int = 1          # effective batch = 16
    lr_phase2: float = 3e-4
    # True = nạp backbone từ phase 1. Đặt False để train phase 2 từ ImageNet.
    # CLI: --phase1 / --no-phase1
    load_phase1_backbone: bool = True
    # Đường dẫn tường minh tới phase1_backbone.pt. ""
    # CLI: --phase1-ckpt
    phase1_ckpt_path: str = ""

    unfreeze_backbone: bool = False
    backbone_lr_multiplier: float = 0.05    # lr backbone = lr_phase2 * hệ số này

    unfreeze_from_stage: Optional[int] = None

    # ── SupCon phụ trợ, train CHUNG với FocalLoss
    # CLI: --supcon-weight
    supcon_weight: float = 0.0
    supcon_temperature: float = 0.1
    supcon_proj_dim: int = 128
    supcon_proj_hidden: int = 512
    # Loại no_finding khỏi định nghĩa positive. 
    supcon_exclude_no_finding: bool = True

    # Optimizer
    weight_decay: float = 1e-2
    warmup_steps: int = 500           
    grad_clip: float = 1.0

    # Loss
    focal_alpha: list = field(default_factory=lambda: [0.25, 0.80, 0.85])
    focal_gamma: float = 2.0
    focal_reduction: str = "mean"

    # Misc
    mixed_precision: bool = True
    prefer_bf16: bool = True
    save_top_k: int = 3
    early_stopping_patience: int = 8
    log_every_n_steps: int = 20


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "mammo-transformer"
    entity: str = None          
    run_name_phase1: str = None 
    run_name_phase2: str = None
    mode: str = "online"        # "online" | "offline" | "disabled"
    log_every_n_steps: int = 20
    watch_model: bool = False  


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        out = {}
        for section in ("data", "model", "train"):
            for k, v in asdict(getattr(self, section)).items():
                out[f"{section}.{k}"] = v
        return out


