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
    data_root: str = "images_224x224"           # image after resize

    raw_annotations_csv: str = "finding_annotations.csv"   
    csv_raw: str = "labels.csv"                 # csv of image after crop dicom
    csv_path: str = "labels_224x224.csv"        # csv of image after resize
    image_ext: str = ".png"

    image_size: Tuple[int, int] = (224, 224)

    num_workers: int = 4
    persistent_workers: bool = True

    val_ratio: float = 0.15
    seed: int = 42

    # Augmentation
    aug_level: int = 3                 # MedAugment level ∈ {1,2,3,4,5}

    num_classes = 2

# @dataclass
# class ModelConfig:
   
# @dataclass
# class TrainConfig:
   

# @dataclass



@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
   
