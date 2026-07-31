"""
Dataset cho VinDr-Mammo.

Cấu trúc CSV mong đợi:
  patient_id | image_id | laterality | view_position | image_path | breast_birads
  P001       | img001   | L          | MLO           | /path/.png | BI-RADS 2
  P001       | img002   | L          | CC            | /path/.png | BI-RADS 2
  P001       | img003   | R          | MLO           | /path/.png | BI-RADS 4
  P001       | img004   | R          | CC            | /path/.png | BI-RADS 4
"""

import os
import ast
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

import numpy as np
import pandas as pd
import pydicom
import torch
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torch.utils.data import DataLoader, Dataset

from data.augmentation import build_transforms
from configs.config import Config
cfg = Config()

# ──────────────────────────────────────────────
# View keys chuẩn hóa
# ──────────────────────────────────────────────
VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]

VIEW_INDEX = {key: i for i, key in enumerate(VIEW_KEYS)}  # {"L_MLO": 0, ...}

# ──────────────────────────────────────────────
# Main Dataset
# ──────────────────────────────────────────────
class MammoDataset(Dataset):
    """
    Mỗi sample = 1 bệnh nhân với 4 views.
    Trả về:
        images : Dict[str, Tensor]  — keys: L_MLO, L_CC, R_MLO, R_CC
        label  : Tensor (0 or 1)
        patient_id: str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_width: int = cfg.data.image_size[0],
        image_height: int = cfg.data.image_size[1],
        is_train: bool = True,
        aug_level: int = cfg.data.aug_level,   
    ):
        self.df = df
        self.image_width = image_width
        self.image_height = image_height
        self.transform = build_transforms(is_train, aug_level)
        self._build_patient_index()

    def _build_patient_index(self):
        """Group rows theo patient_id, tạo index."""
        self.patients: List[str] = self.df["patient_id"].unique().tolist()

        # Dict: patient_id → {view_key → row}
        self.patient_views: Dict[str, Dict[str, pd.Series]] = {}
        self.patient_labels: Dict[str, list] = {}

        for pid, group in self.df.groupby("patient_id"):
            view_map = {}
            label_vec = torch.zeros(cfg.data.num_classes, dtype=torch.float32)
            for _, row in group.iterrows():
                lat = str(row["laterality"]).strip().upper()    # L / R
                view = str(row["view_position"]).strip().upper() # MLO / CC
                key = f"{lat}_{view}"
                view_map[key] = row
                cls_indices  = ast.literal_eval(row["target"])
                for cls_idx in cls_indices:
                    label_vec[cls_idx] = 1.0
            self.patient_views[pid] = view_map

            if label_vec[1:].sum() > 0:
                label_vec[0] = 0.0
            self.patient_labels[pid] = label_vec

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int):
        pid = self.patients[idx]
        view_map = self.patient_views[pid]
        label = self.patient_labels[pid]

        images = {}
        for key in VIEW_KEYS:
            if key in view_map:
                img_path = view_map[key]["image_path"]
                img = self._load_image(img_path)
            else:
                # Thiếu view → tạo ảnh đen (padding)
                # Dùng transform tạo tensor rỗng đúng size
                print("Thiếu view")
                img = self._empty_image()

            images[key] = img

        return {
            "images": images,                          # Dict[str, Tensor(3,H,W)]
            "label": torch.tensor(label, dtype=torch.float32),
            "patient_id": pid,
        }

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")        # PIL Image, grayscale 1 kênh
        img_np = np.array(img, dtype=np.uint8)
        return self.transform(img_np)          # → Tensor(3, H, W)

    def _empty_image(self) -> torch.Tensor:
        """Tạo ảnh đen khi thiếu view (shape chuẩn để không crash collate)."""
        blank = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        return self.transform(blank)          # → Tensor(3, H, W)

    def get_labels(self) -> List[int]:
        """Trả về list labels theo thứ tự patients (dùng cho WeightedSampler)."""
        return [self.patient_labels[pid] for pid in self.patients]


# ──────────────────────────────────────────────
# Patient-level Split
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Patient-level Split — dùng cột split gốc của VinDr-Mammo
# ──────────────────────────────────────────────
def split_patients(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    VinDr-Mammo đã có sẵn cột 'split' = 'training' | 'test'.
    → Giữ nguyên test set gốc.
    → Từ training, tách thêm val theo patient_id (không theo image).
    """
    train_full_df = df[df["split"] == "training"].reset_index(drop=True)
    test_df       = df[df["split"] == "test"].reset_index(drop=True)

    # Tách val từ training theo patient_id (patient-level, tránh leakage)
    train_patients = train_full_df["patient_id"].unique().tolist()
    random.seed(seed)
    random.shuffle(train_patients)

    n_val = max(1,int(len(train_patients) * val_ratio))
    val_pids   = set(train_patients[:n_val])
    train_pids = set(train_patients[n_val:])

    train_df = train_full_df[train_full_df["patient_id"].isin(train_pids)].reset_index(drop=True)
    val_df   = train_full_df[train_full_df["patient_id"].isin(val_pids)].reset_index(drop=True)

    print(f"[Split] Dùng split gốc VinDr-Mammo:")
    print(f"  Train : {train_df['patient_id'].nunique()} patients")
    print(f"  Val   : {val_df['patient_id'].nunique()}   patients  (tách từ training, ratio={val_ratio})")
    print(f"  Test  : {test_df['patient_id'].nunique()}  patients  (gốc từ dataset)")
    print(f"Tổng: {train_df['patient_id'].nunique() + val_df['patient_id'].nunique() + test_df['patient_id'].nunique()}")
    return train_df, val_df, test_df


# ──────────────────────────────────────────────
# DataLoaders Factory
# ──────────────────────────────────────────────
def build_dataloaders(
    csv_path: str,
    image_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    val_ratio: float,        
    seed: int ,
    aug_level: int,      
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    df = pd.read_csv(csv_path)

    # Validate required columns
    required = {"patient_id", "laterality", "view_position", "image_path", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")

    train_df, val_df, test_df = split_patients(df, val_ratio = val_ratio, seed = seed)
    print(f"[Dataset] MedAugment level={aug_level}, PA={0.2*aug_level:.1f}")

    train_ds = MammoDataset(train_df, image_size[0], image_size[1], is_train=True,  aug_level=aug_level)
    val_ds   = MammoDataset(val_df,   image_size[0], image_size[1], is_train=False, aug_level=aug_level)
    test_ds  = MammoDataset(test_df,  image_size[0], image_size[1], is_train=False, aug_level=aug_level)


    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=mammo_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=mammo_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=mammo_collate_fn,
    )

    return train_loader, val_loader, test_loader


def mammo_collate_fn(batch):
    """
    Custom collate: stack images per view key.
    Output images: Dict[str, Tensor(B, 3, H, W)]
    """
    keys = VIEW_KEYS
    images = {k: torch.stack([item["images"][k] for item in batch]) for k in keys}
    labels = torch.stack([item["label"] for item in batch])
    patient_ids = [item["patient_id"] for item in batch]

    return {"images": images, "label": labels, "patient_id": patient_ids}



# trainloader trả về:
# batch = next(iter(trainloader))

# batch["images"] = {
#     "L_MLO": {B, 3 ,H,W},
#     "L_CC": {B, 3 ,H,W},
#     "R_MLO": {B, 3 ,H,W},
#     "R_CC": {B, 3 ,H,W},
# }

# batch["label"] = [B, (multi-hot-vector)] = [(0,0,0,1),(1,0,0,0), (0,1,1,0),...]
# batch["patient_ids"] = [B, (ids)]
