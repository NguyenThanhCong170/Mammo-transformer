"""
Dataset cho VinDr-Mammo (phiên bản dùng ảnh PNG).

Cấu trúc CSV mong đợi:
  patient_id | image_id | laterality | view_position | image_path | breast_birads
  P001       | img001   | L          | MLO           | /path/.png | BI-RADS 2
  ...
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # Thêm thư viện OpenCV để đọc file PNG
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.augmentation import build_transforms


# ──────────────────────────────────────────────
# View keys chuẩn hóa
# ──────────────────────────────────────────────
VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]

VIEW_INDEX = {key: i for i, key in enumerate(VIEW_KEYS)}  # {"L_MLO": 0, ...}


def _parse_birads(raw: str) -> int:
    """'BI-RADS 4' → 4, '4' → 4"""
    return int(str(raw).replace("BI-RADS", "").strip())


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
        image_size: int = 512,
        positive_birads: Tuple[int, ...] = (3,4, 5),
        is_train: bool = True,
        aug_level: int = 3,          # MedAugment level ∈ {1..5}
    ):
        self.df = df
        self.positive_birads = positive_birads
        self.image_size = image_size
        self.transform = build_transforms(image_size, is_train, aug_level)
        self._build_patient_index()

    def _build_patient_index(self):
        """Group rows theo patient_id, tạo index."""
        self.patients: List[str] = self.df["patient_id"].unique().tolist()

        # Dict: patient_id → {view_key → row}
        self.patient_views: Dict[str, Dict[str, pd.Series]] = {}
        self.patient_labels: Dict[str, int] = {}

        for pid, group in self.df.groupby("patient_id"):
            view_map = {}
            for _, row in group.iterrows():
                lat = str(row["laterality"]).strip().upper()    # L / R
                view = str(row["view_position"]).strip().upper() # MLO / CC
                key = f"{lat}_{view}"
                view_map[key] = row

            self.patient_views[pid] = view_map

            # Label: patient positive nếu BẤT KỲ bên nào BI-RADS 4/5
            birads_vals = [_parse_birads(row["breast_birads"]) for row in view_map.values()]
            label = int(any(b in self.positive_birads for b in birads_vals))
            self.patient_labels[pid] = label

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
                img = self._empty_image()

            images[key] = img

        return {
            "images": images,                          # Dict[str, Tensor(3,H,W)]
            "label": torch.tensor(label, dtype=torch.float32),
            "patient_id": pid,
        }

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Đọc file PNG -> numpy uint8 -> transform -> Tensor(3,H,W).
        """
        # Đọc ảnh PNG dưới dạng grayscale (1 channel)
        pixel = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if pixel is None:
            raise FileNotFoundError(f"Không thể đọc được ảnh tại: {path}")

        # Đảm bảo ảnh là uint8 (thường cv2 đã trả về uint8 cho ảnh 8-bit)
        pixel = pixel.astype(np.uint8)

        # Trả về qua transform (MedAugment hoặc Val) → Tensor(3, H, W)
        return self.transform(pixel)

    def _empty_image(self) -> torch.Tensor:
        """Tạo ảnh đen khi thiếu view (shape chuẩn để không crash collate)."""
        blank = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        return self.transform(blank)          # → Tensor(3, H, W)

    def get_labels(self) -> List[int]:
        """Trả về list labels theo thứ tự patients (dùng cho WeightedSampler)."""
        return [self.patient_labels[pid] for pid in self.patients]


# ──────────────────────────────────────────────
# Patient-level Split
# ──────────────────────────────────────────────
def split_patients(
    df: pd.DataFrame,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    VinDr-Mammo đã có sẵn cột 'split' = 'training' | 'test'.
    → Giữ nguyên test set gốc.
    → Từ training, tách thêm val theo patient_id (không theo image).
    """
    if "split" not in df.columns:
        raise ValueError(
            "CSV không có cột 'split'. "
            "VinDr-Mammo chuẩn phải có cột này với giá trị 'training'/'test'."
        )

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
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    val_ratio: float = 0.10,
    seed: int = 42,
    positive_birads: Tuple[int, ...] = (3,4, 5),
    aug_level: int = 3,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    df = pd.read_csv(csv_path)

    # Validate required columns
    required = {"patient_id", "laterality", "view_position", "image_path", "breast_birads"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")

    train_df, val_df, test_df = split_patients(df, val_ratio = val_ratio, seed = seed)
    print(f"[Dataset] MedAugment level={aug_level}, PA={0.2*aug_level:.1f}")

    train_ds = MammoDataset(train_df, image_size, positive_birads, is_train=True,  aug_level=aug_level)
    val_ds   = MammoDataset(val_df,   image_size, positive_birads, is_train=False, aug_level=aug_level)
    test_ds  = MammoDataset(test_df,  image_size, positive_birads, is_train=False, aug_level=aug_level)

    # Weighted sampler để handle class imbalance
    train_labels = train_ds.get_labels()
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    weights = [1.0 / n_neg if l == 0 else 1.0 / n_pos for l in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
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