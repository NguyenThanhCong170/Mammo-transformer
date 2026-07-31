"""
Dataset cho VinDr-Mammo.

CSV mong đợi (sinh ra bởi data/prepare_csv.py):
  patient_id | image_id | image_path | laterality | view_position | target | split

QUY ƯỚC: image_size LUÔN là (H, W).
"""

import ast
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.augmentation import build_transforms
from configs.config import Config, resolve_path

cfg = Config()

VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]
VIEW_INDEX = {key: i for i, key in enumerate(VIEW_KEYS)}


# ──────────────────────────────────────────────
# Main Dataset
# ──────────────────────────────────────────────
class MammoDataset(Dataset):
    """
    Mỗi sample = 1 bệnh nhân với 4 view.

    Trả về:
        images     : Dict[str, Tensor(3, H, W)]
        label      : Tensor(num_classes,) multi-hot float
        patient_id : str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: Tuple[int, int] = cfg.data.image_size,   # (H, W)
        is_train: bool = True,
        aug_level: int = cfg.data.aug_level,
        num_classes: int = cfg.data.num_classes,
        verbose_missing: bool = False,
    ):
        self.df = df
        self.image_height, self.image_width = image_size
        self.num_classes = num_classes
        self.verbose_missing = verbose_missing
        self.transform = build_transforms(is_train, aug_level)
        self._missing_view_count = 0
        self._build_patient_index()

    def _build_patient_index(self):
        self.patients: List[str] = self.df["patient_id"].unique().tolist()
        self.patient_views: Dict[str, Dict[str, pd.Series]] = {}
        self.patient_labels: Dict[str, torch.Tensor] = {}

        for pid, group in self.df.groupby("patient_id"):
            view_map = {}
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

            for _, row in group.iterrows():
                lat = str(row["laterality"]).strip().upper()      # L / R
                view = str(row["view_position"]).strip().upper()  # MLO / CC
                view_map[f"{lat}_{view}"] = row

                target = row["target"]
                if isinstance(target, str):
                    target = ast.literal_eval(target)
                if isinstance(target, (int, np.integer)):
                    target = [int(target)]
                for cls_idx in target:
                    label_vec[int(cls_idx)] = 1.0

            # Nếu có bất kỳ finding nào → không còn là "no finding"
            if label_vec[1:].sum() > 0:
                label_vec[0] = 0.0

            self.patient_views[pid] = view_map
            self.patient_labels[pid] = label_vec

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int):
        pid = self.patients[idx]
        view_map = self.patient_views[pid]

        images = {}
        for key in VIEW_KEYS:
            if key in view_map:
                images[key] = self._load_image(view_map[key]["image_path"])
            else:
                self._missing_view_count += 1
                if self.verbose_missing:
                    print(f"[Dataset] Thiếu view {key} ở bệnh nhân {pid} → dùng ảnh đen.")
                images[key] = self._empty_image()

        return {
            "images": images,
            # label đã là Tensor → clone().detach(), KHÔNG dùng torch.tensor(tensor)
            "label": self.patient_labels[pid].clone().detach(),
            "patient_id": pid,
        }

    def _load_image(self, path: str) -> torch.Tensor:
        # image_path trong CSV là tương đối → resolve theo PROJECT_ROOT chứ không
        # theo cwd, để train chạy được dù bạn đứng ở thư mục nào.
        img = Image.open(resolve_path(path)).convert("L")
        # Ảnh đã resize sẵn, nhưng vẫn ép đúng size để chắc chắn không lệch shape
        if img.size != (self.image_width, self.image_height):   # PIL dùng (W, H)
            img = img.resize((self.image_width, self.image_height), Image.BILINEAR)
        img_np = np.array(img, dtype=np.uint8)
        return self.transform(img_np)          # → Tensor(3, H, W)

    def _empty_image(self) -> torch.Tensor:
        blank = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        return self.transform(blank)

    def get_labels(self) -> List[torch.Tensor]:
        return [self.patient_labels[pid] for pid in self.patients]

    def class_counts(self) -> np.ndarray:
        return torch.stack(self.get_labels()).sum(0).numpy()


# ──────────────────────────────────────────────
# Patient-level Split (dùng cột 'split' gốc của VinDr-Mammo)
# ──────────────────────────────────────────────
def split_patients(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_full_df = df[df["split"] == "training"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_patients = train_full_df["patient_id"].unique().tolist()
    rng = random.Random(seed)          # RNG riêng — không đụng global seed
    rng.shuffle(train_patients)

    n_val = max(1, int(len(train_patients) * val_ratio))
    val_pids = set(train_patients[:n_val])
    train_pids = set(train_patients[n_val:])

    train_df = train_full_df[train_full_df["patient_id"].isin(train_pids)].reset_index(drop=True)
    val_df = train_full_df[train_full_df["patient_id"].isin(val_pids)].reset_index(drop=True)

    print("[Split] Dùng split gốc VinDr-Mammo:")
    print(f"  Train : {train_df['patient_id'].nunique()} patients")
    print(f"  Val   : {val_df['patient_id'].nunique()} patients (tách từ training, ratio={val_ratio})")
    print(f"  Test  : {test_df['patient_id'].nunique()} patients (gốc)")
    return train_df, val_df, test_df


# ──────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────
def mammo_collate_fn(batch):
    images = {k: torch.stack([item["images"][k] for item in batch]) for k in VIEW_KEYS}
    labels = torch.stack([item["label"] for item in batch])
    patient_ids = [item["patient_id"] for item in batch]
    return {"images": images, "label": labels, "patient_id": patient_ids}


# ──────────────────────────────────────────────
# DataLoaders Factory
# ──────────────────────────────────────────────
def build_dataloaders(
    csv_path: str,
    image_size: Tuple[int, int],     # (H, W)
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    aug_level: int,
    num_classes: int = cfg.data.num_classes,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    df = pd.read_csv(csv_path)

    required = {"patient_id", "laterality", "view_position", "image_path", "target", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")

    train_df, val_df, test_df = split_patients(df, val_ratio=val_ratio, seed=seed)
    print(f"[Dataset] MedAugment level={aug_level}, PA={0.2 * aug_level:.1f}, image_size(H,W)={image_size}")

    train_ds = MammoDataset(train_df, image_size, is_train=True, aug_level=aug_level, num_classes=num_classes)
    val_ds = MammoDataset(val_df, image_size, is_train=False, aug_level=aug_level, num_classes=num_classes)
    test_ds = MammoDataset(test_df, image_size, is_train=False, aug_level=aug_level, num_classes=num_classes)

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mammo_collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True,            # ← BẮT BUỘC, đặc biệt với contrastive learning
        drop_last=True,          # giữ batch đều cho BatchNorm trong ProjectionHead
        **common,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common)

    print(f"[Dataset] batches — train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
    return train_loader, val_loader, test_loader


# batch = next(iter(train_loader))
#   batch["images"]["L_MLO"] → (B, 3, H, W)
#   batch["label"]           → (B, 4) multi-hot
#   batch["patient_id"]      → List[str] độ dài B
