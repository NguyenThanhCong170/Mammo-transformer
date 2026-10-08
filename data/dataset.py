"""
Dataset cho VinDr-Mammo — bài toán NHỊ PHÂN theo BI-RADS.
 
CSV mong đợi (labels_224x224.csv):
  patient_id | image_id | image_path | laterality | view_position | birads | split
 
Quy tắc nhãn (mỗi ảnh):
  birads NaN          → 0 (normal)
  birads 3 / 4 / 5    → 1 (abnormal)
  birads khác (1, 2)  → 0 (normal)   # đổi qua ABNORMAL_BIRADS nếu muốn khác
Nhãn bệnh nhân = 1 nếu BẤT KỲ view nào abnormal.
 
QUY ƯỚC: image_size LUÔN là (H, W).
"""
 
import random
import re
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
ABNORMAL_BIRADS = {3, 4, 5}
 
 
# ──────────────────────────────────────────────
# BI-RADS → nhãn nhị phân
# ──────────────────────────────────────────────
def parse_birads(v) -> Optional[int]:
    """NaN → None; 3, 3.0, '3', 'BI-RADS 3' → 3."""
    if v is None or pd.isna(v):
        return None
    if isinstance(v, (int, np.integer, float, np.floating)):
        return int(v)
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None
 
 
def birads_to_label(v) -> int:
    b = parse_birads(v)
    return int(b in ABNORMAL_BIRADS) if b is not None else 0
 
 
# ──────────────────────────────────────────────
# Main Dataset
# ──────────────────────────────────────────────
class MammoDataset(Dataset):
    """
    Mỗi sample = 1 bệnh nhân với 4 view.
 
    Trả về:
        images      : Dict[str, Tensor(3, H, W)]
        label       : Tensor(1,)    — 1 = abnormal, 0 = normal (cấp bệnh nhân)
        view_labels : Tensor(V, 1)  — nhãn theo từng view (thứ tự VIEW_KEYS)
        view_mask   : Tensor(V,)    — 1 = view có thật, 0 = view thiếu (ảnh đen)
        patient_id  : str
    """
 
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: Tuple[int, int] = (224, 224),   # (H, W)
        is_train: bool = True,
        aug_level: int = cfg.data.aug_level,
        verbose_missing: bool = False,
    ):
        if "label" not in df.columns:
            raise ValueError("df cần cột 'label' (xem build_dataloaders).")
        self.df = df
        self.image_height, self.image_width = image_size
        self.verbose_missing = verbose_missing
        self.transform = build_transforms(is_train, aug_level)
        self._build_patient_index()
 
    def _build_patient_index(self):
        self.patients: List[str] = self.df["patient_id"].unique().tolist()
        self.patient_views: Dict[str, Dict[str, str]] = {}        # pid → {key: path}
        self.patient_view_labels: Dict[str, Dict[str, int]] = {}  # pid → {key: 0/1}
        self.patient_labels: Dict[str, int] = {}
 
        for pid, group in self.df.groupby("patient_id"):
            paths, labels = {}, {}
            for row in group.itertuples(index=False):
                lat = str(row.laterality).strip().upper()        # L / R
                view = str(row.view_position).strip().upper()    # MLO / CC
                key = f"{lat}_{view}"
                paths[key] = row.image_path
                labels[key] = int(row.label)
            self.patient_views[pid] = paths
            self.patient_view_labels[pid] = labels
            self.patient_labels[pid] = int(any(labels.values()))
 
    def __len__(self) -> int:
        return len(self.patients)
 
    def __getitem__(self, idx: int):
        pid = self.patients[idx]
        paths = self.patient_views[pid]
        vlabels = self.patient_view_labels[pid]
 
        images, view_labels, view_mask = {}, [], []
        for key in VIEW_KEYS:
            if key in paths:
                images[key] = self._load_image(paths[key])
                view_labels.append(float(vlabels[key]))
                view_mask.append(1.0)
            else:
                if self.verbose_missing:
                    print(f"[Dataset] Thiếu view {key} ở bệnh nhân {pid} → dùng ảnh đen.")
                images[key] = self._empty_image()
                view_labels.append(0.0)
                view_mask.append(0.0)
 
        return {
            "images": images,
            "label": torch.tensor([float(self.patient_labels[pid])], dtype=torch.float32),
            "view_labels": torch.tensor(view_labels, dtype=torch.float32).unsqueeze(-1),
            "view_mask": torch.tensor(view_mask, dtype=torch.float32),
            "patient_id": pid,
        }
 
    def _load_image(self, path: str) -> torch.Tensor:
        # image_path tương đối → resolve theo PROJECT_ROOT, không phụ thuộc cwd
        img = Image.open(resolve_path(path)).convert("L")
        if img.size != (self.image_width, self.image_height):   # PIL dùng (W, H)
            img = img.resize((self.image_width, self.image_height), Image.BILINEAR)
        return self.transform(np.array(img, dtype=np.uint8))    # → Tensor(3, H, W)
 
    def _empty_image(self) -> torch.Tensor:
        blank = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        return self.transform(blank)
 
    # ── tiện ích cho loss ──
    def class_counts(self) -> Tuple[int, int]:
        """(số normal, số abnormal) ở cấp bệnh nhân."""
        pos = sum(self.patient_labels.values())
        return len(self.patients) - pos, pos
 
    def pos_weight(self) -> torch.Tensor:
        """Dùng cho BCEWithLogitsLoss(pos_weight=...)."""
        neg, pos = self.class_counts()
        return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
 
 
# ──────────────────────────────────────────────
# Patient-level Split (split gốc VinDr + val tách từ training, stratified)
# ──────────────────────────────────────────────
def split_patients(
    df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_full_df = df[df["split"] == "training"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
 
    # nhãn cấp bệnh nhân → tách val theo từng lớp để giữ tỉ lệ abnormal
    pat_label = train_full_df.groupby("patient_id")["label"].max()
    rng = random.Random(seed)          # RNG riêng — không đụng global seed
    val_pids = set()
    for cls in (0, 1):
        pids = pat_label[pat_label == cls].index.tolist()
        rng.shuffle(pids)
        val_pids.update(pids[: max(1, int(len(pids) * val_ratio))])
 
    is_val = train_full_df["patient_id"].isin(val_pids)
    train_df = train_full_df[~is_val].reset_index(drop=True)
    val_df = train_full_df[is_val].reset_index(drop=True)
 
    def stat(d):
        p = d.groupby("patient_id")["label"].max()
        return f"{len(p)} patients, abnormal {int(p.sum())} ({p.mean():.1%})"
 
    print("[Split] Split gốc VinDr-Mammo (val tách từ training, stratified):")
    print(f"  Train : {stat(train_df)}")
    print(f"  Val   : {stat(val_df)}  (ratio={val_ratio})")
    print(f"  Test  : {stat(test_df)}")
    return train_df, val_df, test_df
 
 
# ──────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────
def mammo_collate_fn(batch):
    return {
        "images": {k: torch.stack([b["images"][k] for b in batch]) for k in VIEW_KEYS},
        "label": torch.stack([b["label"] for b in batch]),              # (B, 1)
        "view_labels": torch.stack([b["view_labels"] for b in batch]),  # (B, V, 1)
        "view_mask": torch.stack([b["view_mask"] for b in batch]),      # (B, V)
        "patient_id": [b["patient_id"] for b in batch],
    }
 
 
# ──────────────────────────────────────────────
# DataLoaders Factory
# ──────────────────────────────────────────────
def build_dataloaders(
    csv_path: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    aug_level: int,
    image_size: Tuple[int, int] = (224, 224),   # (H, W)
    birads_col: str = "finding_birads",
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
 
    df = pd.read_csv(csv_path)
 
    required = {"patient_id", "laterality", "view_position", "image_path", birads_col, "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")
 
    df["label"] = df[birads_col].map(birads_to_label).astype(int)
 
    train_df, val_df, test_df = split_patients(df, val_ratio=val_ratio, seed=seed)
    print(f"[Dataset] MedAugment level={aug_level}, PA={0.2 * aug_level:.1f}, image_size(H,W)={image_size}")
 
    train_ds = MammoDataset(train_df, image_size, is_train=True, aug_level=aug_level)
    val_ds = MammoDataset(val_df, image_size, is_train=False, aug_level=aug_level)
    test_ds = MammoDataset(test_df, image_size, is_train=False, aug_level=aug_level)
 
    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mammo_collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
    )
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common)
 
    print(f"[Dataset] batches — train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
    return train_loader, val_loader, test_loader