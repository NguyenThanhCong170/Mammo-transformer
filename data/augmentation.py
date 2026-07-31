"""
MedAugment — improved sampling strategy theo paper.

Key properties:
  - Ap (pixel-space): brightness, contrast, posterize, sharpen, blur, noise
  - As (spatial-space): rotate, hflip, vflip, scale, shear_x/y, translate_x/y
  - Mỗi branch: |M| ∈ {2, 3} = đúng 1 op từ Ap + 1..2 op từ As, rồi shuffle
  - PA = 0.2 * level, intensity ∝ level

LƯU Ý: ảnh mammo là GRAYSCALE. Các op ở đây được chọn/ép sao cho chạy được
trên input (H, W, 1) — ColorJitter của albumentations cần 3 kênh nên đã thay
bằng RandomBrightnessContrast.

Tương thích cả albumentations 1.3.x (cval/mode/var_limit) lẫn ≥1.4.21
(fill/border_mode/std_range) thông qua helper _build().
"""

import math
import random

import albumentations as A
import cv2
import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _make_odd(num: float) -> int:
    num = math.ceil(num)
    return num + 1 if num % 2 == 0 else num


def _build(cls, *kwarg_variants):
    """
    Thử khởi tạo `cls` lần lượt với từng bộ kwargs cho tới khi thành công.
    Dùng để chống vỡ khi albumentations đổi tên tham số giữa các version.
    """
    last_err = None
    for kwargs in kwarg_variants:
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as e:
            last_err = e
    raise RuntimeError(f"Không khởi tạo được {cls.__name__}: {last_err}")


# ─────────────────────────────────────────────────────────────
# Operation Pools
# ─────────────────────────────────────────────────────────────
def _build_pixel_pool(level: int) -> list:
    """Ap — pixel-space ops."""
    PA = min(1.0, 0.2 * level)

    gauss_noise = _build(
        A.GaussNoise,
        dict(std_range=(0.01 * level, 0.04 * level), mean_range=(0.0, 0.0),
             per_channel=False, p=PA),                                  # albumentations ≥1.4.21
        dict(var_limit=(level, 8 * level), mean=0, per_channel=False, p=PA),  # 1.3.x / 1.4.x cũ
        dict(p=PA),
    )

    blur = _build(
        A.GaussianBlur,
        dict(blur_limit=(3, _make_odd(3 + 0.6 * level)), p=PA),
        dict(p=PA),
    )

    return [
        # Brightness (grayscale-safe)
        A.RandomBrightnessContrast(
            brightness_limit=(-0.06 * level, 0.06 * level),
            contrast_limit=0.0, p=PA,
        ),
        # Contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.0,
            contrast_limit=(-0.06 * level, 0.06 * level), p=PA,
        ),
        # Posterize — giảm bit depth
        A.Posterize(num_bits=max(1, math.floor(8 - 0.8 * level)), p=PA),
        # Sharpen
        A.Sharpen(alpha=(0.02 * level, 0.08 * level), lightness=(1.0, 1.0), p=PA),
        blur,
        gauss_noise,
    ]


def _build_spatial_pool(level: int) -> list:
    """As — spatial-space ops."""
    PA = min(1.0, 0.2 * level)

    rotate = _build(
        A.Rotate,
        dict(limit=4 * level, interpolation=cv2.INTER_LINEAR,
             border_mode=cv2.BORDER_CONSTANT, fill=0,
             rotate_method="largest_box", crop_border=False, p=PA),
        dict(limit=4 * level, interpolation=cv2.INTER_LINEAR,
             border_mode=cv2.BORDER_CONSTANT, value=0,
             rotate_method="largest_box", crop_border=False, p=PA),
        dict(limit=4 * level, p=PA),
    )

    def affine(**core):
        return _build(
            A.Affine,
            dict(**core, fill=0, border_mode=cv2.BORDER_CONSTANT, p=PA),
            dict(**core, cval=0, mode=cv2.BORDER_CONSTANT, p=PA),
            dict(**core, p=PA),
        )

    return [
        rotate,
        A.HorizontalFlip(p=PA),
        A.VerticalFlip(p=PA),
        affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), keep_ratio=True),
        affine(shear={"x": (-2 * level, 2 * level), "y": (0, 0)}),
        affine(shear={"x": (0, 0), "y": (-2 * level, 2 * level)}),
        affine(translate_percent={"x": (-0.02 * level, 0.02 * level), "y": (0, 0)}),
        affine(translate_percent={"x": (0, 0), "y": (-0.02 * level, 0.02 * level)}),
    ]


# ─────────────────────────────────────────────────────────────
# Core: MedAugment Transform
# ─────────────────────────────────────────────────────────────
class MedAugmentTransform:
    """
    Algorithm 1 (paper):
      1. Sample |M| ∈ {2, 3}
      2. Sample 1 op từ Ap
      3. Sample |M|-1 op từ As (không lặp)
      4. M = shuffle([op_pixel] + ops_spatial)
      5. Apply M lần lượt
    """

    def __init__(self, level: int):
        assert 1 <= level <= 5, "level phải trong khoảng [1, 5]"
        self.level = level
        self.pixel_pool = _build_pixel_pool(level)
        self.spatial_pool = _build_spatial_pool(level)

    def _sample_branch(self) -> A.Compose:
        m_size = random.choice([2, 3])
        n_spatial = m_size - 1

        op_pixel = random.choice(self.pixel_pool)
        ops_spatial = random.sample(self.spatial_pool, n_spatial)

        ops = [op_pixel] + ops_spatial
        random.shuffle(ops)
        return A.Compose(ops)

    def __call__(self, img_np: np.ndarray) -> torch.Tensor:
        if img_np.ndim == 3:
            img_np = img_np[:, :, 0]

        img_hwc = img_np[:, :, np.newaxis]              # (H, W, 1)
        img_aug = self._sample_branch()(image=img_hwc)["image"]

        if img_aug.ndim == 2:
            img_aug = img_aug[:, :, np.newaxis]
        img_aug = np.ascontiguousarray(img_aug)

        # Grayscale → 3 kênh (backbone pretrained ImageNet cần 3 channel)
        img_3c = np.repeat(img_aug, 3, axis=2)          # (H, W, 3)

        tensor = torch.from_numpy(img_3c).permute(2, 0, 1).float() / 255.0
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD


# ─────────────────────────────────────────────────────────────
# Validation / Inference Transform
# ─────────────────────────────────────────────────────────────
class ValTransform:
    def __call__(self, img_np: np.ndarray) -> torch.Tensor:
        if img_np.ndim == 3:
            img_np = img_np[:, :, 0]
        img_3c = np.stack([img_np, img_np, img_np], axis=0)   # (3, H, W)
        tensor = torch.from_numpy(img_3c).float() / 255.0
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────
def build_transforms(is_train: bool, aug_level: int):
    """
    is_train  : True → MedAugment, False → chỉ normalize
    aug_level : 2 = nhẹ | 3 = balanced (khuyến nghị) | 4 = mạnh, cần theo dõi overfit
    """
    return MedAugmentTransform(level=aug_level) if is_train else ValTransform()
