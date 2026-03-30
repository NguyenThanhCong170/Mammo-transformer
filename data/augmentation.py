"""
MedAugment — Improved sampling strategy theo paper.

Key properties:
  - Ap (pixel-space): brightness, contrast, posterize, sharpen, blur, noise
  - As (spatial-space): rotate, hflip, vflip, scale, shear_x, shear_y, translate_x, translate_y
  - Mỗi branch: |M| ∈ {2, 3}, gồm đúng 1 op từ Ap + 1 hoặc 2 op từ As
  - Shuffle thứ tự ops trong mỗi branch → tăng diversity
  - Intensity: Uniform(0, MA) với MA = f(level), PA = 0.2 * level
  - Tránh over-augmentation bằng cách limit số ops và intensity
"""

import math
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as T

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _make_odd(num: float) -> int:
    """Làm tròn lên và đảm bảo số lẻ (yêu cầu của kernel size)."""
    num = math.ceil(num)
    return num + 1 if num % 2 == 0 else num

# ─────────────────────────────────────────────────────────────
# Operation Pools
# ─────────────────────────────────────────────────────────────
def _build_pixel_pool(level: int) -> list:
    """
    Ap — Pixel-space operations.
    level ∈ {1,2,3,4,5} → PA = 0.2*level, intensity ∝ level.
    Mỗi op được wrap thành callable: op(image) → image (numpy HWC uint8).
    """
    PA = 0.2 * level   # Application probability

    pool = [
        # Brightness
        A.ColorJitter(
            brightness=(0.02 * level, 0.06 * level),
            contrast=0, saturation=0, hue=0, p=PA
        ),
        # Contrast
        A.ColorJitter(
            brightness=0,
            contrast=(0.02 * level, 0.06 * level),
            saturation=0, hue=0, p=PA
        ),
        # Posterize — giảm bit depth
        A.Posterize(
            num_bits=max(1, math.floor(8 - 0.8 * level)),
            p=PA
        ),
        # Sharpen
        A.Sharpen(
            alpha=(0.02 * level, 0.08 * level),
            lightness=(1.0, 1.0),
            p=PA
        ),
        # Gaussian Blur
        A.GaussianBlur(
            blur_limit=(3, _make_odd(3 + 0.6 * level)),
            p=PA
        ),
        # Gaussian Noise
        A.GaussNoise(
            var_limit=(level, 8 * level),
            mean=0,
            per_channel=False,
            p=PA
        ),
    ]
    return pool


def _build_spatial_pool(level: int) -> list:
    """
    As — Spatial-space operations.
    """
    PA = 0.2 * level

    pool = [
        # Rotation
        A.Rotate(
            limit=4 * level,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            rotate_method='largest_box',
            crop_border=False,
            p=PA
        ),
        # Horizontal Flip
        A.HorizontalFlip(p=PA),
        # Vertical Flip
        A.VerticalFlip(p=PA),
        # Scale (keep ratio)
        A.Affine(
            scale=(1 - 0.04 * level, 1 + 0.04 * level),
            keep_ratio=True,
            cval=0, mode=cv2.BORDER_CONSTANT,
            p=PA
        ),
        # Shear X
        A.Affine(
            shear={'x': (-2 * level, 2 * level), 'y': (0, 0)},
            cval=0, mode=cv2.BORDER_CONSTANT,
            p=PA
        ),
        # Shear Y
        A.Affine(
            shear={'x': (0, 0), 'y': (-2 * level, 2 * level)},
            cval=0, mode=cv2.BORDER_CONSTANT,
            p=PA
        ),
        # Translate X
        A.Affine(
            translate_percent={'x': (-0.02 * level, 0.02 * level), 'y': (0, 0)},
            cval=0, mode=cv2.BORDER_CONSTANT,
            p=PA
        ),
        # Translate Y
        A.Affine(
            translate_percent={'x': (0, 0), 'y': (-0.02 * level, 0.02 * level)},
            cval=0, mode=cv2.BORDER_CONSTANT,
            p=PA
        ),
    ]
    return pool


# ─────────────────────────────────────────────────────────────
# Core: MedAugment Transform
# ─────────────────────────────────────────────────────────────
class MedAugmentTransform:
    """
    Áp dụng MedAugment improved sampling strategy.

    Algorithm (theo paper):
      1. Sample |M| ∈ {2, 3}  (2 = 1 pixel + 1 spatial, 3 = 1 pixel + 2 spatial)
      2. Sample 1 op từ Ap (pixel pool)
      3. Sample |M|-1 ops từ As (spatial pool), không lặp lại
      4. M = [op_pixel] + [op_spatial(s)]
      5. M = shuffle(M)
      6. Apply M lần lượt lên ảnh

    Args:
        level : int ∈ {1,2,3,4,5} — cường độ augmentation
                  1 = nhẹ nhất, 5 = mạnh nhất
                  Khuyến nghị: 2 hoặc 3 cho mammography
        image_size : int — resize trước khi augment
    """

    def __init__(self, level: int = 3, image_size: int = 256):
        assert 1 <= level <= 5, "level phải trong khoảng [1, 5]"
        self.level      = level
        self.image_size = image_size
        self.pixel_pool   = _build_pixel_pool(level)
        self.spatial_pool = _build_spatial_pool(level)

    def _sample_branch(self) -> A.Compose:
        """
        Sample 1 branch theo Algorithm 1.
        |M| ∈ {2, 3}: 
          - |M|=2 → 1 pixel op + 1 spatial op
          - |M|=3 → 1 pixel op + 2 spatial ops
        """
        m_size = random.choice([2, 3])          # |M| ∈ {2, 3}
        n_spatial = m_size - 1                   # 1 hoặc 2 spatial ops

        op_pixel   = random.choice(self.pixel_pool)
        ops_spatial = random.sample(self.spatial_pool, n_spatial)

        # Gộp lại rồi shuffle → tăng diversity
        ops = [op_pixel] + ops_spatial
        random.shuffle(ops)

        return A.Compose(ops)

    def __call__(self, img_np: np.ndarray) -> torch.Tensor:
        """
        Nhận numpy array đã được load từ DICOM (uint8, shape HW hoặc HWC).
        → Tensor (3, H, W) normalized

        Pipeline:
          numpy(H,W) uint8
            → resize
            → MedAugment branch
            → replicate sang 3 channel (grayscale → RGB)
            → normalize (ImageNet stats)
            → Tensor(3, H, W)
        """
        # Đảm bảo HW (grayscale 2D)
        if img_np.ndim == 3:
            img_np = img_np[:, :, 0]

        # Resize
        img_np = cv2.resize(
            img_np,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )  # shape: (H, W), dtype: uint8

        # Albumentations yêu cầu HWC
        img_hwc = img_np[:, :, np.newaxis]   # (H, W, 1)

        # Apply 1 branch MedAugment
        branch = self._sample_branch()
        result = branch(image=img_hwc)
        img_aug = result["image"]             # (H, W, 1) uint8

        # Grayscale → 3 channel bằng cách replicate
        # (backbone pretrained trên ImageNet cần 3 channel)
        img_3c = np.repeat(img_aug, 3, axis=2)  # (H, W, 3) uint8

        # numpy HWC uint8 → Tensor CHW float, rồi normalize
        tensor = torch.from_numpy(img_3c).permute(2, 0, 1).float() / 255.0
        tensor = self._normalize(tensor)
        return tensor

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        """ImageNet mean/std normalize — áp dụng per-channel."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std


# ─────────────────────────────────────────────────────────────
# Validation / Inference Transform (không augment)
# ─────────────────────────────────────────────────────────────
class ValTransform:
    def __init__(self, image_size: int = 256):
        self.image_size = image_size

    def __call__(self, img_np: np.ndarray) -> torch.Tensor:
        """
        img_np: numpy array uint8 (H, W) từ DICOM
        → Tensor (3, H, W) normalized
        """
        if img_np.ndim == 3:
            img_np = img_np[:, :, 0]

        img_np = cv2.resize(
            img_np,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # Grayscale → 3 channel
        img_3c = np.stack([img_np, img_np, img_np], axis=0)  # (3, H, W)
        tensor = torch.from_numpy(img_3c).float() / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std


# ─────────────────────────────────────────────────────────────
# Factory (drop-in replacement cho build_transforms cũ)
# ─────────────────────────────────────────────────────────────
def build_transforms(image_size: int, is_train: bool, aug_level: int = 3):
    """
    Factory function — tương thích với dataset.py hiện tại.

    Args:
        image_size : kích thước ảnh đầu vào backbone
        is_train   : True → MedAugment, False → chỉ resize + normalize
        aug_level  : cường độ augmentation ∈ {1..5}
                     - level=2 : nhẹ, phù hợp dataset nhỏ / ảnh rất nhạy cảm
                     - level=3 : balanced (khuyến nghị cho VinDr-Mammo 5000 BN)
                     - level=4 : mạnh hơn, cần monitor overfitting
    """
    if is_train:
        return MedAugmentTransform(level=aug_level, image_size=image_size)
    else:
        return ValTransform(image_size=image_size)
