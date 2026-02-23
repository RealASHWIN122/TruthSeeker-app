"""
dataset.py — Stream C: Data Loading & Augmentation Pipeline
============================================================
Provides:
  • JPEGCompression  — custom transform to simulate video codec noise
  • build_transforms — train / val transform factories
  • build_dataloaders — ImageFolder-based loaders with class-balanced sampling

Author : Principal AI Research Engineer
"""

import io
import random
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


# ─────────────────────────────────────────────────────────────────────────────
# Custom transforms
# ─────────────────────────────────────────────────────────────────────────────
class JPEGCompression:
    """
    Randomly re-encodes a PIL Image through JPEG at a uniform random quality
    drawn from [quality_low, quality_high].
    """

    def __init__(
        self,
        quality_low:  int   = 50,
        quality_high: int   = 90,
        p:            float = 0.5,
    ):
        if not (1 <= quality_low <= quality_high <= 95):
            raise ValueError(
                f"Require 1 ≤ quality_low ≤ quality_high ≤ 95, "
                f"got [{quality_low}, {quality_high}]"
            )
        self.quality_low  = quality_low
        self.quality_high = quality_high
        self.p            = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_low, self.quality_high)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"quality=[{self.quality_low},{self.quality_high}], p={self.p})"
        )


class RandomGaussianNoise:
    """
    Adds per-pixel Gaussian noise to a tensor after ToTensor conversion.
    """

    def __init__(self, std: float = 0.02, p: float = 0.3):
        self.std = std
        self.p   = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std}, p={self.p})"


# ─────────────────────────────────────────────────────────────────────────────
# Transform factories
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: ImageNet statistics removed. Frequency models require raw [0,1] input.

def build_transforms(
    image_size:       int   = 224,
    jpeg_quality_low: int   = 50,
    jpeg_quality_high:int   = 90,
    is_train:         bool  = True,
) -> T.Compose:
    """
    Build the transform pipeline for training or validation.
    
    CRITICAL CHANGE: Removed T.Normalize. 
    The FrequencyInputLayer expects raw [0,1] intensity values to compute
    the correct Fourier spectrum. Standardizing with ImageNet mean/std
    destroys the DC component and distorts frequency magnitude.
    """
    if is_train:
        return T.Compose([
            # ── PIL-stage augmentations ──────────────────────────────────────
            JPEGCompression(
                quality_low  = jpeg_quality_low,
                quality_high = jpeg_quality_high,
                p            = 0.5,
            ),
            T.Resize(int(image_size * 1.15)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=0.3,
            ),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            # ── Tensor-stage ─────────────────────────────────────────────────
            T.ToTensor(),
            # REMOVED: T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            RandomGaussianNoise(std=0.02, p=0.3),
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            # REMOVED: T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    train_dir:        str,
    val_dir:          str,
    batch_size:       int   = 32,
    num_workers:      int   = 4,
    image_size:       int   = 224,
    jpeg_quality_low: int   = 50,
    jpeg_quality_high:int   = 90,
    pin_memory:       bool  = True,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Build train and validation DataLoaders backed by ImageFolder.
    """
    train_dataset = ImageFolder(
        root      = train_dir,
        transform = build_transforms(
            image_size        = image_size,
            jpeg_quality_low  = jpeg_quality_low,
            jpeg_quality_high = jpeg_quality_high,
            is_train          = True,
        ),
    )
    val_dataset = ImageFolder(
        root      = val_dir,
        transform = build_transforms(
            image_size = image_size,
            is_train   = False,
        ),
    )

    # ── Weighted sampler for class-balanced training ──────────────────────────
    targets         = torch.tensor(train_dataset.targets)
    class_counts    = torch.bincount(targets)
    class_weights   = 1.0 / class_counts.float()
    sample_weights  = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_dataset),
        replacement = True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
        persistent_workers = num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        persistent_workers = num_workers > 0,
    )

    class_info = {
        "classes"       : train_dataset.classes,
        "class_to_idx"  : train_dataset.class_to_idx,
        "train_samples" : len(train_dataset),
        "val_samples"   : len(val_dataset),
        "class_counts"  : class_counts.tolist(),
    }

    return train_loader, val_loader, class_info