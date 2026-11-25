"""Forensic perturbation battery for evaluating neural generalization and robustness.

Implements realistic transmission, compression, and social-platform perturbations:
- Multi-grade JPEG compression (Q=95, 75, 50, 30)
- Social-media-style recompression (downsampling + 4:2:0 subsampling + aggressive compression)
- Downsampling and resolution scaling
- Central and boundary cropping
- Image contrast/brightness enhancement
"""

from enum import Enum
import io
from typing import Callable, Dict, Tuple

import numpy as np
from PIL import Image, ImageEnhance


class PerturbationType(str, Enum):
    """Categorical taxonomy of generalization stress perturbations."""

    BASELINE = "baseline"
    JPEG_COMPRESSION = "jpeg_compression"
    SOCIAL_RECOMPRESSION = "social_recompression"
    RESIZING = "resizing"
    CROPPING = "cropping"
    ENHANCEMENT = "enhancement"


def apply_jpeg_compression(image: Image.Image, quality: int = 75) -> Image.Image:
    """Apply in-memory JPEG compression at specified quality factor in [1, 100]."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_social_media_recompression(image: Image.Image) -> Image.Image:
    """Simulate multi-stage social platform recompression (resize + heavy JPEG subsampling)."""
    orig_w, orig_h = image.size
    # 1. Downscale by 25% if image is large
    new_w = max(64, int(orig_w * 0.75))
    new_h = max(64, int(orig_h * 0.75))
    resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    # 2. Aggressive JPEG recompression at Q=60
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=60, subsampling=2)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    # 3. Restore to original dimensions
    return recompressed.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)


def apply_resizing(image: Image.Image, scale_factor: float = 0.5) -> Image.Image:
    """Downscale by scale_factor and then upscale back to original dimensions."""
    orig_w, orig_h = image.size
    down_w = max(16, int(orig_w * scale_factor))
    down_h = max(16, int(orig_h * scale_factor))
    downsampled = image.resize((down_w, down_h), resample=Image.Resampling.BILINEAR)
    return downsampled.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)


def apply_cropping(image: Image.Image, crop_fraction: float = 0.90) -> Image.Image:
    """Crop the central fraction of the image and resize back to original dimensions."""
    orig_w, orig_h = image.size
    margin_w = int(orig_w * (1.0 - crop_fraction) / 2.0)
    margin_h = int(orig_h * (1.0 - crop_fraction) / 2.0)

    cropped = image.crop((margin_w, margin_h, orig_w - margin_w, orig_h - margin_h))
    return cropped.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)


def apply_enhancement(
    image: Image.Image,
    contrast_factor: float = 1.25,
    brightness_factor: float = 1.05,
) -> Image.Image:
    """Simulate smartphone auto-enhancement filters."""
    enhanced = ImageEnhance.Contrast(image).enhance(contrast_factor)
    return ImageEnhance.Brightness(enhanced).enhance(brightness_factor)


def get_standard_perturbation_battery() -> Dict[str, Callable[[Image.Image], Image.Image]]:
    """Return dictionary of standard generalization stress functions."""
    return {
        "baseline": lambda img: img.copy(),
        "jpeg_q90": lambda img: apply_jpeg_compression(img, quality=90),
        "jpeg_q75": lambda img: apply_jpeg_compression(img, quality=75),
        "jpeg_q50": lambda img: apply_jpeg_compression(img, quality=50),
        "jpeg_q30": lambda img: apply_jpeg_compression(img, quality=30),
        "social_recompression": apply_social_media_recompression,
        "resizing_downscale_50": lambda img: apply_resizing(img, scale_factor=0.5),
        "center_crop_90": lambda img: apply_cropping(img, crop_fraction=0.90),
        "contrast_enhancement": lambda img: apply_enhancement(img, contrast_factor=1.25),
    }
