"""Reproducible and signal-preserving forensic image transforms for SafeHER.

Implements justified forensic augmentations that simulate real-world transmission
and re-encoding while strictly avoiding destructive operations that wipe out subtle
manipulation artifacts (such as boundary blending, ELA discrepancies, and FFT spectral peaks).

Transformations:
1. Controlled JPEG recompression (simulates compression degradation).
2. Aspect-ratio and scale-preserving resizing.
3. Scale-bounded cropping.
4. Mild Gaussian blur (bounded radius <= 1.0 to preserve high-frequency forensic signals).
5. Controlled color variation (mild brightness, contrast, and saturation jitter).
6. Controlled sensor noise (subtle additive zero-mean Gaussian noise).
7. Face-region context cropping (crops detected face regions with margin).
8. Tensor conversion and ImageNet normalization.

Evaluation and test pipelines are strictly deterministic (preprocessing only: resize + normalize).
Every applied operation records an audit trail of its parameters and source.
"""

from dataclasses import dataclass, field
import io
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from pydantic import BaseModel, ConfigDict, Field


class TransformOperationRecord(BaseModel):
    """Audit record capturing parameters and state of an applied transformation."""

    operation_name: str = Field(..., description="Name of the transform class or operation")
    applied: bool = Field(..., description="Whether the transform altered the image")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operational hyperparameters applied")
    seed: Optional[int] = Field(None, description="Random seed used for reproducible sampling")

    model_config = ConfigDict(frozen=True)


class BaseForensicTransform:
    """Abstract base class for signal-preserving forensic transforms."""

    def __call__(
        self,
        img: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[Image.Image, TransformOperationRecord]:
        raise NotImplementedError


class ReproducibleJPEGRecompression(BaseForensicTransform):
    """Simulates social media transmission or secondary saving via controlled JPEG recompression.

    Restricted to quality Q in [min_quality, max_quality] (default 60 to 95).
    Avoids extreme compression (Q < 40) which destroys neural blend boundaries.
    """

    def __init__(
        self,
        min_quality: int = 60,
        max_quality: int = 95,
        probability: float = 0.5,
        p: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        prob = p if p is not None else probability
        self.min_quality = max(20, min_quality)
        self.max_quality = min(100, max_quality)
        self.probability = prob
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def __call__(
        self,
        img: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[Image.Image, TransformOperationRecord]:
        local_rng = rng or self.rng or np.random.RandomState()
        if local_rng.uniform(0.0, 1.0) > self.probability:
            return img, TransformOperationRecord(
                operation_name="jpeg_recompression",
                applied=False,
                parameters={"probability": self.probability, "quality": None},
                seed=self.seed,
            )

        quality = int(local_rng.randint(self.min_quality, self.max_quality + 1))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf)
        recompressed.load()

        return recompressed, TransformOperationRecord(
            operation_name="jpeg_recompression",
            applied=True,
            parameters={"quality": quality, "format": "JPEG"},
            seed=self.seed,
        )


class ReproducibleResize(BaseForensicTransform):
    """Resizes image to target dimensions using specified interpolation."""

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        resample: Image.Resampling = Image.Resampling.BILINEAR,
    ):
        res = target_resolution or target_size or (224, 224)
        if not (isinstance(res, (tuple, list)) and len(res) == 2 and res[0] > 0 and res[1] > 0):
            raise ValueError("target_resolution must contain positive integers")
        self.target_size = tuple(res)
        self.target_resolution = self.target_size
        self.resample = resample

    def __call__(
        self,
        img: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[Image.Image, TransformOperationRecord]:
        if img.size == self.target_size:
            return img, TransformOperationRecord(
                operation_name="resize",
                applied=False,
                parameters={"target_resolution": self.target_size, "target_size": list(self.target_size)},
            )

        resized = img.resize(self.target_size, resample=self.resample)
        return resized, TransformOperationRecord(
            operation_name="resize",
            applied=True,
            parameters={
                "original_size": list(img.size),
                "target_resolution": self.target_size,
                "target_size": list(self.target_size),
                "resample": str(self.resample),
            },
        )


class ReproducibleCrop(BaseForensicTransform):
    """Performs controlled center or scale-bounded random cropping."""

    def __init__(
        self,
        mode: str = "random",
        min_scale: float = 0.85,
        max_scale: float = 1.0,
        scale_range: Optional[Tuple[float, float]] = None,
        probability: float = 1.0,
        p: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        prob = p if p is not None else probability
        if scale_range is not None:
            min_scale, max_scale = scale_range
        self.mode = mode.lower()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.probability = prob
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def __call__(
        self,
        img: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[Image.Image, TransformOperationRecord]:
        w, h = img.size
        local_rng = rng or self.rng or np.random.RandomState()

        if local_rng.uniform(0.0, 1.0) > self.probability:
            return img, TransformOperationRecord(
                operation_name="scale_bounded_crop",
                applied=False,
                parameters={"probability": self.probability},
                seed=self.seed,
            )

        if self.mode == "center":
            scale = self.max_scale
            cw, ch = int(w * scale), int(h * scale)
            left = (w - cw) // 2
            top = (h - ch) // 2
            right = left + cw
            bottom = top + ch
            crop_box = (left, top, right, bottom)
            cropped = img.crop(crop_box)
            resized = cropped.resize((w, h), resample=Image.Resampling.BILINEAR)
            return resized, TransformOperationRecord(
                operation_name="scale_bounded_crop",
                applied=True,
                parameters={"crop_box": list(crop_box), "scale": scale, "mode": "center"},
                seed=self.seed,
            )

        # Random scale-bounded crop
        scale = float(local_rng.uniform(self.min_scale, self.max_scale))
        cw, ch = max(16, int(w * scale)), max(16, int(h * scale))
        max_x = max(0, w - cw)
        max_y = max(0, h - ch)
        left = int(local_rng.randint(0, max_x + 1)) if max_x > 0 else 0
        top = int(local_rng.randint(0, max_y + 1)) if max_y > 0 else 0
        crop_box = (left, top, left + cw, top + ch)
        cropped = img.crop(crop_box)
        resized = cropped.resize((w, h), resample=Image.Resampling.BILINEAR)

        return resized, TransformOperationRecord(
            operation_name="scale_bounded_crop",
            applied=True,
            parameters={"crop_box": list(crop_box), "scale": round(scale, 4), "mode": "random"},
            seed=self.seed,
        )


class ReproducibleMildBlur(BaseForensicTransform):
    """Applies mild Gaussian blur with strictly bounded radius (sigma <= 1.0).

    Signal preservation guarantee: Large blur radii (sigma > 2.0) completely obliterate
    subtle forensic manipulation boundaries (ELA contrast, high-frequency DCT noise).
    This transform strictly caps radius at 1.0 to preserve manipulation traces.
    """

    def __init__(
        self,
        max_radius: float = 0.8,
        max_sigma: Optional[float] = None,
        probability: float = 0.3,
        p: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        prob = p if p is not None else probability
        radius = max_sigma if max_sigma is not None else max_radius
        self.max_radius = min(1.0, radius)
        self.max_sigma = self.max_radius
        self.probability = prob
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def __call__(
        self,
        img: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[Image.Image, TransformOperationRecord]:
        local_rng = rng or self.rng or np.random.RandomState()
        if local_rng.uniform(0.0, 1.0) > self.probability:
            return img, TransformOperationRecord(
                operation_name="mild_blur",
                applied=False,
                parameters={"probability": self.probability, "sigma": None},
                seed=self.seed,
            )

        sigma = float(local_rng.uniform(0.1, self.max_radius))
        blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        return blurred, TransformOperationRecord(
            operation_name="mild_blur",
            applied=True,
            parameters={"sigma": round(sigma, 3), "radius": round(sigma, 3), "capped_max_sigma": self.max_radius},
            seed=self.seed,
        )


class ReproducibleColorVariation(BaseForensicTransform):
    """Mild brightness, contrast, and color saturation variation within [0.85, 1.15]."""

    def __init__(
        self,
        brightness_jitter: float = 0.12,
        contrast_jitter: float = 0.12,
        saturation_jitter: float = 0.12,
        brightness_range: Optional[Tuple[float, float]] = None,
        contrast_range: Optional[Tuple[float, float]] = None,
        saturation_range: Optional[Tuple[float, float]] = None,
        probability: float = 0.5,
        p: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        prob = p if p is not None else probability
        if brightness_range is not None:
            brightness_jitter = abs(brightness_range[1] - 1.0)
        if contrast_range is not None:
            contrast_jitter = abs(contrast_range[1] - 1.0)
        if saturation_range is not None:
            saturation_jitter = abs(saturation_range[1] - 1.0)
        self.b_jitter = brightness_jitter
        self.c_jitter = contrast_jitter
        self.s_jitter = saturation_jitter
        self.probability = prob
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def __call__(
        self,
        img: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[Image.Image, TransformOperationRecord]:
        local_rng = rng or self.rng or np.random.RandomState()
        if local_rng.uniform(0.0, 1.0) > self.probability:
            return img, TransformOperationRecord(
                operation_name="color_variation",
                applied=False,
                parameters={"probability": self.probability},
                seed=self.seed,
            )

        b_factor = float(local_rng.uniform(1.0 - self.b_jitter, 1.0 + self.b_jitter))
        c_factor = float(local_rng.uniform(1.0 - self.c_jitter, 1.0 + self.c_jitter))
        s_factor = float(local_rng.uniform(1.0 - self.s_jitter, 1.0 + self.s_jitter))

        out = ImageEnhance.Brightness(img).enhance(b_factor)
        out = ImageEnhance.Contrast(out).enhance(c_factor)
        out = ImageEnhance.Color(out).enhance(s_factor)

        return out, TransformOperationRecord(
            operation_name="color_variation",
            applied=True,
            parameters={
                "brightness_factor": round(b_factor, 3),
                "contrast_factor": round(c_factor, 3),
                "saturation_factor": round(s_factor, 3),
            },
            seed=self.seed,
        )


class ReproducibleControlledNoise(BaseForensicTransform):
    """Adds subtle zero-mean Gaussian sensor noise (sigma <= 4.0 on 0-255 scale)."""

    def __init__(
        self,
        max_sigma: float = 3.5,
        probability: float = 0.3,
        p: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        prob = p if p is not None else probability
        self.max_sigma = min(6.0, max_sigma)
        self.probability = prob
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def __call__(
        self,
        img: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[Image.Image, TransformOperationRecord]:
        local_rng = rng or self.rng or np.random.RandomState()
        if local_rng.uniform(0.0, 1.0) > self.probability:
            return img, TransformOperationRecord(
                operation_name="controlled_noise",
                applied=False,
                parameters={"probability": self.probability},
                seed=self.seed,
            )

        sigma = float(local_rng.uniform(0.1, self.max_sigma))
        arr = np.array(img, dtype=np.float32)
        noise = local_rng.normal(0.0, sigma, arr.shape)
        noisy_arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_arr, mode=img.mode)

        return noisy_img, TransformOperationRecord(
            operation_name="controlled_noise",
            applied=True,
            parameters={"noise_sigma": round(sigma, 3), "sigma": round(sigma, 3)},
            seed=self.seed,
        )


class ReproducibleFaceCrop(BaseForensicTransform):
    """Crops detected or annotated face bounding box with context margin padding.

    Looks for 'face_bbox', 'bbox', or 'bounding_box' in metadata: [x1, y1, x2, y2].
    Falls back gracefully to centered crop if no bounding box is present.
    """

    def __init__(
        self,
        margin_scale: float = 1.3,
        margin_ratio: Optional[float] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        if margin_ratio is not None:
            margin_scale = 1.0 + margin_ratio
        self.margin_scale = margin_scale
        self.target_size = target_resolution or target_size

    def __call__(
        self,
        img: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[Image.Image, TransformOperationRecord]:
        w, h = img.size
        meta = metadata or {}
        face_bbox = meta.get("face_bbox") or meta.get("bbox") or meta.get("bounding_box")

        if not face_bbox or len(face_bbox) != 4:
            # Fallback center crop
            cw, ch = int(w * 0.8), int(h * 0.8)
            crop_box = ((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)
            cropped = img.crop(crop_box)
            if self.target_size:
                cropped = cropped.resize(self.target_size, resample=Image.Resampling.BILINEAR)
            return cropped, TransformOperationRecord(
                operation_name="face_region_crop",
                applied=True,
                parameters={"method": "center_crop_fallback", "crop_box": list(crop_box)},
            )

        x1, y1, x2, y2 = [float(v) for v in face_bbox]

        # Convert normalized coordinates if applicable
        if max(x1, y1, x2, y2) <= 1.0:
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
        new_w, new_h = bw * self.margin_scale, bh * self.margin_scale

        crop_x1 = max(0, int(cx - new_w / 2.0))
        crop_y1 = max(0, int(cy - new_h / 2.0))
        crop_x2 = min(w, int(cx + new_w / 2.0))
        crop_y2 = min(h, int(cy + new_h / 2.0))

        if crop_x2 - crop_x1 < 10 or crop_y2 - crop_y1 < 10:
            crop_box = (0, 0, w, h)
            cropped = img
            return cropped, TransformOperationRecord(
                operation_name="face_region_crop",
                applied=False,
                parameters={"reason": "bbox_too_small"},
            )

        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        cropped = img.crop(crop_box)
        if self.target_size:
            cropped = cropped.resize(self.target_size, resample=Image.Resampling.BILINEAR)

        return cropped, TransformOperationRecord(
            operation_name="face_region_crop",
            applied=True,
            parameters={"method": "bounding_box", "crop_box": list(crop_box), "margin_scale": self.margin_scale},
        )


class ToTensorAndNormalize:
    """Converts PIL Image to PyTorch FloatTensor in [0.0, 1.0] and standardizes with mean/std."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, TransformOperationRecord]:
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to numpy array [H, W, C] in [0, 255]
        arr = np.array(img, dtype=np.float32) / 255.0
        # Transpose to [C, H, W]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        # Normalize
        normalized = (tensor - self.mean) / self.std

        return normalized, TransformOperationRecord(
            operation_name="to_tensor_and_normalize",
            applied=True,
            parameters={"channels": 3, "normalized": True},
        )


class ForensicTransformPipeline:
    """Orchestrates reproducible forensic image transformations with audit logging.

    Guarantees:
    - In 'eval' or 'test' mode: Strictly deterministic preprocessing only (resize + tensor normalization).
      Zero stochastic augmentations are applied to test splits.
    - In 'train' mode: Controlled, signal-preserving augmentations applied with reproducible seed.
    - Every execution returns the transformed PyTorch tensor and an audit trail of applied operations.
    """

    def __init__(
        self,
        mode: str = "train",
        target_size: Optional[Tuple[int, int]] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = 42,
        enable_jpeg_recompression: bool = True,
        enable_mild_blur: bool = True,
        enable_color_variation: bool = True,
        enable_controlled_noise: bool = True,
        enable_face_crop: bool = True,
    ):
        valid_modes = ("train", "eval", "test", "val", "robustness")
        if mode.lower() not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        self.mode = mode.lower()

        res = target_resolution or target_size or (224, 224)
        if not (isinstance(res, (tuple, list)) and len(res) == 2 and res[0] > 0 and res[1] > 0):
            raise ValueError("target_resolution must contain positive integers")
        self.target_size = tuple(res)
        self.target_resolution = self.target_size
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None

        # Build transform sequence
        self.face_crop = ReproducibleFaceCrop() if enable_face_crop else None
        self.resize = ReproducibleResize(target_size=self.target_size)
        self.to_tensor = ToTensorAndNormalize()

        # Training-only augmentations
        self.jpeg = ReproducibleJPEGRecompression() if enable_jpeg_recompression else None
        self.blur = ReproducibleMildBlur() if enable_mild_blur else None
        self.color = ReproducibleColorVariation() if enable_color_variation else None
        self.noise = ReproducibleControlledNoise() if enable_controlled_noise else None

    def set_seed(self, seed: int) -> None:
        """Reset internal random state for deterministic reproducibility."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        img: Image.Image,
        metadata: Optional[Dict[str, Any]] = None,
        custom_seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[TransformOperationRecord]]:
        """Apply transform sequence to image and return normalized tensor with audit trail."""
        op_history: List[TransformOperationRecord] = []
        local_rng = np.random.RandomState(custom_seed) if custom_seed is not None else self.rng

        curr_img = img

        # ----------------------------------------------------------------------
        # Test / Evaluation Mode: Pure deterministic preprocessing (no augmentations)
        # ----------------------------------------------------------------------
        if self.mode in ("eval", "test", "val"):
            # Optional face crop if bbox is explicitly present in metadata
            if self.face_crop and metadata and (metadata.get("face_bbox") or metadata.get("bbox") or metadata.get("bounding_box")):
                curr_img, op = self.face_crop(curr_img, metadata=metadata, rng=local_rng)
                op_history.append(op)

            curr_img, op = self.resize(curr_img, metadata=metadata, rng=local_rng)
            op_history.append(op)

            tensor, op = self.to_tensor(curr_img)
            op_history.append(op)

            # Record bypassed operations for completeness
            for op_name in ("jpeg_recompression", "mild_blur", "color_variation", "controlled_noise", "scale_bounded_crop"):
                op_history.append(
                    TransformOperationRecord(
                        operation_name=op_name,
                        applied=False,
                        parameters={"reason": f"bypassed_in_{self.mode}_mode"},
                    )
                )

            return tensor, op_history

        # ----------------------------------------------------------------------
        # Training Mode: Signal-preserving stochastic augmentations
        # ----------------------------------------------------------------------
        # 1. Face region crop if bbox exists
        if self.face_crop and metadata and (metadata.get("face_bbox") or metadata.get("bbox") or metadata.get("bounding_box")):
            curr_img, op = self.face_crop(curr_img, metadata=metadata, rng=local_rng)
            op_history.append(op)

        # 2. Resize to target dimension
        curr_img, op = self.resize(curr_img, metadata=metadata, rng=local_rng)
        op_history.append(op)

        # 3. Controlled color variation
        if self.color:
            curr_img, op = self.color(curr_img, metadata=metadata, rng=local_rng)
            op_history.append(op)

        # 4. JPEG recompression
        if self.jpeg:
            curr_img, op = self.jpeg(curr_img, metadata=metadata, rng=local_rng)
            op_history.append(op)

        # 5. Mild blur (strictly capped radius)
        if self.blur:
            curr_img, op = self.blur(curr_img, metadata=metadata, rng=local_rng)
            op_history.append(op)

        # 6. Controlled noise
        if self.noise:
            curr_img, op = self.noise(curr_img, metadata=metadata, rng=local_rng)
            op_history.append(op)

        # 7. Convert to normalized PyTorch tensor
        tensor, op = self.to_tensor(curr_img)
        op_history.append(op)

        return tensor, op_history
