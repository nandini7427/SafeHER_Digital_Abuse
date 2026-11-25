"""Face region extraction, normalization, and coordinate preservation engine for SafeHER Model B.

Extracts face candidates with forensic context margins (preserving boundary blending seams),
assesses quality and visibility conditions (low-res, truncation), and maintains exact
bidirectional coordinate mappings between patch grids and source image frames.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
from PIL import Image
import torch

from ai_core.neural.facial.schemas import (
    FaceCandidate,
    FaceCondition,
    FaceDetectionStatus,
    NormalizedBoundingBox,
)


class FaceRegionPreprocessor:
    """Extracts, validates, and normalizes face regions while preserving source coordinates."""

    def __init__(
        self,
        min_face_resolution: int = 64,
        margin_ratio: float = 0.20,
        target_resolution: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """Initialize FaceRegionPreprocessor.

        Args:
            min_face_resolution: Minimum pixel edge length to qualify as normal resolution.
            margin_ratio: Proportional margin expansion around face box to capture blend seams.
            target_resolution: Fixed spatial resolution (H, W) for neural backbone input.
            mean: Channel-wise normalization mean values.
            std: Channel-wise normalization standard deviation values.
        """
        self.min_face_resolution = min_face_resolution
        self.margin_ratio = margin_ratio
        self.target_resolution = target_resolution
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def _to_pil(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Convert input image array or PIL to RGB PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected 3-channel RGB image array, got shape {image.shape}")
            if image.dtype == np.uint8:
                return Image.fromarray(image, mode="RGB")
            else:
                arr = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
                return Image.fromarray(arr, mode="RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def detect_face_candidates(
        self,
        image: Union[Image.Image, np.ndarray],
        provided_boxes: Optional[List[NormalizedBoundingBox]] = None,
    ) -> Tuple[FaceDetectionStatus, List[FaceCandidate]]:
        """Identify face candidate regions in image and assess quality conditions.

        Args:
            image: Source image (PIL or NumPy array).
            provided_boxes: Verified upstream or annotated bounding boxes. If None, face
                            localization is unavailable; this component does not pretend that
                            a central image crop is a detected face.

        Returns:
            Tuple of (FaceDetectionStatus, List[FaceCandidate]).
        """
        pil_img = self._to_pil(image)
        img_w, img_h = pil_img.size

        if img_w < 16 or img_h < 16:
            return FaceDetectionStatus.INVALID_IMAGE, []

        if provided_boxes is None:
            # A center crop is not face detection. Require an upstream detector or
            # reviewed/annotated face boxes before running facial-manipulation analysis.
            return FaceDetectionStatus.LOCALIZATION_UNAVAILABLE, []

        boxes_to_process: List[NormalizedBoundingBox] = list(provided_boxes)

        if not boxes_to_process:
            return FaceDetectionStatus.NO_FACE_DETECTED, []

        candidates: List[FaceCandidate] = []
        for idx, box in enumerate(boxes_to_process):
            # Calculate pixel bounds
            x1, y1, x2, y2 = box.to_pixel_coordinates(img_w, img_h)
            pixel_w = x2 - x1
            pixel_h = y2 - y1

            conditions: List[FaceCondition] = []

            # 1. Low resolution assessment
            if min(pixel_w, pixel_h) < self.min_face_resolution:
                conditions.append(FaceCondition.LOW_RESOLUTION)

            # 2. Border truncation / partially visible assessment
            border_margin = 0.02
            if (
                box.x_min <= border_margin
                or box.y_min <= border_margin
                or box.x_max >= (1.0 - border_margin)
                or box.y_max >= (1.0 - border_margin)
            ):
                conditions.append(FaceCondition.PARTIALLY_VISIBLE)

            # 3. Aspect ratio / extreme pose assessment
            aspect_ratio = box.height / max(1e-5, box.width)
            if aspect_ratio < 0.7 or aspect_ratio > 2.2:
                conditions.append(FaceCondition.EXTREME_POSE)

            if not conditions:
                conditions.append(FaceCondition.NORMAL)

            candidate = FaceCandidate(
                face_id=f"FACE-{uuid.uuid4().hex[:8].upper()}",
                bounding_box=box,
                confidence=1.0,
                face_conditions=conditions,
                crop_resolution=(pixel_w, pixel_h),
                is_primary=False,
            )
            candidates.append(candidate)

        # Sort by prominence (area) and designate primary face
        candidates.sort(key=lambda c: c.bounding_box.area, reverse=True)
        if candidates:
            # Reconstruct first candidate as primary (immutable model)
            primary = candidates[0].model_copy(update={"is_primary": True})
            candidates[0] = primary

        status = (
            FaceDetectionStatus.FACE_DETECTED
            if len(candidates) == 1
            else FaceDetectionStatus.MULTIPLE_FACES_DETECTED
        )
        return status, candidates

    def extract_face_crop(
        self,
        image: Union[Image.Image, np.ndarray],
        candidate: FaceCandidate,
        apply_margin: bool = True,
    ) -> Tuple[torch.Tensor, NormalizedBoundingBox]:
        """Extract, resize, and normalize face crop while preserving source coordinates.

        Args:
            image: Source image.
            candidate: Selected FaceCandidate to extract.
            apply_margin: Whether to expand box by margin_ratio to capture blending seams.

        Returns:
            Tuple of (preprocessed_tensor of shape (1, 3, H, W), expanded_bounding_box).
        """
        pil_img = self._to_pil(image)
        img_w, img_h = pil_img.size

        box = candidate.bounding_box
        if apply_margin:
            box_w = box.width
            box_h = box.height
            pad_x = box_w * self.margin_ratio
            pad_y = box_h * self.margin_ratio

            x_min = max(0.0, box.x_min - pad_x)
            y_min = max(0.0, box.y_min - pad_y)
            x_max = min(1.0, box.x_max + pad_x)
            y_max = min(1.0, box.y_max + pad_y)

            expanded_box = NormalizedBoundingBox(
                x_min=round(x_min, 4),
                y_min=round(y_min, 4),
                x_max=round(x_max, 4),
                y_max=round(y_max, 4),
            )
        else:
            expanded_box = box

        x1, y1, x2, y2 = expanded_box.to_pixel_coordinates(img_w, img_h)
        crop = pil_img.crop((x1, y1, x2, y2))

        # Resize to fixed target resolution
        target_h, target_w = self.target_resolution
        resized = crop.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)

        # Convert to normalized float32 tensor
        arr = np.array(resized, dtype=np.float32) / 255.0  # HWC
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()

        return tensor, expanded_box

    def map_crop_coords_to_source(
        self,
        crop_box: NormalizedBoundingBox,
        expanded_box: NormalizedBoundingBox,
    ) -> NormalizedBoundingBox:
        """Map a normalized bounding box within the face crop back into source image coordinates.

        Args:
            crop_box: Bounding box relative to the extracted face crop [0, 1].
            expanded_box: The face crop's bounding box relative to source image [0, 1].

        Returns:
            NormalizedBoundingBox relative to the original source image frame.
        """
        src_x_min = expanded_box.x_min + crop_box.x_min * expanded_box.width
        src_y_min = expanded_box.y_min + crop_box.y_min * expanded_box.height
        src_x_max = expanded_box.x_min + crop_box.x_max * expanded_box.width
        src_y_max = expanded_box.y_min + crop_box.y_max * expanded_box.height

        return NormalizedBoundingBox(
            x_min=max(0.0, min(1.0, round(src_x_min, 4))),
            y_min=max(0.0, min(1.0, round(src_y_min, 4))),
            x_max=max(0.0, min(1.0, round(src_x_max, 4))),
            y_max=max(0.0, min(1.0, round(src_y_max, 4))),
        )
