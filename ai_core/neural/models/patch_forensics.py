"""Patch-based facial manipulation detector for SafeHER Model B.

Implements a pure PyTorch PatchForensics architecture with dual heads:
1. Spatial Patch Anomaly Head: Fully convolutional dense prediction producing an
   H_p x W_p spatial anomaly map to localize manipulation seams and donor patches.
2. Fine-Grained Categorization Head: Global feature classifier distinguishing between
   face swaps, facial reenactment, inpainting, localized replacement, and authentic faces.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_core.neural.facial.schemas import FacialManipulationType
from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.schemas import InputConfiguration, ModelMetadata

# Model B Category Mapping
MODEL_B_CLASS_MAPPING: Dict[int, str] = {
    0: FacialManipulationType.NONE_AUTHENTIC.value,
    1: FacialManipulationType.FACE_SWAP.value,
    2: FacialManipulationType.FACIAL_REENACTMENT.value,
    3: FacialManipulationType.FACIAL_INPAINTING.value,
    4: FacialManipulationType.LOCALIZED_FACE_REPLACEMENT.value,
}

MODEL_B_LABEL_TO_INDEX: Dict[str, int] = {v: k for k, v in MODEL_B_CLASS_MAPPING.items()}

MODEL_B_SUPPORTED_DOMAIN = (
    "Facial manipulation and deepfake detection (face swaps, facial reenactment, inpainting, "
    "and localized face replacement on normalized 224x224 RGB face crops with boundary context margins)."
)


class ConvBNReLU(nn.Module):
    """Standard Conv2d + BatchNorm2d + ReLU block."""

    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchForensicsFacialDetector(BaseNeuralDetector):
    """PatchForensics facial manipulation detector with spatial localization and category classification."""

    def __init__(
        self,
        num_categories: int = 5,
        model_version: str = "1.0.0-m4.4-model-b",
        supported_domain: str = MODEL_B_SUPPORTED_DOMAIN,
        input_config: Optional[InputConfiguration] = None,
        is_trained: bool = False,
        weights_sha256: Optional[str] = None,
        training_dataset_id: Optional[str] = None,
    ):
        super().__init__(
            architecture="patch_forensics_facial",
            model_version=model_version,
            supported_domain=supported_domain,
            class_mapping=MODEL_B_CLASS_MAPPING,
            input_config=input_config or InputConfiguration(),
            is_trained=is_trained,
            weights_sha256=weights_sha256,
            training_dataset_id=training_dataset_id,
        )
        self.num_categories = num_categories

        # Fully convolutional backbone preserving 14x14 spatial patch grid (stride 16 total)
        # Input: (B, 3, 224, 224)
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, k=3, s=2, p=1),   # (B, 32, 112, 112)
            ConvBNReLU(32, 64, k=3, s=2, p=1),  # (B, 64, 56, 56)
        )

        self.stage1 = nn.Sequential(
            ConvBNReLU(64, 64, k=3, s=1, p=1),
            ConvBNReLU(64, 128, k=3, s=2, p=1), # (B, 128, 28, 28)
        )

        self.stage2 = nn.Sequential(
            ConvBNReLU(128, 128, k=3, s=1, p=1),
            ConvBNReLU(128, 256, k=3, s=2, p=1), # (B, 256, 14, 14)
        )

        self.stage3 = nn.Sequential(
            ConvBNReLU(256, 256, k=3, s=1, p=1),
            ConvBNReLU(256, 256, k=3, s=1, p=1), # (B, 256, 14, 14)
        )

        # Head 1: Spatial Patch Anomaly Localization Head (outputs (B, 1, 14, 14) anomaly map)
        self.patch_localization_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Head 2: Fine-Grained Manipulation Categorization Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.category_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_categories),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled 256-dimensional feature representations."""
        self.validate_input_tensor(x)
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        pooled = self.pool(feat)
        return torch.flatten(pooled, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass executing both dense patch localization and category classification.

        Args:
            x: Preprocessed input tensor of shape (B, 3, 224, 224).

        Returns:
            Tuple of:
            - spatial_patch_map: Tensor of shape (B, 1, 14, 14) with patch anomaly scores in [0, 1].
            - category_logits: Tensor of shape (B, num_categories) with uncalibrated logits.
        """
        self.validate_input_tensor(x)

        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        spatial_features = self.stage3(feat)  # (B, 256, 14, 14)

        # Dense patch anomaly map
        patch_anomaly_map = self.patch_localization_head(spatial_features)

        # Global category logits
        pooled = self.pool(spatial_features)
        category_logits = self.category_head(torch.flatten(pooled, 1))

        return patch_anomaly_map, category_logits
