"""MobileNetV3-Large neural image detector for SafeHER Model A.

Implements pure PyTorch MobileNetV3-Large feature extraction with a customized
forensic classification head for differentiating AI-generated images from camera photographs.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.schemas import (
    MODEL_A_CLASS_MAPPING,
    MODEL_A_SUPPORTED_DOMAIN,
    InputConfiguration,
    NeuralArchitecture,
)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, in_channels: int, reduced_dim: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, reduced_dim, 1)
        self.fc2 = nn.Conv2d(reduced_dim, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class InvertedResidualMBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) block with optional Hardswish and SE."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        se_ratio: float = 0.0,
        use_hard_swish: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_c == out_c
        hidden_dim = int(round(in_c * expand_ratio))

        layers: List[nn.Module] = []
        # Expansion phase
        if hidden_dim != in_c:
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True) if use_hard_swish else nn.ReLU(inplace=True),
            ])

        # Depthwise phase
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True) if use_hard_swish else nn.ReLU(inplace=True),
        ])

        # Squeeze-and-Excitation
        if se_ratio > 0:
            reduced = max(1, int(in_c * se_ratio))
            layers.append(SqueezeExcitation(hidden_dim, reduced))

        # Output projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3LargeDetector(BaseNeuralDetector):
    """MobileNetV3-Large neural detector for AI-generated vs. camera photographs."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        hidden_dim: int = 256,
        model_version: str = "1.0.0-m4.3",
        supported_domain: str = MODEL_A_SUPPORTED_DOMAIN,
        class_mapping: Optional[Dict[int, str]] = None,
        input_config: Optional[InputConfiguration] = None,
        is_trained: bool = False,
        weights_sha256: Optional[str] = None,
        training_dataset_id: Optional[str] = None,
    ):
        super().__init__(
            architecture=NeuralArchitecture.MOBILENET_V3_LARGE.value,
            model_version=model_version,
            supported_domain=supported_domain,
            class_mapping=class_mapping or MODEL_A_CLASS_MAPPING,
            input_config=input_config or InputConfiguration(),
            is_trained=is_trained,
            weights_sha256=weights_sha256,
            training_dataset_id=training_dataset_id,
        )
        self.num_classes = num_classes
        self.feature_dim = 960

        in_channels = self.input_config.expected_channels

        # Stem: Conv 3x3 s=2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )

        # Standard MobileNetV3-Large Inverted Residual stages:
        # (in_c, out_c, k, s, exp, se_ratio, use_hard_swish)
        stage_configs = [
            (16, 16, 3, 1, 1.0, 0.0, False),
            (16, 24, 3, 2, 4.0, 0.0, False),
            (24, 24, 3, 1, 3.0, 0.0, False),
            (24, 40, 5, 2, 3.0, 0.25, False),
            (40, 40, 5, 1, 3.0, 0.25, False),
            (40, 40, 5, 1, 3.0, 0.25, False),
            (40, 80, 3, 2, 6.0, 0.0, True),
            (80, 80, 3, 1, 2.5, 0.0, True),
            (80, 80, 3, 1, 2.3, 0.0, True),
            (80, 80, 3, 1, 2.3, 0.0, True),
            (80, 112, 3, 1, 6.0, 0.25, True),
            (112, 112, 3, 1, 6.0, 0.25, True),
            (112, 160, 5, 2, 6.0, 0.25, True),
            (160, 160, 5, 1, 6.0, 0.25, True),
            (160, 160, 5, 1, 6.0, 0.25, True),
        ]

        blocks: List[nn.Module] = []
        for in_c, out_c, k, s, exp, se, hs in stage_configs:
            blocks.append(InvertedResidualMBConv(in_c, out_c, k, s, exp, se, hs))
        self.blocks = nn.Sequential(*blocks)

        # Feature expansion conv: 160 -> 960
        self.conv_head = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            nn.Hardswish(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Customized Forensic Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(960, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 960-dimensional pooled features."""
        self.validate_input_tensor(x)
        feat = self.stem(x)
        feat = self.blocks(feat)
        feat = self.conv_head(feat)
        feat = self.pool(feat)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw logits of shape (B, num_classes)."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
