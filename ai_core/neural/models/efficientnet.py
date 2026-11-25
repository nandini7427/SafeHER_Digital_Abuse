"""EfficientNet-B0 neural image detector for SafeHER Model A.

Implements pure PyTorch EfficientNet-B0 feature extraction with a customized
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
    """Squeeze-and-Excitation channel attention block for EfficientNet."""

    def __init__(self, in_channels: int, reduced_dim: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, reduced_dim, 1)
        self.fc2 = nn.Conv2d(reduced_dim, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class EfficientMBConv(nn.Module):
    """EfficientNet Inverted Residual Bottleneck Block (MBConv)."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_c == out_c
        hidden_dim = in_c * expand_ratio

        layers: List[nn.Module] = []
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
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
            nn.SiLU(inplace=True),
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


class EfficientNetB0Detector(BaseNeuralDetector):
    """EfficientNet-B0 neural detector for AI-generated vs. camera photographs."""

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
            architecture=NeuralArchitecture.EFFICIENTNET_B0.value,
            model_version=model_version,
            supported_domain=supported_domain,
            class_mapping=class_mapping or MODEL_A_CLASS_MAPPING,
            input_config=input_config or InputConfiguration(),
            is_trained=is_trained,
            weights_sha256=weights_sha256,
            training_dataset_id=training_dataset_id,
        )
        self.num_classes = num_classes
        self.feature_dim = 1280

        in_channels = self.input_config.expected_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Standard EfficientNet-B0 stage configurations:
        # (in_c, out_c, k, s, exp, se_ratio)
        stage_configs = [
            (32, 16, 3, 1, 1, 0.25),
            (16, 24, 3, 2, 6, 0.25),
            (24, 24, 3, 1, 6, 0.25),
            (24, 40, 5, 2, 6, 0.25),
            (40, 40, 5, 1, 6, 0.25),
            (40, 80, 3, 2, 6, 0.25),
            (80, 80, 3, 1, 6, 0.25),
            (80, 80, 3, 1, 6, 0.25),
            (80, 112, 5, 1, 6, 0.25),
            (112, 112, 5, 1, 6, 0.25),
            (112, 112, 5, 1, 6, 0.25),
            (112, 192, 5, 2, 6, 0.25),
            (192, 192, 5, 1, 6, 0.25),
            (192, 192, 5, 1, 6, 0.25),
            (192, 192, 5, 1, 6, 0.25),
            (192, 320, 3, 1, 6, 0.25),
        ]

        blocks: List[nn.Module] = []
        for in_c, out_c, k, s, exp, se in stage_configs:
            blocks.append(EfficientMBConv(in_c, out_c, k, s, exp, se))
        self.blocks = nn.Sequential(*blocks)

        # Head conv
        self.conv_head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Customized Forensic Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1280-dimensional pooled features."""
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
