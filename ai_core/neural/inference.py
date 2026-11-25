"""Inference service for SafeHER Model A neural image detectors.

Provides deterministic, CPU-optimized inference execution, strict input validation,
rejection of missing or untrained model artifacts, and structured result packaging
with explicit forensic triage disclaimers.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from ai_core.neural.models.base import BaseNeuralDetector
from ai_core.neural.models.factory import (
    create_image_detector,
    load_detector_checkpoint,
)
from ai_core.neural.schemas import (
    MANDATORY_NEURAL_DISCLAIMER,
    MODEL_A_CLASS_MAPPING,
    BatchInferenceResult,
    InferenceStatus,
    InputConfiguration,
    ModelInferenceResult,
    ModelMetadata,
    NeuralArchitecture,
)


def configure_deterministic_inference(
    num_threads: int = 2,
    seed: int = 42,
    use_deterministic_algorithms: bool = True,
) -> None:
    """Configure CPU execution for deterministic reproducibility."""
    torch.set_num_threads(max(1, num_threads))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if use_deterministic_algorithms:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some operations on certain PyTorch builds lack deterministic CPU kernels
            pass


class ModelAInferenceService:
    """SafeHER Model A Neural Detector Inference Service."""

    def __init__(
        self,
        model: Optional[BaseNeuralDetector] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        expected_sha256: Optional[str] = None,
        architecture: Union[str, NeuralArchitecture] = NeuralArchitecture.MOBILENET_V3_LARGE,
        num_threads: int = 2,
        deterministic: bool = True,
        enforce_trained: bool = True,
    ):
        """Initialize the Model A inference service.

        Args:
            model: Pre-instantiated detector instance. If None, architecture is instantiated.
            checkpoint_path: Optional path to serialized weights file.
            expected_sha256: Optional expected SHA-256 digest of weights file.
            architecture: Architecture to instantiate if model is None.
            num_threads: Number of CPU threads to allocate for inference.
            deterministic: Whether to enforce deterministic CPU execution.
            enforce_trained: Legacy compatibility parameter; unverified weights are always rejected.
        """
        self.num_threads = num_threads
        self.deterministic = deterministic
        self.enforce_trained = enforce_trained
        self._input_config = InputConfiguration()
        self._class_mapping = MODEL_A_CLASS_MAPPING
        self._architecture_name = (
            architecture.value if isinstance(architecture, NeuralArchitecture) else str(architecture)
        )
        self._model_version = "1.0.0-m4.3"

        if deterministic:
            configure_deterministic_inference(num_threads=num_threads)

        self.model: Optional[BaseNeuralDetector] = None
        self.status: InferenceStatus = InferenceStatus.READY
        self.status_message: str = "Ready"

        if model is not None:
            self.model = model
            self._architecture_name = model.get_metadata().architecture
            self._model_version = model.get_metadata().model_version
            self._input_config = model.input_config
            self._class_mapping = model.class_mapping

            if not model.is_trained:
                self.model = None
                self.status = InferenceStatus.UNTRAINED_MODEL_REJECTED
                self.status_message = "Model rejected: weights have not undergone verified training."
            else:
                self.model.eval()
                self.status = InferenceStatus.READY
                self.status_message = "Model loaded and verified."

        elif checkpoint_path is not None:
            path = Path(checkpoint_path)
            if not path.is_file():
                self.status = InferenceStatus.MODEL_UNAVAILABLE
                self.status_message = f"Weights artifact not found at {path}."
            else:
                try:
                    detector = create_image_detector(
                        architecture=self._architecture_name,
                        input_config=self._input_config,
                        class_mapping=self._class_mapping,
                    )
                    self.model = load_detector_checkpoint(
                        model=detector,
                        checkpoint_path=path,
                        expected_sha256=expected_sha256,
                        enforce_trained=True,
                    )
                    self.status = InferenceStatus.READY
                    self.status_message = "Weights loaded and verified."
                except ValueError as ve:
                    if "UNTRAINED" in str(ve).upper():
                        self.status = InferenceStatus.UNTRAINED_MODEL_REJECTED
                    else:
                        self.status = InferenceStatus.INTEGRITY_VERIFICATION_FAILED
                    self.status_message = str(ve)
                except Exception as ex:
                    self.status = InferenceStatus.ERROR
                    self.status_message = f"Checkpoint loading failed: {ex}"
        else:
            # No model and no checkpoint provided
            if enforce_trained:
                self.status = InferenceStatus.MODEL_UNAVAILABLE
                self.status_message = (
                    "No trained weights checkpoint supplied. "
                    "SafeHER policy rejects unweighted or unverified models for inference."
                )
            else:
                # Never make predictions from randomly initialized weights,
                # including when the legacy development bypass is requested.
                self.status = InferenceStatus.UNTRAINED_MODEL_REJECTED
                self.status_message = (
                    "No verified trained checkpoint was supplied. Randomly initialized weights "
                    "are not eligible for image analysis."
                )

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess a raw image into a normalized 4D PyTorch tensor.

        Args:
            image: RGB image as PIL Image or HWC numpy array (uint8 or float32 in [0, 1]).

        Returns:
            Normalized PyTorch float32 tensor of shape (1, 3, 224, 224).

        Raises:
            ValueError: If image dimensions or format cannot be processed.
        """
        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(f"Expected 3D image array (H, W, C), got ndim={image.ndim}")
            if image.shape[2] != 3:
                raise ValueError(f"Expected 3 color channels (RGB), got {image.shape[2]}")
            if image.dtype == np.uint8:
                image_arr = image.astype(np.float32) / 255.0
            else:
                image_arr = image.astype(np.float32)
                if image_arr.max() > 1.0:
                    image_arr = image_arr / 255.0
            pil_img = Image.fromarray((image_arr * 255.0).astype(np.uint8), mode="RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image).__name__}")

        # Resize to expected spatial resolution
        target_h, target_w = self._input_config.spatial_resolution
        resized = pil_img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)

        # Convert to float numpy array in [0, 1]
        arr = np.array(resized, dtype=np.float32) / 255.0  # (H, W, C)

        # Normalize with channel mean and std
        mean = np.array(self._input_config.mean, dtype=np.float32)
        std = np.array(self._input_config.std, dtype=np.float32)
        arr = (arr - mean) / std

        # Transpose HWC -> CHW and add batch dimension -> (1, C, H, W)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    def predict(self, x: torch.Tensor) -> ModelInferenceResult:
        """Execute single-sample or batch inference on preprocessed tensor.

        Args:
            x: Preprocessed tensor of shape (1, 3, 224, 224) or (B, 3, 224, 224).

        Returns:
            ModelInferenceResult payload.

        Raises:
            TypeError, ValueError: If input tensor fails shape, channel, dtype, or value validation.
        """
        if self.status != InferenceStatus.READY or self.model is None:
            # Return explicit non-fabricated triage result indicating model unavailability or rejection
            return ModelInferenceResult(
                raw_logits=[],
                uncalibrated_scores={},
                predicted_class="UNKNOWN",
                model_version=self._model_version,
                architecture=self._architecture_name,
                input_config=self._input_config,
                status=self.status,
                latency_ms=0.0,
                disclaimer=MANDATORY_NEURAL_DISCLAIMER,
            )

        # Validate input strictly (raises TypeError/ValueError if invalid)
        self.model.validate_input_tensor(x)

        start_time = time.perf_counter()
        with torch.no_grad():
            logits_tensor = self.model(x)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract first sample from batch
        sample_logits = logits_tensor[0].cpu().numpy().tolist()
        probs_tensor = torch.softmax(logits_tensor[0], dim=-1)
        sample_probs = probs_tensor.cpu().numpy().tolist()

        # Map per-class uncalibrated scores
        uncalibrated_scores: Dict[str, float] = {}
        for idx, score in enumerate(sample_probs):
            class_name = self._class_mapping.get(idx, f"CLASS_{idx}")
            uncalibrated_scores[class_name] = float(score)

        predicted_idx = int(torch.argmax(logits_tensor[0]).item())
        predicted_class = self._class_mapping.get(predicted_idx, f"CLASS_{predicted_idx}")

        return ModelInferenceResult(
            raw_logits=[float(v) for v in sample_logits],
            uncalibrated_scores=uncalibrated_scores,
            predicted_class=predicted_class,
            model_version=self._model_version,
            architecture=self._architecture_name,
            input_config=self._input_config,
            status=InferenceStatus.READY,
            latency_ms=round(elapsed_ms, 3),
            disclaimer=MANDATORY_NEURAL_DISCLAIMER,
        )

    def predict_batch(self, x: torch.Tensor) -> BatchInferenceResult:
        """Execute batched inference on preprocessed tensor of shape (B, 3, 224, 224).

        Args:
            x: Batched tensor of shape (B, 3, 224, 224).

        Returns:
            BatchInferenceResult payload containing per-sample results.
        """
        if self.status != InferenceStatus.READY or self.model is None:
            batch_size = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim >= 1 else 0
            results = [
                ModelInferenceResult(
                    raw_logits=[],
                    uncalibrated_scores={},
                    predicted_class="UNKNOWN",
                    model_version=self._model_version,
                    architecture=self._architecture_name,
                    input_config=self._input_config,
                    status=self.status,
                    latency_ms=0.0,
                    disclaimer=MANDATORY_NEURAL_DISCLAIMER,
                )
                for _ in range(batch_size)
            ]
            return BatchInferenceResult(
                results=results,
                batch_size=batch_size,
                total_latency_ms=0.0,
                average_latency_ms=0.0,
            )

        self.model.validate_input_tensor(x)
        batch_size = x.shape[0]

        start_time = time.perf_counter()
        with torch.no_grad():
            logits_tensor = self.model(x)
            probs_tensor = torch.softmax(logits_tensor, dim=-1)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        avg_latency_ms = total_latency_ms / max(1, batch_size)

        results: List[ModelInferenceResult] = []
        for i in range(batch_size):
            sample_logits = logits_tensor[i].cpu().numpy().tolist()
            sample_probs = probs_tensor[i].cpu().numpy().tolist()

            uncalibrated_scores: Dict[str, float] = {}
            for idx, score in enumerate(sample_probs):
                class_name = self._class_mapping.get(idx, f"CLASS_{idx}")
                uncalibrated_scores[class_name] = float(score)

            predicted_idx = int(torch.argmax(logits_tensor[i]).item())
            predicted_class = self._class_mapping.get(predicted_idx, f"CLASS_{predicted_idx}")

            results.append(
                ModelInferenceResult(
                    raw_logits=[float(v) for v in sample_logits],
                    uncalibrated_scores=uncalibrated_scores,
                    predicted_class=predicted_class,
                    model_version=self._model_version,
                    architecture=self._architecture_name,
                    input_config=self._input_config,
                    status=InferenceStatus.READY,
                    latency_ms=round(avg_latency_ms, 3),
                    disclaimer=MANDATORY_NEURAL_DISCLAIMER,
                )
            )

        return BatchInferenceResult(
            results=results,
            batch_size=batch_size,
            total_latency_ms=round(total_latency_ms, 3),
            average_latency_ms=round(avg_latency_ms, 3),
        )
