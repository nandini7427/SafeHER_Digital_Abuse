"""Model runner abstractions and inference engines for SafeHER Image Intelligence.

Provides lazy-loaded, thread-safe execution, deterministic baseline heuristics (Error Level Analysis),
and graceful neural model integration without automatic startup downloads.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import io
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageChops

from ai_core.image_intelligence.config import ImageDetectorConfig, default_image_config
from ai_core.image_intelligence.preprocessing import PreprocessedImage
from ai_core.image_intelligence.schemas import (
    DetectionTask,
    ManipulationRegion,
    ManipulationRegionType,
)

logger = logging.getLogger("safeher.image_intelligence.runner")

# Permitted model weight serialization extensions (prevents arbitrary pickle execution CWE-502)
ALLOWED_MODEL_EXTENSIONS: Set[str] = {".safetensors", ".pt", ".pth", ".bin"}


@dataclass(frozen=True)
class ModelPrediction:
    """Standardized internal prediction container from a model runner."""

    raw_score: float
    manipulation_regions: List[ManipulationRegion]
    auxiliary_metrics: Dict[str, Any]
    runner_name: str
    runner_version: str
    is_neural: bool = False


class BaseImageModelRunner(ABC):
    """Abstract base class for image intelligence model execution engines."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if model weights and runtime dependencies are ready for inference."""
        pass

    @abstractmethod
    def get_status(self) -> str:
        """Return human-readable availability or error status."""
        pass

    @abstractmethod
    def predict(self, preprocessed: PreprocessedImage) -> ModelPrediction:
        """Execute inference against a preprocessed image."""
        pass


class BaselineHeuristicRunner(BaseImageModelRunner):
    """Deterministic, zero-download baseline forensic engine using Error Level Analysis (ELA).

    Analyzes pixel-level compression discrepancies when resaved at known JPEG quality tiers.
    Spliced or synthetically generated regions typically exhibit higher error variance than
    organically captured camera backgrounds.
    """

    def __init__(self, config: Optional[ImageDetectorConfig] = None) -> None:
        self.config = config or default_image_config

    def is_available(self) -> bool:
        return True

    def get_status(self) -> str:
        return "READY: Deterministic ELA and frequency anomaly baseline active."

    def _compute_ela(self, image: Image.Image) -> Tuple[np.ndarray, float, float]:
        """Compute Error Level Analysis array and global statistics."""
        # Resave working copy to an in-memory buffer at fixed JPEG quality
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=self.config.ela_quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        # Compute absolute difference
        diff = ImageChops.difference(image, recompressed)
        diff_arr = np.array(diff, dtype=np.float32)

        mean_diff = float(np.mean(diff_arr))
        var_diff = float(np.var(diff_arr))
        return diff_arr, mean_diff, var_diff

    def _analyze_spatial_grid(
        self,
        diff_arr: np.ndarray,
        global_mean: float,
        global_var: float,
        grid_size: int = 4,
    ) -> List[ManipulationRegion]:
        """Partition image into a grid and identify localized compression outliers."""
        height, width, _ = diff_arr.shape
        regions: List[ManipulationRegion] = []

        cell_h = height // grid_size
        cell_w = width // grid_size

        if cell_h == 0 or cell_w == 0:
            return regions

        cell_stats: List[Tuple[int, int, float, float]] = []
        for r in range(grid_size):
            for c in range(grid_size):
                sub_arr = diff_arr[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                local_mean = float(np.mean(sub_arr))
                local_var = float(np.var(sub_arr))
                cell_stats.append((r, c, local_mean, local_var))

        # Flag outlier cells where variance exceeds 1.5x the global variance and local_mean is elevated
        var_threshold = max(global_var * 1.5, 4.0)
        for r, c, local_mean, local_var in cell_stats:
            if local_var > var_threshold and local_mean > global_mean * 1.3:
                ymin = round(r / grid_size, 3)
                xmin = round(c / grid_size, 3)
                ymax = round((r + 1) / grid_size, 3)
                xmax = round((c + 1) / grid_size, 3)

                # Normalized anomaly ratio in [0.0, 1.0]
                anomaly_score = min(round((local_var / (var_threshold * 2.0)), 3), 1.0)

                regions.append(
                    ManipulationRegion(
                        box_normalized=(ymin, xmin, ymax, xmax),
                        anomaly_score=max(0.1, anomaly_score),
                        region_type=ManipulationRegionType.GRID_CELL,
                        description=(
                            f"Compression error variance outlier ({local_var:.1f} vs global {global_var:.1f}) "
                            f"in grid quadrant ({r+1},{c+1})."
                        ),
                    )
                )

        return regions

    def predict(self, preprocessed: PreprocessedImage) -> ModelPrediction:
        """Execute heuristic forensic scan on the preprocessed working image."""
        working_img = preprocessed.working_image

        diff_arr, mean_diff, var_diff = self._compute_ela(working_img)
        regions = self._analyze_spatial_grid(diff_arr, mean_diff, var_diff)

        # Baseline score calculation:
        # Organic, untouched camera images typically show low uniform ELA error (mean_diff < 4.0, var < 30.0).
        # Spliced or AI-manipulated composite images typically produce higher variance or multiple outlier cells.
        base_score = 0.10
        if mean_diff > 8.0:
            base_score += 0.25
        elif mean_diff > 4.5:
            base_score += 0.15

        if var_diff > 80.0:
            base_score += 0.35
        elif var_diff > 40.0:
            base_score += 0.20

        # Region penalty
        if regions:
            region_boost = min(len(regions) * 0.10, 0.30)
            base_score += region_boost

        raw_score = round(min(max(base_score, 0.05), 0.95), 3)

        auxiliary_metrics = {
            "ela_mean_diff": round(mean_diff, 2),
            "ela_variance": round(var_diff, 2),
            "flagged_quadrants_count": len(regions),
            "analysis_dimensions": f"{working_img.width}x{working_img.height}",
        }

        return ModelPrediction(
            raw_score=raw_score,
            manipulation_regions=regions,
            auxiliary_metrics=auxiliary_metrics,
            runner_name="SafeHER-Baseline-ELA-Heuristics",
            runner_version="1.0.0",
            is_neural=False,
        )


class NeuralVisionModelRunner(BaseImageModelRunner):
    """Neural vision model runner with lazy-loading, thread-capping, and graceful fallback.

    Operates strictly from pre-staged local weights without automatic download.
    """

    def __init__(self, config: Optional[ImageDetectorConfig] = None) -> None:
        self.config = config or default_image_config
        self._model = None
        self._weights = None
        self._lock = threading.Lock()
        self._load_attempted = False
        self._error_status: Optional[str] = None
        self._computed_sha256: Optional[str] = None

    def is_available(self) -> bool:
        if not self.config.enable_neural_detector:
            return False
        if self._error_status is not None:
            return False
        self._ensure_loaded()
        return self._model is not None

    def get_status(self) -> str:
        if not self.config.enable_neural_detector:
            return "DISABLED: Neural vision detector is disabled by configuration."
        if self._error_status is not None:
            return f"UNAVAILABLE: {self._error_status}"
        if self._model is not None:
            return "READY: Neural vision detector initialized on CPU."
        return "STANDBY: Neural vision detector will lazy-load on first invocation."

    def _ensure_loaded(self) -> None:
        if self._load_attempted or not self.config.enable_neural_detector:
            return

        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True

            # Verify hardware device support
            target_device = self.config.device
            if target_device.startswith("cuda"):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        self._error_status = f"Unsupported device: CUDA device '{target_device}' requested but torch.cuda.is_available() is False on this system."
                        logger.warning("Neural runner unavailable: %s. Using baseline fallback.", self._error_status)
                        return
                except ImportError:
                    self._error_status = "Unsupported environment: PyTorch is not installed to support CUDA acceleration."
                    logger.warning("Neural runner unavailable: %s. Using baseline fallback.", self._error_status)
                    return
            elif target_device == "mps":
                try:
                    import torch
                    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                        self._error_status = "Unsupported device: Apple Metal Performance Shaders (MPS) requested but unavailable on this system."
                        logger.warning("Neural runner unavailable: %s. Using baseline fallback.", self._error_status)
                        return
                except ImportError:
                    self._error_status = "PyTorch is not installed to check MPS availability."
                    return

            # Verify local model path
            model_path = self.config.model_name_or_path
            if not model_path:
                self._error_status = "No model_name_or_path configured."
                logger.info("Neural runner skipped: %s", self._error_status)
                return

            if not os.path.exists(model_path):
                self._error_status = f"Local model weights not found at path: {model_path}"
                logger.warning("Neural runner unavailable: %s. Using baseline fallback.", self._error_status)
                return

            # 1. Format verification (CWE-502 Deserialization Defense)
            ext = os.path.splitext(model_path)[1].lower()
            if ext not in ALLOWED_MODEL_EXTENSIONS:
                self._error_status = (
                    f"Untrusted model serialization format '{ext}' rejected. "
                    f"Allowed safe formats: {', '.join(sorted(ALLOWED_MODEL_EXTENSIONS))} (CWE-502 Defense)."
                )
                logger.critical("Supply-chain security violation: %s", self._error_status)
                return

            # 2. Cryptographic SHA-256 hash calculation & integrity audit
            try:
                hasher = hashlib.sha256()
                with open(model_path, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        hasher.update(chunk)
                self._computed_sha256 = hasher.hexdigest().lower()

                if self.config.expected_model_sha256:
                    expected = self.config.expected_model_sha256.lower()
                    if self._computed_sha256 != expected:
                        self._error_status = (
                            f"Model artifact cryptographic hash mismatch! Expected: {expected}, "
                            f"Computed: {self._computed_sha256}. Possible supply-chain tampering or corruption."
                        )
                        logger.critical("Supply-chain security violation: %s", self._error_status)
                        return
            except Exception as exc:
                self._error_status = f"Failed to compute model weight checksum: {str(exc)}"
                logger.error("Model hash computation failure: %s", self._error_status)
                return

            if self.config.is_development_placeholder:
                self._error_status = (
                    "Development placeholder mode is enabled; neural inference is intentionally unavailable."
                )
                logger.info("Neural runner remains disabled: %s", self._error_status)
                return

            # 3. Safe weight loading (prefer safetensors; enforce weights_only=True for torch)
            try:
                if ext == ".safetensors":
                    try:
                        from safetensors.torch import load_file
                        checkpoint = load_file(model_path, device="cpu")
                    except ImportError:
                        raise ImportError("safetensors package required to load .safetensors model weights.")
                else:
                    import torch
                    torch.set_num_threads(self.config.num_threads)
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

                # A checkpoint/state dictionary is not itself an executable model.
                # This repository does not yet contain a registered model architecture
                # or inference adapter; reporting READY or emitting a constant score
                # here would falsely imply that neural inference occurred. Keep the
                # verified tensors for future Milestone 4 integration, but fail closed.
                self._weights = checkpoint
                self._model = None
                self._error_status = (
                    "Weights passed integrity checks, but no neural model architecture or "
                    "inference adapter is registered. Neural inference remains unavailable."
                )
                logger.warning(
                    "Neural weights verified but no inference adapter is registered: %s (SHA-256: %s...)",
                    model_path,
                    self._computed_sha256[:12] if self._computed_sha256 else "none",
                )
            except Exception as exc:
                self._error_status = f"Failed to load neural model weights: {str(exc)}"
                logger.error("Neural vision detector initialization error: %s", self._error_status)
                self._model = None

    def predict(self, preprocessed: PreprocessedImage) -> ModelPrediction:
        """Run neural inference if available, otherwise raise RuntimeError."""
        if not self.is_available():
            raise RuntimeError(f"Neural vision model runner is not available: {self.get_status()}")

        # A verified state dictionary without a registered architecture cannot
        # produce a valid prediction. Keep this guard even if a future refactor
        # accidentally makes is_available() return True too early.
        raise RuntimeError(
            "Neural inference cannot run because no model architecture/inference adapter is registered."
        )


class FrequencySpectralRunner(BaseImageModelRunner):
    """Deterministic, zero-download frequency-domain forensic engine using 2D FFT spectral analysis.

    Evaluates 2D Fourier power spectra for azimuthal decay anomalies and periodic lattice peaks
    characteristic of generative upsampling (GAN / diffusion transposed convolutions and VAE decoders).
    Natural camera imagery adheres to power-law decay P(f) ~ 1/f^alpha (alpha ~ 2.0) without discrete
    periodic spectral peaks.
    """

    def __init__(self, config: Optional[ImageDetectorConfig] = None) -> None:
        self.config = config or default_image_config

    def is_available(self) -> bool:
        return True

    def get_status(self) -> str:
        return "READY: Deterministic 2D-FFT azimuthal power decay and lattice peak detector active."

    def _compute_spectral_features(self, image: Image.Image) -> Dict[str, Any]:
        """Compute 2D FFT spectral decay slope, high-frequency energy ratio, and lattice peaks."""
        gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
        h, w = gray.shape

        if h < 16 or w < 16:
            return {
                "spectral_slope": -2.0,
                "high_freq_ratio": 0.05,
                "periodic_peaks_count": 0,
                "spectral_entropy": 0.0,
            }

        # Mitigate boundary discontinuity leakage using 2D Hanning window
        wy = np.hanning(h)
        wx = np.hanning(w)
        w2d = np.outer(wy, wx)
        windowed = (gray - np.mean(gray)) * w2d

        # Compute centered 2D FFT & power spectrum
        fft2 = np.fft.fft2(windowed)
        fft_shifted = np.fft.fftshift(fft2)
        power = np.abs(fft_shifted) ** 2
        log_power = np.log1p(power)

        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        R = np.hypot(Y - cy, X - cx)
        r_max = min(cy, cx)

        if r_max <= 4:
            return {
                "spectral_slope": -2.0,
                "high_freq_ratio": 0.05,
                "periodic_peaks_count": 0,
                "spectral_entropy": 0.0,
            }

        # Radial binning
        num_bins = min(32, int(r_max))
        bin_edges = np.linspace(2, r_max, num_bins + 1)
        bin_radii = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        radial_profile = []

        for r_low, r_high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (R >= r_low) & (R < r_high)
            if np.any(mask):
                radial_profile.append(float(np.mean(power[mask])))
            else:
                radial_profile.append(1e-12)

        radial_arr = np.array(radial_profile, dtype=np.float32)
        radial_arr = np.maximum(radial_arr, 1e-12)

        # Estimate power-law decay slope log(P) vs log(r) over first 75% band
        log_r = np.log(bin_radii)
        log_p = np.log(radial_arr)
        cutoff_idx = max(3, int(num_bins * 0.75))
        valid_idx = np.isfinite(log_r[:cutoff_idx]) & np.isfinite(log_p[:cutoff_idx])
        if np.sum(valid_idx) >= 3:
            slope, _ = np.polyfit(log_r[:cutoff_idx][valid_idx], log_p[:cutoff_idx][valid_idx], 1)
        else:
            slope = -2.0

        # High-frequency energy ratio (r >= 0.5 * r_max vs total energy)
        total_energy = float(np.sum(power[R >= 2]))
        high_freq_energy = float(np.sum(power[R >= (0.5 * r_max)]))
        high_freq_ratio = float(high_freq_energy / max(total_energy, 1e-12))

        # Periodic lattice peak detection (upsampling artifacts in frequency space)
        residual = np.zeros_like(log_power)
        for r_low, r_high, mean_val in zip(bin_edges[:-1], bin_edges[1:], radial_profile):
            mask = (R >= r_low) & (R < r_high)
            if np.any(mask):
                log_mean = np.log1p(mean_val)
                residual[mask] = log_power[mask] - log_mean

        # Search outer annulus where lattice peaks from 2x/4x upsamplers typically reside
        annulus_mask = (R >= 0.15 * r_max) & (R <= 0.90 * r_max)
        if np.any(annulus_mask):
            annulus_res = residual[annulus_mask]
            res_mean = float(np.mean(annulus_res))
            res_std = float(np.std(annulus_res))
            peak_threshold = res_mean + max(3.5 * res_std, 1.5)
            peak_pixels = int(np.sum((residual >= peak_threshold) & annulus_mask))
            periodic_peaks_count = min(peak_pixels // 6, 10)
        else:
            periodic_peaks_count = 0

        # Spectral entropy
        prob = radial_arr / np.sum(radial_arr)
        prob = prob[prob > 0]
        spectral_entropy = float(-np.sum(prob * np.log2(prob)))

        return {
            "spectral_slope": round(float(slope), 3),
            "high_freq_ratio": round(float(high_freq_ratio), 4),
            "periodic_peaks_count": int(periodic_peaks_count),
            "spectral_entropy": round(float(spectral_entropy), 3),
        }

    def predict(self, preprocessed: PreprocessedImage) -> ModelPrediction:
        """Execute frequency-domain forensic scan on the preprocessed working image."""
        working_img = preprocessed.working_image
        features = self._compute_spectral_features(working_img)

        spectral_slope = features["spectral_slope"]
        high_freq_ratio = features["high_freq_ratio"]
        periodic_peaks_count = features["periodic_peaks_count"]

        # Base score
        base_score = 0.10

        # Periodic lattice peak contribution
        if periodic_peaks_count >= 2:
            base_score += 0.35
        elif periodic_peaks_count == 1:
            base_score += 0.18

        # Spectral slope anomaly: natural photographic decay is ~ -2.0
        slope_deviation = abs(spectral_slope - (-2.0))
        if slope_deviation > 0.85:
            base_score += 0.25
        elif slope_deviation > 0.50:
            base_score += 0.15

        # High-frequency ratio anomaly (unnatural high-frequency energy retention)
        if high_freq_ratio > 0.40:
            base_score += 0.20
        elif high_freq_ratio > 0.28:
            base_score += 0.10

        raw_score = round(min(max(base_score, 0.05), 0.95), 3)

        regions: List[ManipulationRegion] = []
        if periodic_peaks_count >= 1:
            regions.append(
                ManipulationRegion(
                    box_normalized=(0.0, 0.0, 1.0, 1.0),
                    anomaly_score=min(0.20 + periodic_peaks_count * 0.15, 0.90),
                    region_type=ManipulationRegionType.BOUNDING_BOX,
                    description=(
                        f"Detected {periodic_peaks_count} discrete periodic spectral peak cluster(s) "
                        f"in 2D Fourier power spectrum (characteristic of neural upsampling / deconvolution)."
                    ),
                )
            )

        auxiliary_metrics = {
            "spectral_slope": spectral_slope,
            "high_freq_ratio": high_freq_ratio,
            "periodic_peaks_count": periodic_peaks_count,
            "spectral_entropy": features["spectral_entropy"],
            "analysis_dimensions": f"{working_img.width}x{working_img.height}",
        }

        return ModelPrediction(
            raw_score=raw_score,
            manipulation_regions=regions,
            auxiliary_metrics=auxiliary_metrics,
            runner_name="SafeHER-Frequency-Spectral-Detector",
            runner_version="1.0.0",
            is_neural=False,
        )
