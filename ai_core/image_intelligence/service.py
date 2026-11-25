"""Service layer managing SafeHER Image Intelligence lifecycles, execution, and privacy safeguards."""

import logging
import threading
import time
from typing import Optional
from uuid import UUID

from ai_core.image_intelligence.config import ImageDetectorConfig, default_image_config
from ai_core.image_intelligence.detector import BaseImageDetector, SafeHERImageDetector
from ai_core.image_intelligence.schemas import ImageAssessmentResponse

logger = logging.getLogger("safeher.image_intelligence.service")

_detector_instance: Optional[BaseImageDetector] = None
_detector_lock = threading.Lock()
_initialization_count: int = 0
_initialization_time_ms: float = 0.0


def get_detector_initialization_count() -> int:
    """Return the total number of times the detector was initialized."""
    with _detector_lock:
        return _initialization_count


def get_detector_initialization_time_ms() -> float:
    """Return the elapsed time in milliseconds for the first detector initialization."""
    with _detector_lock:
        return _initialization_time_ms


def reset_detector_initialization_stats() -> None:
    """Reset the initialization counter and timing statistics for benchmarking."""
    global _initialization_count, _initialization_time_ms
    with _detector_lock:
        _initialization_count = 0
        _initialization_time_ms = 0.0


def get_image_detector(config: Optional[ImageDetectorConfig] = None) -> BaseImageDetector:
    """Return a thread-safe singleton image detector instance configured by settings."""
    global _detector_instance, _initialization_count, _initialization_time_ms
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                t0 = time.perf_counter()
                cfg = config or default_image_config
                logger.info("Initializing SafeHERImageDetector singleton.")
                _detector_instance = SafeHERImageDetector(config=cfg)
                _initialization_count += 1
                _initialization_time_ms = (time.perf_counter() - t0) * 1000.0
    return _detector_instance


def set_image_detector(detector: Optional[BaseImageDetector]) -> None:
    """Override or reset the singleton image detector instance (primarily for testing)."""
    global _detector_instance
    with _detector_lock:
        _detector_instance = detector


class ImageIntelligenceService:
    """Service layer managing visual manipulation triage with privacy safeguards."""

    def __init__(self, detector: Optional[BaseImageDetector] = None) -> None:
        self._detector = detector

    @property
    def detector(self) -> BaseImageDetector:
        if self._detector is not None:
            return self._detector
        return get_image_detector()

    def assess_image(
        self,
        data: bytes,
        image_id: Optional[UUID] = None,
    ) -> ImageAssessmentResponse:
        """Execute visual manipulation assessment on raw image bytes.

        Args:
            data: Raw input image bytes from upload.
            image_id: Optional UUID of the associated evidence record.

        Returns:
            Structured ImageAssessmentResponse.

        Raises:
            ValueError: For input preprocessing or validation failures.
            RuntimeError: For unexpected internal execution errors.
        """
        start_time = time.perf_counter()

        try:
            response = self.detector.detect(data=data, image_id=image_id)
        except ValueError as exc:
            # Rejection due to format, dimension, or corrupt bytes
            logger.warning(
                "Image assessment validation rejected: reason=%s, byte_size=%d",
                str(exc),
                len(data) if data else 0,
            )
            raise
        except Exception as exc:
            # Never log raw image content or sensitive evidence
            logger.exception("Unexpected error during image assessment.")
            raise RuntimeError(f"Internal error during image assessment: {str(exc)}") from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Privacy-safe telemetry logging: log ONLY non-sensitive metrics
        logger.info(
            "Image assessment completed: assessment_id=%s, label=%s, raw_score=%.2f, regions_count=%d, elapsed_ms=%.2f",
            response.assessment_id,
            response.assessment_label.value,
            response.raw_model_score,
            len(response.manipulation_regions),
            elapsed_ms,
        )

        return response


# Default service instance for convenience
image_service = ImageIntelligenceService()
