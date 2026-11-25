"""Secure, evidence-preserving image preprocessing pipeline for SafeHER.

Strictly preserves original evidence bytes without mutation, validates magic bytes,
rejects decompression bombs and corrupt files, sanitizes EXIF metadata, normalizes
orientation, and provides isolated working copies for analysis.
"""

from dataclasses import dataclass, field
import logging
import hashlib
import io
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import ExifTags, Image, ImageOps

# Pillow's TIFF plugin can log EXIF tag values (including names, GPS and serial
# data) at DEBUG level. Suppress metadata-bearing logs in the ingestion process.
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)

from ai_core.image_intelligence.config import ImageDetectorConfig, default_image_config
from ai_core.image_intelligence.schemas import ImageMetadataSummary


# Known image binary signatures (magic bytes)
MAGIC_SIGNATURES: Dict[str, bytes] = {
    "JPEG": b"\xFF\xD8\xFF",
    "PNG": b"\x89PNG\r\n\x1a\n",
    "WEBP": b"RIFF",  # WEBP has RIFF at offset 0 and WEBP at offset 8
    "BMP": b"BM",
    "TIFF_LE": b"II*\x00",
    "TIFF_BE": b"MM\x00*",
}

# EXIF tags to strictly strip for survivor privacy (PII, geolocation, hardware serials, user metadata)
SENSITIVE_EXIF_TAGS: Set[str] = {
    "GPSInfo",
    "MakerNote",
    "UserComment",
    "DeviceSettingDescription",
    "CameraSerialNumber",
    "OwnerName",
    "CameraOwnerName",
    "BodySerialNumber",
    "LensSerialNumber",
    "SerialNumber",
    "ImageUniqueID",
    "Artist",
    "Copyright",
    "HostComputer",
    "XPAuthor",
    "XPComment",
    "XPKeywords",
    "XPSubject",
    "XPTitle",
    "UniqueCameraModel",
}


@dataclass(frozen=True)
class PreprocessedImage:
    """Immutable container separating pristine evidence from the working copy."""

    raw_bytes: bytes
    sha256: str
    working_image: Image.Image
    model_input_image: Image.Image
    metadata: ImageMetadataSummary
    warnings: List[str] = field(default_factory=list)
    is_downscaled: bool = False


def identify_magic_format(data: bytes) -> Optional[str]:
    """Inspect the first bytes of data to identify valid image formats."""
    if len(data) < 12:
        return None

    if data.startswith(MAGIC_SIGNATURES["JPEG"]):
        return "JPEG"
    if data.startswith(MAGIC_SIGNATURES["PNG"]):
        return "PNG"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "WEBP"
    if data.startswith(MAGIC_SIGNATURES["BMP"]):
        return "BMP"
    if data.startswith(MAGIC_SIGNATURES["TIFF_LE"]) or data.startswith(MAGIC_SIGNATURES["TIFF_BE"]):
        return "TIFF"

    return None


def _sanitize_exif(img: Image.Image) -> Tuple[bool, Dict[str, Any]]:
    """Extract non-sensitive technical EXIF metadata and strip PII/GPS."""
    has_exif = False
    sanitized: Dict[str, Any] = {}

    try:
        exif = img.getexif()
        if not exif:
            return False, sanitized

        has_exif = True
        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name in SENSITIVE_EXIF_TAGS or "gps" in tag_name.lower():
                continue

            # Convert binary or non-serializable values to safe representations
            if isinstance(value, bytes):
                try:
                    sanitized[tag_name] = value.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue
            elif isinstance(value, (int, float, str, bool)):
                sanitized[tag_name] = value
            elif isinstance(value, tuple) and len(value) <= 4:
                sanitized[tag_name] = list(value)
    except Exception:
        # If EXIF reading fails, treat as no EXIF without crashing
        return False, {}

    return has_exif, sanitized


def preprocess_image(
    data: bytes,
    config: Optional[ImageDetectorConfig] = None,
    target_model_size: Tuple[int, int] = (256, 256),
) -> PreprocessedImage:
    """Validate, sanitize, and normalize image bytes into an immutable PreprocessedImage.

    Args:
        data: Raw image bytes directly from upload.
        config: Configuration containing dimension and size limits.
        target_model_size: Dimensions for the scaled model input tensor copy.

    Returns:
        PreprocessedImage with original bytes, SHA-256, working copy, and metadata.

    Raises:
        ValueError: For empty bytes, oversized payloads, unrecognized formats,
                    corrupted images, or dimension violations.
    """
    cfg = config or default_image_config

    # 1. Validation of raw payload existence and size
    if not data:
        raise ValueError("Image data cannot be empty.")

    if len(data) > cfg.max_file_size_bytes:
        raise ValueError(
            f"Image payload size ({len(data)} bytes) exceeds the maximum limit of {cfg.max_file_size_bytes} bytes."
        )

    # 2. Cryptographic evidence hash calculation
    sha256_hash = hashlib.sha256(data).hexdigest()

    # 3. Magic bytes format validation
    detected_format = identify_magic_format(data)
    if detected_format is None:
        raise ValueError(
            "Unsupported or invalid image file format. Supported formats: JPEG, PNG, WEBP, BMP, TIFF."
        )

    # 4. Decompression bomb protection: align maximum allowed pixels with max dimension square
    max_pixels = cfg.max_image_dimension * cfg.max_image_dimension
    Image.MAX_IMAGE_PIXELS = max_pixels

    # 5. Image decoding, dimension pre-check, and corrupt-image verification
    try:
        buffer = io.BytesIO(data)
        # Inspect header and dimensions BEFORE load to prevent memory allocation exhaustion (CWE-400)
        with Image.open(buffer) as raw_img:
            width, height = raw_img.size
            if width < cfg.min_image_dimension or height < cfg.min_image_dimension:
                raise ValueError(
                    f"Image dimensions ({width}x{height}) are smaller than the required minimum of {cfg.min_image_dimension}x{cfg.min_image_dimension}."
                )
            if width > cfg.max_image_dimension or height > cfg.max_image_dimension:
                raise ValueError(
                    f"Image dimensions ({width}x{height}) exceed the maximum allowable limit of {cfg.max_image_dimension}x{cfg.max_image_dimension}."
                )
            raw_img.verify()

        # Re-open for actual processing (verify closes stream state)
        buffer.seek(0)
        img = Image.open(buffer)
        img.load()  # Force decode to ensure file is not truncated or corrupted
    except Image.DecompressionBombError as exc:
        raise ValueError(
            f"Image exceeds maximum allowable pixel decompression limit (decompression bomb protection): {str(exc)}"
        ) from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Corrupt or unreadable image stream: {str(exc)}") from exc

    warnings: List[str] = []
    width, height = img.size

    # 7. Metadata extraction and PII stripping
    has_exif, sanitized_exif = _sanitize_exif(img)

    # 8. EXIF orientation handling (transpose working copy)
    try:
        oriented_img = ImageOps.exif_transpose(img)
        if oriented_img is None:
            oriented_img = img
    except Exception:
        oriented_img = img
        warnings.append("EXIF orientation transpose could not be computed; using standard raster.")

    # 9. RGB color conversion (working copy isolation)
    original_color_mode = oriented_img.mode
    if original_color_mode != "RGB":
        try:
            working_rgb = oriented_img.convert("RGB")
            warnings.append(f"Image color mode converted from {original_color_mode} to standard RGB.")
        except Exception as exc:
            raise ValueError(f"Failed to convert image color mode {original_color_mode} to RGB: {str(exc)}") from exc
    else:
        # Create an independent in-memory copy so original decoded instance is untouched
        working_rgb = oriented_img.copy()

    # 10. Safe model input resizing (bilinear/bicubic standard working copy)
    is_downscaled = False
    if working_rgb.size != target_model_size:
        model_input = working_rgb.resize(target_model_size, resample=Image.Resampling.BILINEAR)
        is_downscaled = (working_rgb.width > target_model_size[0] or working_rgb.height > target_model_size[1])
    else:
        model_input = working_rgb.copy()

    metadata_summary = ImageMetadataSummary(
        format=detected_format,
        dimensions=(width, height),
        size_bytes=len(data),
        sha256=sha256_hash,
        has_exif=has_exif,
        sanitized_exif=sanitized_exif,
        color_mode=original_color_mode,
    )

    return PreprocessedImage(
        raw_bytes=data,
        sha256=sha256_hash,
        working_image=working_rgb,
        model_input_image=model_input,
        metadata=metadata_summary,
        warnings=warnings,
        is_downscaled=is_downscaled,
    )
