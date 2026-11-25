"""File integrity, cryptographic hashing, and path security utilities for SafeHER datasets.

Enforces:
1. Streaming SHA-256 calculation for large files without excessive memory consumption.
2. Cryptographic hash verification against expected manifest digests.
3. File existence, file size limits, and format verification.
4. Safe relative path resolution with strict defense against directory traversal and symlink escapes.
"""

import hashlib
import io
import os
from pathlib import Path
import re
from typing import BinaryIO, Optional, Set, Tuple, Union

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

# Default streaming buffer size: 64 KB
DEFAULT_CHUNK_SIZE = 65536

# Maximum default allowed file size for image assets (50 MB)
DEFAULT_MAX_IMAGE_BYTES = 50 * 1024 * 1024

# Recognized standard forensic image formats
STANDARD_IMAGE_FORMATS: Set[str] = {"JPEG", "JPG", "PNG", "WEBP", "TIFF", "BMP"}

# Approved safe URI schemes for non-local references
APPROVED_DATASET_SCHEMES: Tuple[str, ...] = (
    "safeher://",
    "upstream://",
    "dataset://",
    "faceforensics://",
    "genimage://",
    "openimages://",
    "s3://",
    "gs://",
    "https://",
    "http://",
)


class PathTraversalError(ValueError):
    """Raised when a path traversal, drive escape, or symlink escape attempt is detected."""


class FileIntegrityError(ValueError):
    """Raised when a file fails cryptographic hash verification, existence, or size limits."""


class InvalidFormatError(ValueError):
    """Raised when a file format is unrecognized, invalid, or violates schema restrictions."""


class CorruptedFileError(ValueError):
    """Raised when an image file header or byte stream is truncated, corrupted, or unreadable."""


class ImageFileMetadata(BaseModel):
    """Metadata and integrity verification result for a local image file."""

    path: str = Field(..., description="Canonical or relative path to the image file")
    sha256: str = Field(..., description="Computed 64-character lowercase SHA-256 digest")
    file_size_bytes: int = Field(..., ge=0, description="Size of file in bytes")
    format: str = Field(..., description="Detected image format (e.g. 'JPEG', 'PNG')")
    width: int = Field(..., ge=1, description="Image width in pixels")
    height: int = Field(..., ge=1, description="Image height in pixels")
    mode: str = Field(..., description="Image color mode (e.g. 'RGB', 'RGBA', 'L')")
    is_valid: bool = Field(default=True, description="Whether the file passes integrity and decode checks")
    error_message: Optional[str] = Field(None, description="Integrity or format failure details if invalid")

    model_config = ConfigDict(frozen=True)


def compute_stream_sha256(stream: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Compute SHA-256 hex digest for an open binary stream using chunked streaming.

    Args:
        stream: Readable binary stream.
        chunk_size: Size of chunk read per iteration in bytes.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    hasher = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        hasher.update(chunk)
    return hasher.hexdigest().lower()


def compute_file_sha256(
    file_path: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Compute SHA-256 hex digest for a file using chunked streaming without buffering into RAM.

    Args:
        file_path: Filesystem path to the target file.
        chunk_size: Buffer size for streaming read operations.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path references a directory.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected file, but found directory: {path}")

    with open(path, "rb") as f:
        return compute_stream_sha256(f, chunk_size=chunk_size)


def verify_file_sha256(
    file_path: Union[str, Path],
    expected_hash: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tuple[bool, str]:
    """Verify that a file's on-disk cryptographic digest matches an expected SHA-256 hash.

    Args:
        file_path: Target file path.
        expected_hash: Expected 64-character hexadecimal SHA-256 digest.
        chunk_size: Streaming chunk size.

    Returns:
        Tuple of (is_valid, computed_hash).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    clean_expected = (expected_hash or "").strip().lower()
    computed = compute_file_sha256(file_path, chunk_size=chunk_size)
    is_match = computed == clean_expected
    return is_match, computed


def resolve_safe_relative_path(
    root_dir: Union[str, Path],
    relative_path: str,
    allow_symlinks_within_root: bool = False,
) -> Path:
    """Safely resolve a relative image path within an approved dataset root directory.

    Guarantees:
    - Rejects path traversal sequences ('..', '/../', '\\..\\').
    - Rejects absolute paths (leading '/', '\\', or Windows drive letters 'C:').
    - Rejects null bytes in path strings.
    - Prevents directory traversal attacks via symlinks escaping the root boundary.

    Args:
        root_dir: Canonical root directory under which the asset must reside.
        relative_path: Relative file path provided by dataset record or user request.
        allow_symlinks_within_root: Whether symlinks resolving inside root_dir are permitted.

    Returns:
        Canonical absolute Path within root_dir.

    Raises:
        PathTraversalError: If path attempts to escape the root directory or uses illegal characters.
        FileNotFoundError: If root_dir does not exist.
    """
    if not relative_path or not relative_path.strip():
        raise PathTraversalError("Relative path must not be empty.")

    clean_rel = relative_path.strip()

    # Reject null bytes
    if "\x00" in clean_rel:
        raise PathTraversalError("Path contains prohibited null byte.")

    # Reject Windows drive letters
    if re.match(r"^[a-zA-Z]:", clean_rel):
        raise PathTraversalError(
            f"Absolute drive letter paths are prohibited in relative dataset references: '{clean_rel}'."
        )

    # Reject leading slashes
    if clean_rel.startswith("/") or clean_rel.startswith("\\"):
        raise PathTraversalError(
            f"Absolute root-level paths are prohibited in relative dataset references: '{clean_rel}'."
        )

    # Reject path traversal segments
    normalized_segments = clean_rel.replace("\\", "/").split("/")
    if ".." in normalized_segments:
        raise PathTraversalError(
            f"Path traversal ('..') detected in relative dataset path: '{clean_rel}'."
        )

    canonical_root = Path(root_dir).resolve()
    if not canonical_root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {canonical_root}")

    # Inspect every existing component before resolving the final path. Checking
    # only the final entry misses nested directory symlinks (e.g. root/link/file).
    raw_path = canonical_root
    for segment in normalized_segments:
        raw_path = raw_path / segment
        try:
            is_link = raw_path.is_symlink()
        except OSError as exc:
            raise PathTraversalError(f"Unable to inspect dataset path component: '{segment}'.") from exc
        if is_link:
            if not allow_symlinks_within_root:
                raise PathTraversalError(
                    f"Symlinks are disallowed by default in dataset references: '{clean_rel}'."
                )
            try:
                link_target = raw_path.resolve(strict=False)
                link_target.relative_to(canonical_root)
            except (OSError, ValueError) as exc:
                raise PathTraversalError(
                    f"Symlink in dataset path '{clean_rel}' points outside approved root directory."
                ) from exc

    # Resolve combined path after symlink policy has been enforced.
    try:
        target_path = raw_path.resolve(strict=False)
        target_path.relative_to(canonical_root)
    except (OSError, ValueError) as exc:
        raise PathTraversalError(
            f"Path '{clean_rel}' resolves outside approved root '{canonical_root}'."
        ) from exc

    return target_path


def validate_image_file(
    file_path: Union[str, Path],
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    allowed_formats: Optional[Set[str]] = None,
    verify_pixels: bool = False,
) -> ImageFileMetadata:
    """Validate image file existence, format, headers, decode integrity, and dimensions.

    Args:
        file_path: Path to the target image file.
        max_bytes: Maximum allowed file size in bytes (default 50 MB).
        allowed_formats: Permitted format strings (e.g. {'JPEG', 'PNG'}). Defaults to STANDARD_IMAGE_FORMATS.
        verify_pixels: If True, executes full bitmap decode to detect subtle corruptions.

    Returns:
        ImageFileMetadata with dimensions, format, size, and SHA-256 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        FileIntegrityError: If the file is empty, exceeds max_bytes, or is not a regular file.
        InvalidFormatError: If the format is unrecognized or not in allowed_formats.
        CorruptedFileError: If the file headers or image bytes cannot be parsed by PIL.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise FileIntegrityError(f"Target is not a regular file: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise FileIntegrityError(f"Image file is empty (0 bytes): {path}")
    if file_size > max_bytes:
        raise FileIntegrityError(
            f"Image file size ({file_size} bytes) exceeds safety limit of {max_bytes} bytes: {path}"
        )

    # Compute streaming cryptographic hash
    sha256_hash = compute_file_sha256(path)

    valid_formats = allowed_formats or STANDARD_IMAGE_FORMATS

    try:
        with open(path, "rb") as f:
            # Read first 16 bytes for quick non-image check
            header = f.read(16)
            if not header:
                raise CorruptedFileError(f"File header is empty: {path}")

            f.seek(0)
            try:
                with Image.open(f) as img:
                    img_format = (img.format or "").upper()
                    img_width, img_height = img.size
                    img_mode = img.mode

                    if img_format not in valid_formats and (img_format != "JPG" or "JPEG" not in valid_formats):
                        raise InvalidFormatError(
                            f"Image format '{img_format}' is not in allowed formats ({valid_formats}) for file: {path}"
                        )

                    if img_width <= 0 or img_height <= 0:
                        raise CorruptedFileError(f"Invalid image dimensions ({img_width}x{img_height}) in: {path}")

                    # Header structural verification
                    img.verify()

                # If requested, perform full pixel decompression to catch truncated data
                if verify_pixels:
                    with Image.open(path) as img_full:
                        img_full.load()

            except (UnidentifiedImageError, OSError, SyntaxError) as exc:
                raise CorruptedFileError(f"Corrupted or invalid image byte stream in {path}: {exc}") from exc

    except (InvalidFormatError, CorruptedFileError, FileIntegrityError):
        raise
    except Exception as exc:
        raise CorruptedFileError(f"Unexpected error decoding image file {path}: {exc}") from exc

    return ImageFileMetadata(
        path=str(path),
        sha256=sha256_hash,
        file_size_bytes=file_size,
        format=img_format,
        width=img_width,
        height=img_height,
        mode=img_mode,
        is_valid=True,
        error_message=None,
    )
