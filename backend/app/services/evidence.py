
import hashlib
import uuid
import warnings
from pathlib import Path

import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError

from fastapi import UploadFile

from backend.app.config import settings


CHUNK_SIZE = 1024 * 1024  # 1 MB
MAX_IMAGE_PIXELS = 25_000_000

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".mp4",
    ".webm",
    ".pdf",
    ".txt",
}


def validate_file_signature(header: bytes, extension: str) -> bool:
    """Check whether file content matches its extension."""

    if extension in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")

    if extension == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")

    if extension == ".webp":
        return (
            len(header) >= 12
            and header[:4] == b"RIFF"
            and header[8:12] == b"WEBP"
        )

    if extension == ".mp4":
        return len(header) >= 8 and header[4:8] == b"ftyp"

    if extension == ".webm":
        return header.startswith(b"\x1a\x45\xdf\xa3")

    if extension == ".pdf":
        return header.startswith(b"%PDF-")

    if extension == ".txt":
        try:
            header.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    return False


def generate_perceptual_hash(
    file_path: Path,
    extension: str,
) -> str | None:
    """
    Generate a perceptual hash for supported image evidence.

    Returns:
        A 16-character hexadecimal pHash for images,
        or None for non-image files.
    """

    if extension.lower() not in IMAGE_EXTENSIONS:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(file_path) as image:
                width, height = image.size

                if (
                    width <= 0
                    or height <= 0
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise ValueError(
                        "Image dimensions exceed the allowed limit."
                    )

                image.verify()

            with Image.open(file_path) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                return str(imagehash.phash(image))

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(
            "Unable to process image for perceptual hashing."
        ) from exc


def calculate_hash_distance(
    first_hash: str,
    second_hash: str,
) -> int:
    """Calculate Hamming distance between two 64-bit pHashes."""

    if (
        len(first_hash) != 16
        or len(second_hash) != 16
    ):
        raise ValueError("Invalid perceptual hash format.")

    try:
        first_value = int(first_hash, 16)
        second_value = int(second_hash, 16)
    except ValueError as exc:
        raise ValueError(
            "Invalid perceptual hash format."
        ) from exc

    return (first_value ^ second_value).bit_count()


async def save_evidence_file(
    upload: UploadFile,
) -> tuple[str, int, str]:
    """
    Save an uploaded file privately.

    Returns:
        storage_key, file_size, sha256_hash
    """

    original_filename = upload.filename or ""
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        await upload.close()
        raise ValueError("Unsupported evidence file format.")

    storage_dir = Path(settings.PRIVATE_STORAGE_PATH).resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)

    storage_key = f"{uuid.uuid4().hex}.bin"
    destination = (storage_dir / storage_key).resolve()

    if destination.parent != storage_dir:
        await upload.close()
        raise ValueError("Invalid storage location.")

    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    file_size = 0
    sha256 = hashlib.sha256()
    created = False
    first_chunk = True

    try:
        with destination.open("xb") as evidence_file:
            created = True

            while True:
                chunk = await upload.read(CHUNK_SIZE)

                if not chunk:
                    break

                if first_chunk:
                    if not validate_file_signature(chunk[:16], extension):
                        raise ValueError(
                            "File content does not match its extension."
                        )
                    first_chunk = False

                file_size += len(chunk)

                if file_size > max_size:
                    raise ValueError(
                        f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB upload limit."
                    )

                sha256.update(chunk)
                evidence_file.write(chunk)

        if file_size == 0:
            raise ValueError("Empty files cannot be uploaded.")

        return storage_key, file_size, sha256.hexdigest()

    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise

    finally:
        await upload.close()
