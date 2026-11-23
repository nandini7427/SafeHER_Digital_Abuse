
import hashlib
import uuid
from pathlib import Path

from fastapi import UploadFile

from backend.app.config import settings


CHUNK_SIZE = 1024 * 1024  # 1 MB

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
