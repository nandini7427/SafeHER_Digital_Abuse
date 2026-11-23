"""Text preprocessing and sanitization pipeline for SafeHER threat assessment.

Processes working copies of text while preserving the original evidence unmodified.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List


# Zero-width and invisible control characters commonly used to evade keyword filters
ZERO_WIDTH_CHARS_PATTERN = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF\u202A-\u202E]")

# Non-printable control characters excluding newline (\n) and tab (\t)
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Multiple consecutive spaces or tabs
MULTI_SPACE_PATTERN = re.compile(r"[^\S\r\n]+")

# Multiple consecutive newlines (limit to max 2)
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class PreprocessedText:
    """Immutable preprocessed text container maintaining evidence isolation.

    Attributes:
        original_text: The pristine, unaltered input string (evidence record).
        working_copy: Sanitized and normalized text for model inference.
        char_count: Character count of the working copy.
        word_count: Approximate word count of the working copy.
        is_truncated: Whether input was truncated due to length limits.
        warnings: List of preprocessing alerts (e.g. control chars removed).
    """

    original_text: str
    working_copy: str
    char_count: int
    word_count: int
    is_truncated: bool = False
    warnings: List[str] = field(default_factory=list)


def preprocess_text(
    text: str,
    max_length: int = 10000,
    min_length: int = 1,
) -> PreprocessedText:
    """Preprocess and normalize text for threat assessment inference.

    Args:
        text: Raw text string to process.
        max_length: Maximum allowed character length for working copy.
        min_length: Minimum required character length.

    Returns:
        PreprocessedText container with separate original and working copy.

    Raises:
        ValueError: If input is not a string, or is empty / whitespace-only.
    """
    if text is None or not isinstance(text, str):
        raise ValueError("Input text must be a valid non-null string.")

    # Validate non-empty after stripping whitespace
    stripped = text.strip()
    if len(stripped) < min_length:
        raise ValueError("Input text cannot be empty or whitespace only.")

    warnings: List[str] = []

    # 1. Unicode normalization (NFKC ensures standard canonical compatibility)
    working = unicodedata.normalize("NFKC", text)

    # 2. Check and remove zero-width characters
    if ZERO_WIDTH_CHARS_PATTERN.search(working):
        working = ZERO_WIDTH_CHARS_PATTERN.sub("", working)
        warnings.append("Removed invisible zero-width characters.")

    # 3. Check and remove non-printable control characters
    if CONTROL_CHARS_PATTERN.search(working):
        working = CONTROL_CHARS_PATTERN.sub("", working)
        warnings.append("Removed non-printable control characters.")

    # 4. Normalize spaces and multiple newlines
    working = MULTI_SPACE_PATTERN.sub(" ", working)
    working = MULTI_NEWLINE_PATTERN.sub("\n\n", working)
    working = working.strip()

    # Re-validate that normalization didn't reduce text to empty
    if not working:
        raise ValueError("Input text contains only non-printable or invisible characters.")

    # 5. Enforce input length limit
    is_truncated = False
    if len(working) > max_length:
        working = working[:max_length].rstrip()
        is_truncated = True
        warnings.append(f"Input text exceeded {max_length} characters and was truncated.")

    # Word count estimation
    words = working.split()
    word_count = len(words)

    return PreprocessedText(
        original_text=text,
        working_copy=working,
        char_count=len(working),
        word_count=word_count,
        is_truncated=is_truncated,
        warnings=warnings,
    )
