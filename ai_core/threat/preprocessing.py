"""Text preprocessing and sanitization pipeline for SafeHER threat assessment.

The pipeline never mutates original evidence. It returns a separate working copy,
records every transformation that can change its content, rejects isolated Unicode
surrogates, and truncates only at Unicode extended-grapheme boundaries.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

import regex as unicode_regex


ZERO_WIDTH_CHARS_PATTERN = re.compile(r"[\u200B\uFEFF\u202A-\u202E\u2066-\u2069]")
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
MULTI_SPACE_PATTERN = re.compile(r"[^\S\r\n]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
LINE_ENDING_PATTERN = re.compile(r"\r\n?")
GRAPHEME_PATTERN = unicode_regex.compile(r"\X")


@dataclass(frozen=True)
class PreprocessedText:
    """Immutable container separating pristine evidence from the analysis copy."""

    original_text: str
    working_copy: str
    char_count: int
    word_count: int
    is_truncated: bool = False
    warnings: List[str] = field(default_factory=list)


def _validate_lengths(max_length: int, min_length: int) -> None:
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be a positive integer.")
    if type(min_length) is not int or min_length <= 0:
        raise ValueError("min_length must be a positive integer.")
    if min_length > max_length:
        raise ValueError("min_length cannot exceed max_length.")


def _contains_isolated_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in text)


def _truncate_at_grapheme_boundary(text: str, max_length: int) -> str:
    r"""Return the longest prefix no longer than max_length ending at a \X boundary."""
    if len(text) <= max_length:
        return text

    safe_end = 0
    for match in GRAPHEME_PATTERN.finditer(text):
        if match.end() > max_length:
            break
        safe_end = match.end()

    if safe_end == 0:
        raise ValueError(
            "max_length is too small to retain the first complete Unicode grapheme cluster."
        )
    return text[:safe_end].rstrip()


def preprocess_text(
    text: str,
    max_length: int = 10000,
    min_length: int = 1,
) -> PreprocessedText:
    """Normalize user text into a bounded working copy while preserving original evidence.

    Args:
        text: Raw input text to assess.
        max_length: Maximum working-copy length measured in Python code points.
        min_length: Minimum substantive working-copy length after preprocessing.

    Raises:
        ValueError: For invalid input, invalid length settings, isolated surrogates,
            invisible-only content, or content shorter than min_length after cleanup.
    """
    _validate_lengths(max_length, min_length)

    if not isinstance(text, str):
        raise ValueError("Input text must be a valid non-null string.")
    if _contains_isolated_surrogate(text):
        raise ValueError("Input text contains an isolated Unicode surrogate code point.")
    if len(text.strip()) < min_length:
        raise ValueError("Input text cannot be empty or whitespace only.")

    warnings: List[str] = []

    working = unicodedata.normalize("NFKC", text)
    if working != text:
        warnings.append("Applied Unicode NFKC normalization to the analysis copy.")

    before = working
    working = ZERO_WIDTH_CHARS_PATTERN.sub("", working)
    if working != before:
        warnings.append(
            "Removed zero-width spacing/BOM and bidirectional formatting controls; "
            "preserved language-significant joiners and directional marks."
        )

    before = working
    working = CONTROL_CHARS_PATTERN.sub("", working)
    if working != before:
        warnings.append("Removed non-printable control characters from the analysis copy.")

    before = working
    working = LINE_ENDING_PATTERN.sub("\n", working)
    working = MULTI_SPACE_PATTERN.sub(" ", working)
    working = MULTI_NEWLINE_PATTERN.sub("\n\n", working)
    working = working.strip()
    if working != before:
        warnings.append("Normalized line endings, repeated whitespace, or surrounding whitespace.")

    has_substantive_content = any(
        not character.isspace()
        and not unicodedata.category(character).startswith("M")
        and unicodedata.category(character) not in {"Cf", "Cc", "Cs"}
        for character in working
    )
    if not has_substantive_content:
        raise ValueError("Input text contains only non-printable or invisible characters.")

    if len(working) < min_length:
        raise ValueError(
            f"Input text is shorter than min_length ({min_length}) after preprocessing."
        )

    is_truncated = False
    if len(working) > max_length:
        working = _truncate_at_grapheme_boundary(working, max_length)
        if len(working) < min_length:
            raise ValueError("Truncation left less than min_length substantive text.")
        is_truncated = True
        warnings.append(
            f"Input text exceeded {max_length} characters and was truncated at a Unicode grapheme boundary."
        )

    words = working.split()
    return PreprocessedText(
        original_text=text,
        working_copy=working,
        char_count=len(working),
        word_count=len(words),
        is_truncated=is_truncated,
        warnings=warnings,
    )
