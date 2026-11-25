"""Safe evidence highlighting and strict UTF-16 span partitioning for SafeHER."""

import html
from dataclasses import dataclass, field
from typing import List, Sequence

from ai_core.threat.schemas import FlaggedSpan, ThreatCategory


@dataclass(frozen=True)
class HighlightSegment:
    """An elementary non-overlapping segment of the analyzed text."""

    text: str
    categories: List[ThreatCategory] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    start_utf16: int = 0
    end_utf16: int = 0
    is_flagged: bool = False


def _encode_utf16(text: str) -> bytes:
    try:
        return text.encode("utf-16-le")
    except UnicodeEncodeError as exc:
        raise ValueError("Text contains an isolated Unicode surrogate code point.") from exc


def _is_code_point_boundary(encoded: bytes, offset: int) -> bool:
    """Return False when a UTF-16 offset falls between a surrogate pair."""
    total_units = len(encoded) // 2
    if offset <= 0 or offset >= total_units:
        return True
    previous_unit = int.from_bytes(encoded[(offset - 1) * 2 : offset * 2], "little")
    next_unit = int.from_bytes(encoded[offset * 2 : (offset + 1) * 2], "little")
    return not (0xD800 <= previous_unit <= 0xDBFF and 0xDC00 <= next_unit <= 0xDFFF)


def slice_utf16(text: str, start_u16: int, end_u16: int) -> str:
    """Slice text using JavaScript UTF-16 code-unit offsets; reject invalid boundaries."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if type(start_u16) is not int or type(end_u16) is not int:
        raise TypeError("UTF-16 offsets must be integers.")
    if start_u16 < 0 or end_u16 < start_u16:
        raise ValueError("UTF-16 offsets must be non-negative and ordered.")

    encoded = _encode_utf16(text)
    total_units = len(encoded) // 2
    if end_u16 > total_units:
        raise ValueError("UTF-16 end offset exceeds the text length.")
    if not _is_code_point_boundary(encoded, start_u16):
        raise ValueError("UTF-16 start offset splits a surrogate pair.")
    if not _is_code_point_boundary(encoded, end_u16):
        raise ValueError("UTF-16 end offset splits a surrogate pair.")
    return encoded[start_u16 * 2 : end_u16 * 2].decode("utf-16-le")


def _validate_span_for_text(analyzed_text: str, span: FlaggedSpan, index: int) -> None:
    if not isinstance(span, FlaggedSpan):
        raise TypeError(f"spans[{index}] must be a FlaggedSpan instance.")

    if not (0 <= span.start < span.end <= len(analyzed_text)):
        raise ValueError(f"spans[{index}] has invalid or out-of-range code-point offsets.")
    if analyzed_text[span.start : span.end] != span.text:
        raise ValueError(f"spans[{index}] text does not match analyzed_text at its code-point offsets.")

    encoded = _encode_utf16(analyzed_text)
    total_units = len(encoded) // 2
    if not (0 <= span.start_utf16 < span.end_utf16 <= total_units):
        raise ValueError(f"spans[{index}] has invalid or out-of-range UTF-16 offsets.")
    if not _is_code_point_boundary(encoded, span.start_utf16):
        raise ValueError(f"spans[{index}] start offset splits a UTF-16 surrogate pair.")
    if not _is_code_point_boundary(encoded, span.end_utf16):
        raise ValueError(f"spans[{index}] end offset splits a UTF-16 surrogate pair.")

    expected_start = len(analyzed_text[: span.start].encode("utf-16-le")) // 2
    expected_end = len(analyzed_text[: span.end].encode("utf-16-le")) // 2
    if span.start_utf16 != expected_start or span.end_utf16 != expected_end:
        raise ValueError(f"spans[{index}] UTF-16 offsets do not correspond to its code-point offsets.")


def partition_spans(
    analyzed_text: str,
    spans: Sequence[FlaggedSpan],
) -> List[HighlightSegment]:
    """Partition analyzed_text into disjoint segments after strictly validating spans.

    Invalid offsets are rejected. They are never clamped into a different highlight.
    Every valid input character is represented exactly once in the returned segments.
    """
    if not isinstance(analyzed_text, str):
        raise TypeError("analyzed_text must be a string.")
    if analyzed_text:
        encoded = _encode_utf16(analyzed_text)
    else:
        encoded = b""

    if not spans:
        if not analyzed_text:
            return []
        return [
            HighlightSegment(
                text=analyzed_text,
                categories=[],
                explanations=[],
                start_utf16=0,
                end_utf16=len(encoded) // 2,
                is_flagged=False,
            )
        ]

    if not analyzed_text:
        raise ValueError("Cannot highlight spans against empty analyzed_text.")

    for index, span in enumerate(spans):
        _validate_span_for_text(analyzed_text, span, index)

    text_utf16_len = len(encoded) // 2
    boundaries = {0, text_utf16_len}
    for span in spans:
        boundaries.add(span.start_utf16)
        boundaries.add(span.end_utf16)

    points = sorted(boundaries)
    segments: List[HighlightSegment] = []
    for p_start, p_end in zip(points, points[1:]):
        if p_start >= p_end:
            continue

        segment_text = slice_utf16(analyzed_text, p_start, p_end)
        active_spans = [
            span
            for span in spans
            if span.start_utf16 <= p_start and p_end <= span.end_utf16
        ]

        categories: List[ThreatCategory] = []
        explanations: List[str] = []
        seen_categories = set()
        seen_explanations = set()
        for span in active_spans:
            if span.category not in seen_categories:
                categories.append(span.category)
                seen_categories.add(span.category)
            description = f"{span.category.value}: {span.explanation}"
            if description not in seen_explanations:
                explanations.append(description)
                seen_explanations.add(description)

        segments.append(
            HighlightSegment(
                text=segment_text,
                categories=categories,
                explanations=explanations,
                start_utf16=p_start,
                end_utf16=p_end,
                is_flagged=bool(active_spans),
            )
        )

    if "".join(segment.text for segment in segments) != analyzed_text:
        raise RuntimeError("Highlight partition failed to preserve analyzed_text exactly.")
    return segments


def render_highlighted_html(
    analyzed_text: str,
    spans: Sequence[FlaggedSpan],
) -> str:
    """Render escaped HTML; never interpolate untrusted text or explanations raw."""
    segments = partition_spans(analyzed_text, spans)
    html_parts: List[str] = []

    for segment in segments:
        escaped_text = html.escape(segment.text, quote=True)
        if not segment.is_flagged:
            html_parts.append(escaped_text)
            continue

        # Category values originate from the closed ThreatCategory enum.
        category_classes = " ".join(f"threat-{category.value}" for category in segment.categories)
        data_categories = html.escape(",".join(c.value for c in segment.categories), quote=True)
        title = html.escape("; ".join(segment.explanations), quote=True)
        html_parts.append(
            f'<mark class="threat-span {category_classes}" '
            f'data-categories="{data_categories}" title="{title}">{escaped_text}</mark>'
        )

    return "".join(html_parts)
