"""Per-page text statistics: character-class shares and symbol usage.

Complements the image-based metrics in `image_stats.py` / `layout_stats.py`
with what the *content* of a page looks like, so a lorem-ipsum synthetic page
does not read as "close" to real text just because the pixels line up. All
functions operate on plain strings; for synthetic `GT_markdown`, strip
markdown syntax first with `strip_markdown_syntax` (the CLI does this
automatically, see `distribution text-stats --metadata`).

Character-class shares are computed against `len(text)` (including
whitespace), so they need not sum to 1.0 exactly: characters outside the
tracked scripts (Cyrillic, Arabic, control characters, ...) are simply left
uncounted.

**`mean_line_length` is not comparable across sources with different line
conventions.** `GT_markdown` paragraphs are one long logical line each (line
breaks only at explicit markdown boundaries such as list items or paragraph
ends), while OCR transcripts and most real reference `.txt` files keep the
document's *visual* line breaks (one line per printed line, wrapped at the
page margin). Comparing the two directly with `distribution compare` will
show a large, meaningless gap driven by this convention difference rather
than by real content statistics; either reformat one side to match the
other's line convention first, or exclude `mean_line_length` from the
comparison.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from src.realism.distribution_stats import summarize_metric_rows

# Unicode block ranges (inclusive) used to classify a character's script.
_HANGUL_RANGES = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
)
_HANJA_RANGES = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)
_KANA_RANGES = (
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
)
_LATIN_RANGES = (
    (0x0041, 0x005A),  # Basic Latin upper
    (0x0061, 0x007A),  # Basic Latin lower
    (0x00C0, 0x024F),  # Latin-1 Supplement + Latin Extended-A/B
    (0x1E00, 0x1EFF),  # Latin Extended Additional
)

# Characters that read as reference/list "symbols" in Korean/CJK documents
# (bullets, reference marks) even though Unicode's general category classes
# some of them as punctuation rather than "So" / "No".
_SYMBOL_OVERRIDE = {"※", "‡", "†", "§", "¶", "·"}

CHAR_CLASSES = (
    "hangul",
    "hanja",
    "kana",
    "latin",
    "digit",
    "punctuation",
    "symbol",
    "whitespace",
)


def _in_ranges(codepoint: int, ranges: Tuple[Tuple[int, int], ...]) -> bool:
    return any(lo <= codepoint <= hi for lo, hi in ranges)


def classify_char(ch: str) -> Optional[str]:
    """Classify one character into a `CHAR_CLASSES` bucket, or None."""
    if ch.isspace():
        return "whitespace"
    codepoint = ord(ch)
    if _in_ranges(codepoint, _HANGUL_RANGES):
        return "hangul"
    if _in_ranges(codepoint, _HANJA_RANGES):
        return "hanja"
    if _in_ranges(codepoint, _KANA_RANGES):
        return "kana"
    if _in_ranges(codepoint, _LATIN_RANGES):
        return "latin"
    category = unicodedata.category(ch)
    if category == "Nd":
        return "digit"
    if ch in _SYMBOL_OVERRIDE:
        return "symbol"
    if category.startswith("P"):
        return "punctuation"
    if category.startswith("S") or category == "No":
        return "symbol"
    return None


def compute_text_stats(text: str) -> Dict[str, float]:
    """Character-class shares, length and line-length stats for one page."""
    counts: Counter[str] = Counter()
    for ch in text:
        cls = classify_char(ch)
        if cls is not None:
            counts[cls] += 1

    total_chars = len(text)
    lines = [line for line in text.split("\n") if line.strip()]
    mean_line_length = float(sum(len(line) for line in lines) / len(lines)) if lines else 0.0

    stats: Dict[str, float] = {
        "chars_per_page": float(total_chars),
        "line_count": float(len(lines)),
        "mean_line_length": mean_line_length,
    }
    denom = max(1, total_chars)
    for cls in CHAR_CLASSES:
        stats[f"{cls}_share"] = counts.get(cls, 0) / denom
    return stats


def top_symbols(texts: Iterable[str], top_k: int = 20) -> List[Tuple[str, int]]:
    """Most frequent punctuation/symbol characters across `texts`."""
    counter: Counter[str] = Counter()
    for text in texts:
        for ch in text:
            if classify_char(ch) in ("punctuation", "symbol"):
                counter[ch] += 1
    return counter.most_common(top_k)


# --- Markdown stripping -----------------------------------------------------

# Best-effort, not a full parser: good enough to keep character-class and
# line-length stats from being skewed by markdown/HTML syntax (`#`, `*`,
# `|`, list markers, HTML table tags) that would not appear in real OCR text.
_MARKDOWN_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"```.*?```", re.DOTALL), " "),  # fenced code blocks
    (re.compile(r"`([^`]*)`"), r"\1"),  # inline code
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),  # images
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),  # links -> link text
    (re.compile(r"<[^>]+>"), " "),  # HTML tags (tables, math, page furniture)
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),  # ATX headings
    (re.compile(r"^\s{0,3}>\s?", re.MULTILINE), ""),  # blockquote markers
    (re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE), ""),  # bullet markers
    (re.compile(r"^\s{0,3}\d+[.)]\s+", re.MULTILINE), ""),  # numbered markers
    (re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE), ""),  # table separators
    (re.compile(r"\*\*([^*\n]+)\*\*"), r"\1"),  # bold
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), r"\1"),  # italic (asterisk)
    (re.compile(r"(?<!_)_([^_\n]+)_(?!_)"), r"\1"),  # italic (underscore)
    (re.compile(r"\|"), " "),  # remaining table pipes
]


def strip_markdown_syntax(text: str) -> str:
    """Best-effort removal of markdown/HTML syntax, leaving plain prose."""
    result = text
    for pattern, repl in _MARKDOWN_PATTERNS:
        result = pattern.sub(repl, result)
    return result


# --- Summary + CLI-facing iterators -----------------------------------------


def summarize_text_stats(
    rows: List[Dict[str, float]],
    *,
    source: str = "",
    top_symbols: Optional[List[Tuple[str, int]]] = None,
) -> Dict[str, Any]:
    """Summary with the same `{source, count, metrics}` shape `summarize_stats`
    produces, so `distribution compare` can diff two text-stat summaries."""
    summary: Dict[str, Any] = {
        "source": source,
        "count": len(rows),
        "metrics": summarize_metric_rows(rows),
    }
    if top_symbols is not None:
        summary["top_symbols"] = [{"char": ch, "count": count} for ch, count in top_symbols]
    return summary


def iter_metadata_texts(
    metadata_path: Path,
    *,
    field: str = "GT_markdown",
    limit: Optional[int] = None,
) -> Iterator[str]:
    """Yield `field` from each row of a generated `metadata.jsonl`, with
    markdown syntax stripped (the field holds `GT_markdown` by default)."""
    count = 0
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get(field)
            if not text:
                continue
            yield strip_markdown_syntax(str(text))
            count += 1
            if limit is not None and count >= limit:
                return


def iter_jsonl_texts(path: Path, *, field: str = "text", limit: Optional[int] = None) -> Iterator[str]:
    """Yield `field` from each row of a plain-text JSONL file (no stripping)."""
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get(field)
            if not text:
                continue
            yield str(text)
            count += 1
            if limit is not None and count >= limit:
                return


def iter_text_dir(root: Path, *, pattern: str = "*.txt", limit: Optional[int] = None) -> Iterator[str]:
    """Yield the contents of each matching file under `root` (no stripping)."""
    count = 0
    for path in sorted(root.rglob(pattern)):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        yield text
        count += 1
        if limit is not None and count >= limit:
            return
