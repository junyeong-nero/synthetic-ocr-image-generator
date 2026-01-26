"""Text normalization utilities for OCR evaluation."""

import re
import unicodedata
from typing import Callable, Optional


def normalize_text(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    normalize_whitespace: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """
    Normalize text for OCR comparison.

    Args:
        text: Input text to normalize.
        lowercase: Convert to lowercase.
        remove_punctuation: Remove punctuation characters.
        normalize_whitespace: Collapse multiple whitespace to single space.
        normalize_unicode: Apply NFKC unicode normalization.

    Returns:
        Normalized text.
    """
    if text is None:
        return ""

    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    if normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(r"[^\w\s]", "", text)

    return text


def normalize_for_language(text: str, lang: str) -> str:
    """
    Apply language-specific normalization.

    Args:
        text: Input text.
        lang: Language code (e.g., "ko", "ja", "zh", "en").

    Returns:
        Language-normalized text.
    """
    if lang in ("ko", "ja", "zh"):
        # Remove spaces between CJK characters (they don't use spaces between words)
        cjk_pattern = (
            r"(?<=[\u4e00-\u9fff\uac00-\ud7af\u3040-\u309f\u30a0-\u30ff])"
            r"\s+"
            r"(?=[\u4e00-\u9fff\uac00-\ud7af\u3040-\u309f\u30a0-\u30ff])"
        )
        text = re.sub(cjk_pattern, "", text)

    if lang == "ko":
        # Korean-specific: normalize Hangul jamo
        text = unicodedata.normalize("NFC", text)

    if lang == "ja":
        # Japanese-specific: normalize katakana/hiragana variations
        # Convert full-width to half-width where appropriate
        pass

    return text


def normalize_for_ocr(text: str, strict: bool = False) -> str:
    """
    Apply OCR-specific normalization.

    Handles common OCR artifacts and variations.

    Args:
        text: Input text.
        strict: Apply stricter normalization (more aggressive).

    Returns:
        OCR-normalized text.
    """
    if text is None:
        return ""

    # Basic normalization
    text = unicodedata.normalize("NFKC", text)

    # Common OCR substitutions
    ocr_substitutions = {
        # Zero vs O
        "０": "0",
        "Ｏ": "O",
        # One vs l vs I
        "１": "1",
        "ｌ": "l",
        # Common ligatures
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        # Quotes
        """: '"',
        """: '"',
        "'": "'",
        "'": "'",
        # Dashes
        "–": "-",
        "—": "-",
        "−": "-",
    }

    for old, new in ocr_substitutions.items():
        text = text.replace(old, new)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if strict:
        # Remove all non-alphanumeric except basic punctuation
        text = re.sub(r"[^\w\s.,!?;:\-'\"]", "", text)
        text = text.lower()

    return text


def create_normalizer(
    lowercase: bool = False,
    remove_punctuation: bool = False,
    language: Optional[str] = None,
    ocr_normalize: bool = False,
) -> Callable[[str], str]:
    """
    Create a normalizer function with specified options.

    Args:
        lowercase: Convert to lowercase.
        remove_punctuation: Remove punctuation.
        language: Language code for language-specific normalization.
        ocr_normalize: Apply OCR-specific normalization.

    Returns:
        Normalizer function.
    """

    def normalizer(text: str) -> str:
        if ocr_normalize:
            text = normalize_for_ocr(text)
        else:
            text = normalize_text(
                text,
                lowercase=lowercase,
                remove_punctuation=remove_punctuation,
            )

        if language:
            text = normalize_for_language(text, language)

        return text

    return normalizer


def strip_markdown(text: str) -> str:
    """
    Strip markdown formatting for comparison.

    Args:
        text: Markdown text.

    Returns:
        Plain text without markdown formatting.
    """
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Remove links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # Remove list markers
    text = re.sub(r"^[\*\-\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
