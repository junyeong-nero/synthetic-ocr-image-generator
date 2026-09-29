"""Remove annotated page furniture from predictions before block scoring."""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Mapping, Optional


def _normalize(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).casefold().strip()
    line = re.sub(r"^#{1,6}\s+", "", line)
    return re.sub(r"\s+", "", line).strip("*_`")


def strip_page_furniture(prediction: str, furniture: Optional[Mapping[str, str]]) -> str:
    """Match whole lines; short strings/numbers require an exact match.

    Longer annotations tolerate a small OCR error (similarity >= 0.9).
    Never use substring matches, which could remove ordinary body sentences.
    Missing annotations preserve the prediction byte for byte.
    """
    if not isinstance(furniture, Mapping):
        return prediction
    targets = {
        _normalize(line)
        for key in ("header", "footer", "page_number")
        for line in str(furniture.get(key) or "").splitlines()
        if _normalize(line)
    }
    if not targets:
        return prediction

    def matches(line: str) -> bool:
        normalized = _normalize(line)
        return any(
            normalized == target
            or (
                min(len(normalized), len(target)) >= 10
                and SequenceMatcher(None, normalized, target).ratio() >= 0.9
            )
            for target in targets
        )

    return "".join(line for line in prediction.splitlines(keepends=True) if not matches(line))
