import re
from typing import List


_LEADING_LIST_MARKER_RE = re.compile(
    r"^\s*(?:\*\*|\*|__|_)?\??\d{1,3}[.)](?:\*\*|\*|__|_)?\s+"
)


def _normalize_item_text(text: str) -> str:
    cleaned = _LEADING_LIST_MARKER_RE.sub("", text.strip(), count=1)
    if cleaned.startswith("- ") or cleaned.startswith("• ") or cleaned.startswith("* "):
        cleaned = cleaned[2:]
    return cleaned.strip()


def normalize_corpus_item(text: str, category: str) -> str:
    if category == "paragraphs":
        return _normalize_item_text(text)
    return _normalize_item_text(text)


def parse_response(response: str, category: str) -> List[str]:
    lines = response.strip().split("\n")

    if category == "paragraphs":
        paragraphs: List[str] = []
        current: List[str] = []
        for line in lines:
            cleaned = _normalize_item_text(line)
            if cleaned:
                current.append(cleaned)
            elif current:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
        return paragraphs

    items: List[str] = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = _normalize_item_text(cleaned)
        if cleaned:
            items.append(cleaned)

    return items
