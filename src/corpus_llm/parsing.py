from typing import List


def parse_response(response: str, category: str) -> List[str]:
    lines = response.strip().split("\n")

    if category == "paragraphs":
        paragraphs: List[str] = []
        current: List[str] = []
        for line in lines:
            cleaned = line.strip()
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
        if cleaned[0].isdigit() and (". " in cleaned[:4] or ") " in cleaned[:4]):
            cleaned = cleaned.split(". ", 1)[-1].split(") ", 1)[-1]
        if cleaned.startswith("- ") or cleaned.startswith("• ") or cleaned.startswith("* "):
            cleaned = cleaned[2:]
        cleaned = cleaned.strip()
        if cleaned:
            items.append(cleaned)

    return items
