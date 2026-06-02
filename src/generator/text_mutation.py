import random
import re
from typing import Callable


PROSE_SECTION_TYPES = {
    "text",
    "paragraph",
    "bullet_list",
    "numbered_list",
    "checklist",
    "quote",
}
FENCED_SECTION_TYPES = {"code", "command"}

_CHECKLIST_RE = re.compile(r"^-\s+\[[ xX]\]\s+\S")
_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\([^)]+\)$")
_NUMBERED_LIST_RE = re.compile(r"^\d+\.\s+\S")


def mutate_similar_text(
    text: str,
    ratio: float,
    similarity_db: dict,
    protected_chars: set[str],
    candidate_lookup: Callable[[str], list[tuple[str, float]]],
) -> tuple[str, int]:
    if ratio <= 0 or not similarity_db:
        return text, 0

    chars = list(text)
    candidate_indices: list[int] = []

    for idx, ch in enumerate(chars):
        if ch in protected_chars or ch.isspace():
            continue
        if candidate_lookup(ch):
            candidate_indices.append(idx)

    if not candidate_indices:
        return text, 0

    target = int(len(candidate_indices) * ratio)
    if target == 0:
        target = 1
    target = min(target, len(candidate_indices))

    mutated_count = 0
    for idx in random.sample(candidate_indices, target):
        source = chars[idx]
        candidates = candidate_lookup(source)
        if not candidates:
            continue
        replacement, _ = random.choice(candidates)
        if not replacement or any(c in protected_chars or c.isspace() for c in replacement):
            continue
        if replacement == source:
            continue
        chars[idx] = replacement
        mutated_count += 1

    return "".join(chars), mutated_count


def _is_fence_line(line: str) -> bool:
    return line.strip().startswith("```")


def _normalize_block_type(block_type: str) -> str:
    return str(block_type).strip().lower().replace("-", "_")


def _parse_markdown_sections(markdown_text: str) -> tuple[list[str], list[str]] | None:
    lines = markdown_text.splitlines()
    preface_lines: list[str] = []
    sections: list[str] = []
    current_section: list[str] = []
    in_section = False
    in_fenced_block = False

    for line in lines:
        if _is_fence_line(line):
            if in_section:
                current_section.append(line)
            else:
                preface_lines.append(line)
            in_fenced_block = not in_fenced_block
            continue

        if not in_fenced_block and line.startswith("## "):
            if current_section:
                sections.append("\n".join(current_section).rstrip())
            current_section = [line]
            in_section = True
            continue

        if in_section:
            current_section.append(line)
        else:
            preface_lines.append(line)

    if current_section:
        sections.append("\n".join(current_section).rstrip())

    if in_fenced_block:
        return None

    return preface_lines, sections


def _join_preface_and_sections(preface_lines: list[str], sections: list[str]) -> str:
    prefix = "\n".join(preface_lines).rstrip()
    body = "\n\n".join(sections).strip()
    if prefix and body:
        return f"{prefix}\n\n{body}"
    if body:
        return body
    return prefix


def _split_section_block_chunks(section_text: str) -> tuple[str, list[str]] | None:
    lines = section_text.splitlines()
    if not lines or not lines[0].startswith("## "):
        return None

    heading = lines[0]
    chunks: list[str] = []
    current_chunk: list[str] = []
    in_fenced_block = False

    for line in lines[1:]:
        stripped = line.strip()
        if _is_fence_line(line):
            current_chunk.append(line)
            in_fenced_block = not in_fenced_block
            continue

        if not in_fenced_block and not stripped:
            if current_chunk:
                chunks.append("\n".join(current_chunk).rstrip())
                current_chunk = []
            continue

        current_chunk.append(line)

    if in_fenced_block:
        return None

    if current_chunk:
        chunks.append("\n".join(current_chunk).rstrip())

    return heading, chunks


def _is_table_separator_line(line: str) -> bool:
    if not line.startswith("|") or not line.endswith("|"):
        return False

    cells = [cell.strip() for cell in line.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _is_table_chunk(lines: list[str]) -> bool:
    if not lines:
        return False
    if not all(line.startswith("|") and line.endswith("|") for line in lines):
        return False
    if len(lines) == 1:
        return True
    return _is_table_separator_line(lines[1])


def _classify_markdown_chunk(chunk: str) -> str | None:
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    if not lines:
        return None

    fence_lines = [index for index, line in enumerate(lines) if line.startswith("```")]
    if fence_lines:
        if len(fence_lines) == 2 and fence_lines == [0, len(lines) - 1]:
            return "code"
        return None

    stripped_chunk = "\n".join(lines)
    if stripped_chunk.startswith("$$") and stripped_chunk.endswith("$$"):
        return "formula"
    if _is_table_chunk(lines):
        return "table"
    if len(lines) == 1 and _IMAGE_RE.fullmatch(lines[0]):
        return "image"
    if len(lines) == 1 and lines[0] in {"---", "***"}:
        return "rule"
    if all(_CHECKLIST_RE.match(line) for line in lines):
        return "checklist"
    if all(line.startswith("- ") for line in lines):
        return "bullet_list"
    if all(_NUMBERED_LIST_RE.match(line) for line in lines):
        return "numbered_list"
    if all(line.startswith("> ") for line in lines):
        return "quote"

    for line in lines:
        if (
            line.startswith(("#", "```", "|", "$$", "![", "> ", "- ", "* ", "+ "))
            or line in {"---", "***"}
            or _NUMBERED_LIST_RE.match(line)
        ):
            return None

    return "paragraph"


def _is_chunk_compatible(expected_type: str, actual_type: str | None) -> bool:
    if actual_type is None:
        return False

    normalized_expected = _normalize_block_type(expected_type)
    if actual_type == "code":
        return normalized_expected in FENCED_SECTION_TYPES
    if normalized_expected == "text":
        normalized_expected = "paragraph"
    return normalized_expected == actual_type


def _section_matches_expected_type(section_text: str, expected_type: str) -> bool:
    parsed_section = _split_section_block_chunks(section_text)
    if parsed_section is None:
        return False

    _heading, chunks = parsed_section
    if not chunks:
        return False

    normalized_expected = _normalize_block_type(expected_type)
    if normalized_expected not in PROSE_SECTION_TYPES and len(chunks) != 1:
        return False

    return all(
        _is_chunk_compatible(expected_type, _classify_markdown_chunk(chunk))
        for chunk in chunks
    )


def _mutate_rich_block_sections(
    sections: list[str],
    ratio: float,
    merge_order: list[str],
    mutate_section: Callable[[str, float], tuple[str, int]],
) -> tuple[list[str], int] | None:
    parsed_sections: list[tuple[str, list[str]]] = []
    chunk_count = 0

    for section_text in sections:
        parsed_section = _split_section_block_chunks(section_text)
        if parsed_section is None:
            return None
        _heading, chunks = parsed_section
        if not chunks:
            return None
        parsed_sections.append(parsed_section)
        chunk_count += len(chunks)

    if chunk_count != len(merge_order):
        return None

    mutated_sections: list[str] = []
    mutated_count = 0
    merge_index = 0

    for heading, chunks in parsed_sections:
        mutated_chunks: list[str] = []
        for chunk in chunks:
            block_type = merge_order[merge_index]
            merge_index += 1
            actual_type = _classify_markdown_chunk(chunk)
            if not _is_chunk_compatible(block_type, actual_type):
                return None

            if _normalize_block_type(block_type) in PROSE_SECTION_TYPES:
                mutated_chunk, chunk_mutations = mutate_section(chunk, ratio)
                mutated_chunks.append(mutated_chunk)
                mutated_count += chunk_mutations
            else:
                mutated_chunks.append(chunk)

        mutated_sections.append(f"{heading}\n\n" + "\n\n".join(mutated_chunks))

    return mutated_sections, mutated_count


def mutate_text_generator_sections(
    markdown_text: str,
    ratio: float,
    merge_order: list[str],
    mutate_section: Callable[[str, float], tuple[str, int]],
) -> tuple[str, int]:
    if not merge_order:
        return markdown_text, 0

    if not markdown_text.splitlines():
        return markdown_text, 0

    parsed_sections = _parse_markdown_sections(markdown_text)
    if parsed_sections is None:
        return markdown_text, 0

    preface_lines, sections = parsed_sections

    if len(sections) == len(merge_order):
        mutated_sections: list[str] = []
        mutated_count = 0
        for section_text, section_type in zip(sections, merge_order):
            if not _section_matches_expected_type(section_text, section_type):
                return markdown_text, 0

            if _normalize_block_type(section_type) in PROSE_SECTION_TYPES:
                mutated_section, section_mutations = mutate_section(section_text, ratio)
                mutated_sections.append(mutated_section)
                mutated_count += section_mutations
            else:
                mutated_sections.append(section_text)

        if mutated_count == 0:
            return markdown_text, 0
        return _join_preface_and_sections(preface_lines, mutated_sections), mutated_count

    rich_result = _mutate_rich_block_sections(
        sections=sections,
        ratio=ratio,
        merge_order=merge_order,
        mutate_section=mutate_section,
    )
    if rich_result is None:
        return markdown_text, 0

    mutated_sections, mutated_count = rich_result
    if mutated_count == 0:
        return markdown_text, 0
    return _join_preface_and_sections(preface_lines, mutated_sections), mutated_count
