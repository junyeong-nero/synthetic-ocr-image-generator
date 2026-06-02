import random
from typing import Callable


PROSE_SECTION_TYPES = {
    "text",
    "paragraph",
    "bullet_list",
    "numbered_list",
    "checklist",
    "quote",
}


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


def _parse_markdown_sections(markdown_text: str) -> tuple[list[str], list[str]]:
    lines = markdown_text.splitlines()
    preface_lines: list[str] = []
    sections: list[str] = []
    current_section: list[str] = []
    in_section = False

    for line in lines:
        if line.startswith("## "):
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
        if stripped.startswith("```"):
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

            if block_type in PROSE_SECTION_TYPES:
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

    preface_lines, sections = _parse_markdown_sections(markdown_text)

    if len(sections) == len(merge_order):
        mutated_sections: list[str] = []
        mutated_count = 0
        for section_text, section_type in zip(sections, merge_order):
            if section_type in PROSE_SECTION_TYPES:
                mutated_section, section_mutations = mutate_section(section_text, ratio)
                mutated_sections.append(mutated_section)
                mutated_count += section_mutations
            else:
                mutated_sections.append(section_text)

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
    return _join_preface_and_sections(preface_lines, mutated_sections), mutated_count
