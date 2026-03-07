import random
from typing import Callable


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


def mutate_text_generator_sections(
    markdown_text: str,
    ratio: float,
    merge_order: list[str],
    mutate_section: Callable[[str, float], tuple[str, int]],
) -> tuple[str, int]:
    if not merge_order:
        return markdown_text, 0

    lines = markdown_text.splitlines()
    if not lines:
        return markdown_text, 0

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

    if len(sections) != len(merge_order):
        return markdown_text, 0

    mutated_sections: list[str] = []
    mutated_count = 0
    for section_text, section_type in zip(sections, merge_order):
        if section_type == "text":
            mutated_section, section_mutations = mutate_section(section_text, ratio)
            mutated_sections.append(mutated_section)
            mutated_count += section_mutations
        else:
            mutated_sections.append(section_text)

    prefix = "\n".join(preface_lines).rstrip()
    body = "\n\n".join(mutated_sections).strip()
    if prefix and body:
        return f"{prefix}\n\n{body}", mutated_count
    if body:
        return body, mutated_count
    return prefix, mutated_count
