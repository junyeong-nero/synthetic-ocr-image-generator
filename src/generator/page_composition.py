"""Continuation pages: a page cut from the middle of a multi-page document.

The composer always writes a document's *first* page: a `#` title followed by
sections. Most pages of real multi-page files are later pages (DocLayNet
labels a Title on only 5,071 of its 80,863 pages), which start at some later
section, often in the middle of a paragraph that began on the previous page.

Two ``content`` profile keys control this, both off unless a profile sets them:

- ``continuation_page`` (p): drop the `#` title and a random number of leading
  sections, so the page starts at a later `## ` section (a single-section
  document only loses its title).
- ``start_mid_paragraph`` (p, sampled only on continuation pages): the page
  starts with the tail of a paragraph, cut at a sentence boundary. The cut
  paragraph is either the first block of the first kept section (whose
  heading then stays on the previous page) or the last block of the last
  dropped section. The section break is chosen among those that allow such a
  cut, so the requested rate holds whenever the document has a paragraph of
  at least two sentences next to some break.

The page is rebuilt from the composer's blank-line separated chunks (the same
convention as ``profile_application.fit_markdown_to_sheet``), and
``merge_order`` / composition metadata are recomputed from the blocks that
remain, so they keep describing exactly the page. Documents whose
``merge_order`` is not one entry per non-heading chunk (the legacy
text/table/formula orchestrator lists sections, not blocks) are left
untouched rather than risk desynchronising merge_order and GT.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

CONTINUATION_KEYS = ("continuation_page", "start_mid_paragraph")

# Same sentence split as DataProvider._split_sentences.
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?。！？])\s+")
# Same block-marker prefix as document_blocks._neutralize_block_markup.
_BLOCK_MARKUP_PREFIX_RE = re.compile(r"^\s*(?:[#>|*+\-]+|\d+[.)])\s+")


@dataclass
class PageComposition:
    markdown: str
    merge_order: List[str]
    composition_metadata: Dict[str, Any]
    continuation_page: bool = False
    start_mid_paragraph: bool = False

    def metadata(self) -> Dict[str, Any]:
        return {
            "continuation_page": bool(self.continuation_page),
            "start_mid_paragraph": bool(self.start_mid_paragraph),
        }


@dataclass
class _Section:
    heading: Optional[str]
    blocks: List[Tuple[str, str]] = field(default_factory=list)  # (markdown, block_type)


def compose_page(
    markdown_text: str,
    merge_order: List[str],
    composition_metadata: Mapping[str, Any],
    content_specs: Mapping[str, Any],
    rng: Any,
) -> Optional[PageComposition]:
    """Turn a composed first page into a continuation page per the profile.

    Returns ``None`` (and draws nothing from ``rng``) when ``content_specs``
    sets neither key, so runs without these keys follow the legacy path.
    """
    if not any(key in content_specs for key in CONTINUATION_KEYS):
        return None

    unchanged = PageComposition(markdown_text, list(merge_order), dict(composition_metadata))
    if not _sample_flag(content_specs.get("continuation_page"), rng):
        return unchanged
    start_mid = _sample_flag(content_specs.get("start_mid_paragraph"), rng)

    parsed = _parse_sections(markdown_text, merge_order)
    if parsed is None:
        return unchanged
    sections = parsed

    starts = _candidate_starts(sections)
    mid_starts = [k for k in starts if _mid_paragraph_cuts(sections, k)] if start_mid else []
    if mid_starts:
        start = rng.choice(mid_starts)
        kept = _cut_mid_paragraph(sections, start, rng)
    else:
        start_mid = False
        start = rng.choice(starts)
        kept = [_Section(section.heading, list(section.blocks)) for section in sections[start:]]

    chunks: List[str] = []
    page_merge_order: List[str] = []
    for section in kept:
        if section.heading is not None:
            chunks.append(section.heading)
        for block_markdown, block_type in section.blocks:
            chunks.append(block_markdown)
            page_merge_order.append(block_type)
    page_markdown = "\n\n".join(chunks) + "\n"

    metadata = dict(composition_metadata)
    metadata["block_types"] = list(page_merge_order)
    metadata["block_type_counts"] = dict(Counter(page_merge_order))
    metadata["section_count"] = sum(1 for line in page_markdown.splitlines() if line.startswith("## "))
    return PageComposition(
        markdown=page_markdown,
        merge_order=page_merge_order,
        composition_metadata=metadata,
        continuation_page=True,
        start_mid_paragraph=start_mid,
    )


def _sample_flag(spec: Any, rng: Any) -> bool:
    """``{p: ...}`` spec or a bare probability; unset is ``False``."""
    if spec is None:
        return False
    if isinstance(spec, bool):
        return spec
    if isinstance(spec, (int, float)):
        return rng.random() < float(spec)
    from src.generator.distribution_profile import sample_value

    return bool(sample_value(spec, rng))


def _is_heading(chunk: str) -> bool:
    return chunk.lstrip().startswith("#") and "\n" not in chunk.strip()


def _parse_sections(markdown_text: str, merge_order: List[str]) -> Optional[List[_Section]]:
    """Split the page into sections of (block, type) pairs, dropping the `#` title.

    Returns ``None`` when merge_order does not list one type per block.
    """
    chunks = [chunk for chunk in markdown_text.strip().split("\n\n") if chunk.strip()]
    if chunks and chunks[0].startswith("# "):
        chunks = chunks[1:]
    blocks = [chunk for chunk in chunks if not _is_heading(chunk)]
    if not blocks or len(blocks) != len(merge_order):
        return None

    types = iter(merge_order)
    sections: List[_Section] = []
    current = _Section(heading=None)
    for chunk in chunks:
        if _is_heading(chunk):
            if current.heading is not None or current.blocks:
                sections.append(current)
            current = _Section(heading=chunk)
        else:
            current.blocks.append((chunk, next(types)))
    sections.append(current)
    return sections


def _candidate_starts(sections: List[_Section]) -> List[int]:
    """Indices of the first kept section: a later section that still has blocks."""
    later = [
        index
        for index in range(1, len(sections))
        if any(section.blocks for section in sections[index:])
    ]
    return later or [0]


def _sentence_tails(paragraph: str) -> List[str]:
    """Every proper tail of ``paragraph`` that starts at a sentence boundary.

    Tails that would open with block markup (``1. ``, ``- ``, ``# ``, ...) are
    skipped: at the start of a chunk they would render as a list or heading.
    """
    sentences = [part for part in _SENTENCE_BREAK_RE.split(paragraph.strip()) if part]
    tails = (" ".join(sentences[cut:]) for cut in range(1, len(sentences)))
    return [tail for tail in tails if not _BLOCK_MARKUP_PREFIX_RE.match(tail)]


def _mid_paragraph_cuts(sections: List[_Section], start: int) -> List[str]:
    """Which paragraphs next to the break before ``sections[start]`` can be cut.

    ``"own"``: the kept section's first block; ``"previous"``: the last block
    of the last dropped section.
    """
    cuts: List[str] = []
    first_blocks = sections[start].blocks
    if first_blocks and first_blocks[0][1] == "paragraph" and _sentence_tails(first_blocks[0][0]):
        cuts.append("own")
    if start > 0:
        previous_blocks = sections[start - 1].blocks
        if (
            previous_blocks
            and previous_blocks[-1][1] == "paragraph"
            and _sentence_tails(previous_blocks[-1][0])
        ):
            cuts.append("previous")
    return cuts


def _cut_mid_paragraph(sections: List[_Section], start: int, rng: Any) -> List[_Section]:
    kept = [_Section(section.heading, list(section.blocks)) for section in sections[start:]]
    cut = rng.choice(_mid_paragraph_cuts(sections, start))
    if cut == "own":
        # The section heading and the paragraph's opening stay on the previous page.
        paragraph = kept[0].blocks[0][0]
        kept[0].heading = None
        kept[0].blocks[0] = (rng.choice(_sentence_tails(paragraph)), "paragraph")
    else:
        paragraph = sections[start - 1].blocks[-1][0]
        kept.insert(0, _Section(heading=None, blocks=[(rng.choice(_sentence_tails(paragraph)), "paragraph")]))
    return kept
