"""Korean document surface conventions layered onto composed markdown.

Real Korean documents rarely look like plain web markdown (`# title`,
`## section`, `- item`, `1. item`). Reports number headings (`1.`, `Ⅰ.`,
`가.`), 개조식 (itemized) lists use `□ / ○ / ·` bullets and `1) / 가) / ①`
numbered markers, and administrative/policy text is written as
`제N조(제목) ...` articles with `①②` clauses.

This module renders those surface forms as **plain text**. Every marker here
must parse identically under python-markdown (which renders the training
image, see `markdown_renderers.py`) and mistune (which builds `GT_json`, see
`utils.markdown_to_json_ast`) -- otherwise the image and the two ground-truth
representations would disagree about structure. Concretely this rules out:

- an unescaped `1)`: mistune (CommonMark-ish) parses a digit followed by `)`
  as an ordered-list marker, but python-markdown does not. Escaping the
  parenthesis (`1\\)`) keeps both parsers reading it as literal text.
- any marker that starts a line with `-`, `+`, `*`, `#`, `>`, backtick,
  `|`, `$$`, `![`, or a bare `digit.` -- both parsers treat those as block
  syntax, so the corresponding block would render as a real HTML list/quote
  in the image while GT_json would need to encode it the same way (which it
  already does for the legacy `- item` / `1. item` blocks; the point here is
  that the *new* Korean-style markers stay outside that syntax entirely).

`tests/generator/test_document_conventions.py` pins the two-parser agreement
for every marker this module emits.

Design notes (documented since the task brief leaves them open):

- All three knobs (`heading_numbering`, `list_style`, whether law articles
  are used) are sampled **once per document** from the seeded global
  `random`, then applied consistently -- a single real document keeps one
  numbering convention throughout, it does not mix heading styles paragraph
  by paragraph.
- `korean_admin` heading numbering samples one marker pool (roman numerals /
  arabic / 가나다) per document rather than modelling true multi-level
  nesting, because `DocumentComposer` only ever emits one heading depth
  (`## `). This still produces the three marker glyphs the brief calls out.
- `law_articles` rewrites an entire `numbered_list` block into one
  `제N조(제목) ...` paragraph followed by `①②...` clause lines (the block's
  first item becomes the article body, the remaining items become clauses).
  The article title is drawn from a small pool of common Korean statute
  section titles (목적/정의/...) rather than derived from corpus text, since
  corpus paragraphs carry no legal-drafting vocabulary. Article numbers
  (`제1조`, `제2조`, ...) increment across every `numbered_list` block in one
  document.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

from src.generator.data_provider import DataProvider

HEADING_NUMBERING_CHOICES = ("none", "arabic", "korean_admin", "roman")
LIST_STYLE_CHOICES = ("markdown", "korean_admin")

# Heading marker pools.
_ROMAN_UNICODE = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ", "Ⅺ", "Ⅻ"]
_ROMAN_ASCII = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
_KOREAN_CONSONANT_SYLLABLES = [
    "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하",
]
_CIRCLED_DIGITS = [
    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
    "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳",
]
_BULLET_MARKERS = ("□", "○", "·")
_NOTE_MARKER = "※"
_NOTE_PROBABILITY = 0.3
_LAW_ARTICLE_TITLES = (
    "목적", "정의", "적용범위", "책임", "역할", "절차",
    "기준", "관리", "운영", "시행일", "위임", "준수사항",
)

_BULLET_ITEM_RE = re.compile(r"^-\s+(?P<body>\S.*)$")
_NUMBERED_ITEM_RE = re.compile(r"^\d+\.\s+(?P<body>\S.*)$")

_NUMBERED_MARKER_STYLES = ("arabic_paren", "korean_consonant_paren", "circled_digit")


def _pool_marker(pool: List[str], index: int) -> str:
    """1-based lookup into a marker pool; beyond the pool, fall back to the
    plain number so numbering never crashes on unusually long documents."""
    if 1 <= index <= len(pool):
        return pool[index - 1]
    return str(index)


def _extract_items(markdown_text: str, pattern: re.Pattern[str]) -> List[str]:
    """Split a plain `bullet_list` / `numbered_list` block back into item
    texts. Returns `[]` (caller keeps the block unchanged) if any line does
    not match the expected prefix, so an unexpected block shape is never
    silently mangled."""
    items: List[str] = []
    for line in markdown_text.splitlines():
        match = pattern.match(line.strip())
        if not match:
            return []
        items.append(match.group("body").strip())
    return items


def _sample_choice(spec: Any, default: str, allowed: tuple) -> str:
    if spec is None:
        return default
    # Imported lazily, mirroring `DocumentComposer._sample_content`, to keep
    # this module usable without pulling in the profile machinery.
    from src.generator.distribution_profile import sample_value

    value = str(sample_value(spec, random))
    return value if value in allowed else default


def _sample_probability_flag(spec: Any) -> bool:
    if spec is None:
        return False
    from src.generator.distribution_profile import sample_value

    sampled = sample_value(spec, random)
    if isinstance(sampled, bool):
        return sampled
    try:
        probability = float(sampled)
    except (TypeError, ValueError):
        return False
    return random.random() < probability


@dataclass(frozen=True)
class ConventionsPlan:
    """Per-document styling choices, also recorded in composition metadata."""

    heading_numbering: str
    list_style: str
    law_articles_used: bool

    def to_dict(self) -> dict:
        return {
            "heading_numbering": self.heading_numbering,
            "list_style": self.list_style,
            "law_articles_used": self.law_articles_used,
        }


class DocumentConventions:
    """Applies heading numbering, 개조식 list markers and law-article
    formatting to one composed document.

    `content.heading_numbering` / `content.list_style` / `content.law_articles`
    resolve to :attr:`plan` once, in the constructor, from the seeded global
    `random` (these are choice/probability specs, not the scalar floats
    `DocumentComposer._sample_content` samples). `korean_admin` and the law
    article style are Korean-only; outside `lang == "ko"` they fall back to
    `arabic` (headings) / `markdown` (lists / no articles).
    """

    def __init__(
        self,
        *,
        lang: str,
        document_shape: str,
        content_specs: Mapping[str, Any],
        data: DataProvider,
        clip_text: Callable[[str, int], str],
    ) -> None:
        self._data = data
        self._clip_text = clip_text
        self._is_korean = str(lang).lower() == "ko"

        requested_heading = _sample_choice(
            content_specs.get("heading_numbering"), "none", HEADING_NUMBERING_CHOICES
        )
        requested_list_style = _sample_choice(
            content_specs.get("list_style"), "markdown", LIST_STYLE_CHOICES
        )
        requested_law_articles = _sample_probability_flag(content_specs.get("law_articles"))

        effective_heading = requested_heading
        if requested_heading == "korean_admin" and not self._is_korean:
            effective_heading = "arabic"
        effective_list_style = requested_list_style
        if requested_list_style == "korean_admin" and not self._is_korean:
            effective_list_style = "markdown"

        self._heading_marker_style = self._pick_heading_marker_style(effective_heading)
        self._numbered_marker_style: Optional[str] = (
            random.choice(_NUMBERED_MARKER_STYLES)
            if effective_list_style == "korean_admin"
            else None
        )

        law_articles_used = bool(
            requested_law_articles and self._is_korean and document_shape == "policy_document"
        )

        self.plan = ConventionsPlan(
            heading_numbering=effective_heading,
            list_style=effective_list_style,
            law_articles_used=law_articles_used,
        )
        self._heading_index = 0
        self._article_index = 0

    @staticmethod
    def _pick_heading_marker_style(effective_heading: str) -> Optional[str]:
        if effective_heading == "arabic":
            return random.choice(["arabic_plain", "arabic_dotted"])
        if effective_heading == "roman":
            return "roman_ascii"
        if effective_heading == "korean_admin":
            return random.choice(["roman_unicode", "arabic_plain", "korean_consonant"])
        return None

    def style_heading(self, text: str) -> str:
        """Number one `## ` section heading in document order."""
        self._heading_index += 1
        marker = self._heading_marker(self._heading_marker_style, self._heading_index)
        if marker is None:
            return text
        return f"{marker} {text}"

    @staticmethod
    def _heading_marker(style: Optional[str], index: int) -> Optional[str]:
        if style is None:
            return None
        if style == "arabic_plain":
            return f"{index}."
        if style == "arabic_dotted":
            return f"1.{index}"
        if style == "roman_ascii":
            return f"{_pool_marker(_ROMAN_ASCII, index)}."
        if style == "roman_unicode":
            return f"{_pool_marker(_ROMAN_UNICODE, index)}."
        if style == "korean_consonant":
            return f"{_pool_marker(_KOREAN_CONSONANT_SYLLABLES, index)}."
        return None

    def style_block(self, block_type: str, markdown_text: str) -> Optional[str]:
        """Return replacement markdown for a block, or `None` to keep it
        unchanged. The block type itself never changes -- only its literal
        text -- so `merge_order` / `block_type_counts` bookkeeping in
        `DocumentComposer` stays untouched."""
        if block_type == "numbered_list" and self.plan.law_articles_used:
            styled = self._law_article_text(markdown_text)
            if styled is not None:
                return styled
        if block_type == "bullet_list" and self.plan.list_style == "korean_admin":
            return self._korean_bullet_text(markdown_text)
        if (
            block_type == "numbered_list"
            and self.plan.list_style == "korean_admin"
            and self._numbered_marker_style is not None
        ):
            return self._korean_numbered_text(markdown_text)
        return None

    def _korean_bullet_text(self, markdown_text: str) -> Optional[str]:
        items = _extract_items(markdown_text, _BULLET_ITEM_RE)
        if not items:
            return None
        marker = random.choice(_BULLET_MARKERS)
        lines = [f"{marker} {item}" for item in items]
        self._maybe_append_note(lines)
        return "\n".join(lines)

    def _korean_numbered_text(self, markdown_text: str) -> Optional[str]:
        items = _extract_items(markdown_text, _NUMBERED_ITEM_RE)
        if not items:
            return None
        lines = [self._numbered_marker_line(index, item) for index, item in enumerate(items, start=1)]
        self._maybe_append_note(lines)
        return "\n".join(lines)

    def _numbered_marker_line(self, index: int, item: str) -> str:
        if self._numbered_marker_style == "arabic_paren":
            # Escaped: both python-markdown and mistune otherwise disagree
            # about an unescaped "N)" (see module docstring).
            return f"{index}\\) {item}"
        if self._numbered_marker_style == "korean_consonant_paren":
            return f"{_pool_marker(_KOREAN_CONSONANT_SYLLABLES, index)}) {item}"
        if index <= len(_CIRCLED_DIGITS):
            return f"{_pool_marker(_CIRCLED_DIGITS, index)} {item}"
        return f"{index}\\) {item}"

    def _maybe_append_note(self, lines: List[str]) -> None:
        if random.random() >= _NOTE_PROBABILITY:
            return
        note = self._clip_text(self._data.sentence(), 80)
        if note:
            lines.append(f"{_NOTE_MARKER} {note}")

    def _law_article_text(self, markdown_text: str) -> Optional[str]:
        items = _extract_items(markdown_text, _NUMBERED_ITEM_RE)
        if not items:
            return None
        self._article_index += 1
        title = random.choice(_LAW_ARTICLE_TITLES)
        lines = [f"제{self._article_index}조({title}) {items[0]}"]
        for clause_index, item in enumerate(items[1:], start=1):
            if clause_index <= len(_CIRCLED_DIGITS):
                marker = _pool_marker(_CIRCLED_DIGITS, clause_index)
            else:
                marker = f"{clause_index}\\)"
            lines.append(f"{marker} {item}")
        return "\n".join(lines)
