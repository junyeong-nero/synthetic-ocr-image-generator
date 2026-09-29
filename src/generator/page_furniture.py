"""Page furniture: running header, footer text and page number in the margins.

Real document pages carry a running header (short document title,
organisation name, date, often with a rule under it), footer text and a page
number (DocLayNet: 58,022 Page-header and 70,878 Page-footer instances on
80,863 pages). They are drawn by the HTML renderers inside the page margins
but are **not** part of ``GT_markdown`` (OmniDocBench convention: page
headers/footers/numbers are ignored when scoring). Each sample records what
was drawn in the ``page_furniture`` metadata column so evaluation can drop
predicted lines that transcribe it (``src/evaluation/page_furniture.py``).

Three ``page`` profile keys control it, all off unless a profile sets them:

- ``page.header`` (p): draw a running header in the top margin.
- ``page.footer`` (p): draw footer text in the bottom margin.
- ``page.page_number``: one of ``PAGE_NUMBER_STYLES`` (``none`` draws nothing).

With furniture, the renderer wraps the page in a sheet container of the sheet
height (page width x ``page.aspect_ratio``) whose header sits in the top
margin and whose footer line (footer text + page number) is pinned to the
bottom margin, so it lands at the bottom of the sheet rather than under the
last line of content. The container uses ``min-height``: content taller than
one sheet still grows the page, so ``sheet_overflow_ratio`` keeps detecting
overflow and ``fit_markdown_to_sheet`` trims as before. The header and footer
use the space the margins already reserve (an HWP page's 35mm top margin
includes its 15mm header zone); only a margin too small to hold a line of
furniture text is enlarged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover
    from src.generator.data_provider import DataProvider
    from src.generator.markdown_render_utils import MarkdownStyle

FURNITURE_KEYS = ("header", "footer", "page_number")
PAGE_NUMBER_STYLES = ("none", "dashed", "plain", "of_total", "korean", "english")

# Running headers and footers are short: long titles are cut at a word boundary.
_MAX_PART_CHARS = 24
_MAX_ORG_CHARS = 16
# Page numbers of a continuation page, and how many pages a document may
# have after the current one ("3 / 12").
_MAX_CONTINUATION_PAGE = 40
_MAX_EXTRA_PAGES = 30
# Layout in CSS px.
_LINE_HEIGHT = 1.25
_EDGE_GAP_PX = 4
_RULE_GAP_PX = 3

_HEADER_LAYOUTS = (
    "title_center",
    "org_left_title_right",
    "title_left_date_right",
    "title_right",
    "org_left_date_right",
)
_HEADER_LAYOUT_WEIGHTS = (0.3, 0.25, 0.2, 0.15, 0.1)
_FOOTER_KINDS = ("org", "copyright", "code", "confidential")
_FOOTER_KIND_WEIGHTS = (0.35, 0.25, 0.2, 0.2)
_CONFIDENTIAL = {"ko": "대외비", "ja": "社外秘"}
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


@dataclass(frozen=True)
class FurnitureLine:
    """One line of furniture text in left / centre / right slots."""

    left: str = ""
    center: str = ""
    right: str = ""

    def parts(self) -> List[str]:
        return [part for part in (self.left, self.center, self.right) if part]


@dataclass(frozen=True)
class PageFurniture:
    header: FurnitureLine = field(default_factory=FurnitureLine)
    header_rule: bool = False
    # Footer line: footer text and page number, positioned in slots.
    footer: FurnitureLine = field(default_factory=FurnitureLine)
    footer_text: str = ""
    page_number: str = ""
    # Furniture font size relative to the body font.
    font_scale: float = 0.85
    # Header top edge / footer bottom edge as a fraction of the top / bottom margin.
    header_offset: float = 0.5
    footer_offset: float = 0.42

    def has_content(self) -> bool:
        return bool(self.header.parts() or self.footer.parts())

    def metadata(self) -> Dict[str, str]:
        """``page_furniture`` column: strings only, ``""`` for an absent part.

        Header parts placed separately on the line (e.g. organisation left,
        title right) are joined with a newline.
        """
        return {
            "header": "\n".join(self.header.parts()),
            "footer": self.footer_text,
            "page_number": self.page_number,
        }


def furniture_requested(page: Mapping[str, Any]) -> bool:
    """Whether the sampled ``page`` settings use any furniture key."""
    return any(key in page for key in FURNITURE_KEYS)


def format_page_number(style: str, number: int, total: int, *, lang: str) -> str:
    """Render a page number; ``""`` for ``none`` or an unknown style."""
    if style == "dashed":
        return f"- {number} -"
    if style == "plain":
        return str(number)
    if style == "of_total":
        return f"{number} / {total}"
    if style == "korean":
        return f"{number}쪽" if lang == "ko" else str(number)
    if style == "english":
        return f"Page {number}"
    return ""


def sample_page_furniture(
    page: Mapping[str, Any],
    data: "DataProvider",
    rng: Any,
    *,
    lang: str,
    continuation: bool,
) -> PageFurniture:
    """Sample one page's furniture from its sampled ``page`` settings.

    A document's first page is page 1; a continuation page (see
    ``page_composition``) gets a later page number.
    """
    show_header = _flag(page.get("header"), rng)
    show_footer = _flag(page.get("footer"), rng)
    number_style = str(page.get("page_number") or "none")

    header = _sample_header(data, rng, lang) if show_header else FurnitureLine()
    footer_text = _sample_footer_text(data, rng, lang) if show_footer else ""
    page_number = ""
    if number_style in PAGE_NUMBER_STYLES and number_style != "none":
        number = rng.randint(2, _MAX_CONTINUATION_PAGE) if continuation else 1
        total = rng.randint(max(number, 2), number + _MAX_EXTRA_PAGES)
        page_number = format_page_number(number_style, number, total, lang=lang)

    return PageFurniture(
        header=header,
        header_rule=bool(header.parts()) and rng.random() < 0.5,
        footer=_footer_line(footer_text, page_number, rng),
        footer_text=footer_text,
        page_number=page_number,
        font_scale=rng.uniform(0.8, 0.95),
        header_offset=rng.uniform(0.42, 0.58),
        footer_offset=rng.uniform(0.35, 0.5),
    )


def apply_page_furniture(
    style: "MarkdownStyle",
    furniture: PageFurniture,
    aspect_ratio: Optional[float],
) -> None:
    """Attach ``furniture`` and the sheet height to ``style`` for the HTML renderer.

    Empty furniture leaves ``style`` untouched (legacy page, no container).
    Without an aspect ratio the sheet ends where its content ends.
    """
    if not furniture.has_content():
        return
    style.page_furniture = furniture
    page_width = style.margin_left + style.content_width + style.margin_right
    style.sheet_height = int(round(page_width * float(aspect_ratio))) if aspect_ratio else None

    line_px = _line_px(style, furniture)
    if furniture.header.parts():
        header_px = line_px + (_RULE_GAP_PX + 1 if furniture.header_rule else 0)
        style.margin_top = max(style.margin_top, header_px + 2 * _EDGE_GAP_PX)
    if furniture.footer.parts():
        style.margin_bottom = max(style.margin_bottom, line_px + 2 * _EDGE_GAP_PX)


def page_sheet_css(style: "MarkdownStyle") -> str:
    """CSS for the sheet container and its furniture; ``""`` without furniture."""
    furniture: Optional[PageFurniture] = getattr(style, "page_furniture", None)
    if furniture is None:
        return ""
    page_width = style.margin_left + style.content_width + style.margin_right
    font_px = _font_px(style, furniture)
    line_px = _line_px(style, furniture)
    header_px = line_px + (_RULE_GAP_PX + 1 if furniture.header_rule else 0)
    header_top = _clamp(
        round(style.margin_top * furniture.header_offset),
        _EDGE_GAP_PX,
        style.margin_top - header_px - _EDGE_GAP_PX,
    )
    footer_bottom = _clamp(
        round(style.margin_bottom * furniture.footer_offset),
        _EDGE_GAP_PX,
        style.margin_bottom - line_px - _EDGE_GAP_PX,
    )
    min_height = f"  min-height: {style.sheet_height}px;\n" if style.sheet_height else ""
    rule_color = tuple(style.rule_color or style.text_color)
    header_rule = (
        f"  padding-bottom: {_RULE_GAP_PX}px;\n  border-bottom: 1px solid rgb{rule_color};\n"
        if furniture.header_rule
        else ""
    )
    return f"""
.page-sheet {{
  position: relative;
  width: {page_width}px;
{min_height}  background: rgb{style.background_color};
}}
.page-furniture {{
  position: absolute;
  left: {style.margin_left}px;
  right: {style.margin_right}px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  column-gap: 1em;
  align-items: end;
  font-family: 'RenderFont', sans-serif;
  font-size: {font_px}px;
  line-height: {_LINE_HEIGHT};
  color: rgb{style.text_color};
  word-break: keep-all;
}}
.page-furniture .pf-left {{ text-align: left; }}
.page-furniture .pf-center {{ text-align: center; white-space: nowrap; }}
.page-furniture .pf-right {{ text-align: right; }}
.page-header {{
  top: {header_top}px;
{header_rule}}}
.page-footer {{
  bottom: {footer_bottom}px;
}}
"""


def page_sheet_html(style: "MarkdownStyle", content_html: str) -> str:
    """Wrap the ``.markdown-body`` element in the sheet container, if any."""
    furniture: Optional[PageFurniture] = getattr(style, "page_furniture", None)
    if furniture is None:
        return content_html
    lines = ['<div class="page-sheet">']
    if furniture.header.parts():
        lines.append(f"  {_line_html('page-header', furniture.header)}")
    lines.append(f"  {content_html}")
    if furniture.footer.parts():
        lines.append(f"  {_line_html('page-footer', furniture.footer)}")
    lines.append("  </div>")
    return "\n".join(lines)


def _line_html(css_class: str, line: FurnitureLine) -> str:
    slots = "".join(
        f'<span class="pf-{slot}">{escape(text)}</span>'
        for slot, text in (("left", line.left), ("center", line.center), ("right", line.right))
    )
    return f'<div class="page-furniture {css_class}">{slots}</div>'


def _font_px(style: "MarkdownStyle", furniture: PageFurniture) -> int:
    return max(8, int(round(style.body_font_size * furniture.font_scale)))


def _line_px(style: "MarkdownStyle", furniture: PageFurniture) -> int:
    return int(math.ceil(_font_px(style, furniture) * _LINE_HEIGHT))


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(int(value), upper))


def _flag(spec: Any, rng: Any) -> bool:
    """A sampled ``{p: ...}`` value (bool) or a bare probability."""
    if spec is None:
        return False
    if isinstance(spec, bool):
        return spec
    if isinstance(spec, (int, float)):
        return rng.random() < float(spec)
    return bool(spec)


def _shorten(text: str, max_chars: int) -> str:
    """Collapse whitespace and cut at a word boundary (no ellipsis)."""
    normalized = " ".join(str(text).split()).strip(" -_:,;.")
    if len(normalized) <= max_chars:
        return normalized
    kept = ""
    for word in normalized.split(" "):
        candidate = f"{kept} {word}".strip()
        if len(candidate) > max_chars:
            break
        kept = candidate
    return kept or normalized[:max_chars].rstrip()


def _sample_date(data: "DataProvider", rng: Any, lang: str) -> str:
    year, month, day = (int(value) for value in data.date("%Y-%m-%d").split("-"))
    if lang == "ko":
        formats = [f"{year}. {month}. {day}.", f"{year}년 {month}월 {day}일", f"{year}-{month:02d}-{day:02d}"]
    elif lang == "ja":
        formats = [f"{year}年{month}月{day}日", f"{year}/{month:02d}/{day:02d}"]
    else:
        formats = [
            f"{year}-{month:02d}-{day:02d}",
            f"{_MONTHS[month - 1]} {day}, {year}",
            f"{day} {_MONTHS[month - 1]} {year}",
        ]
    return rng.choice(formats)


def _sample_header(data: "DataProvider", rng: Any, lang: str) -> FurnitureLine:
    layout = rng.choices(_HEADER_LAYOUTS, weights=_HEADER_LAYOUT_WEIGHTS, k=1)[0]
    if layout == "title_center":
        return FurnitureLine(center=_shorten(data.title(), _MAX_PART_CHARS))
    if layout == "title_right":
        return FurnitureLine(right=_shorten(data.title(), _MAX_PART_CHARS))
    if layout == "org_left_title_right":
        return FurnitureLine(
            left=_shorten(data.company(), _MAX_ORG_CHARS),
            right=_shorten(data.title(), _MAX_PART_CHARS),
        )
    if layout == "title_left_date_right":
        return FurnitureLine(
            left=_shorten(data.title(), _MAX_PART_CHARS),
            right=_sample_date(data, rng, lang),
        )
    return FurnitureLine(
        left=_shorten(data.company(), _MAX_ORG_CHARS),
        right=_sample_date(data, rng, lang),
    )


def _sample_footer_text(data: "DataProvider", rng: Any, lang: str) -> str:
    kind = rng.choices(_FOOTER_KINDS, weights=_FOOTER_KIND_WEIGHTS, k=1)[0]
    if kind == "org":
        return _shorten(data.company(), _MAX_PART_CHARS)
    if kind == "confidential":
        return _CONFIDENTIAL.get(lang, "Confidential")
    year = data.date("%Y")
    if kind == "copyright":
        return _shorten(f"© {year} {_shorten(data.company(), _MAX_ORG_CHARS)}", _MAX_PART_CHARS)
    serial = rng.randint(1, 999)
    if lang == "ko":
        return _shorten(f"{data.department()}-{year}-{serial:03d}", _MAX_PART_CHARS)
    if lang == "ja":
        return f"文書番号 {year}-{serial:03d}"
    return f"Doc. No. {year}-{serial:03d}"


def _footer_line(footer_text: str, page_number: str, rng: Any) -> FurnitureLine:
    if footer_text and page_number:
        if rng.random() < 0.5:
            return FurnitureLine(left=footer_text, center=page_number)
        return FurnitureLine(left=footer_text, right=page_number)
    if page_number:
        return FurnitureLine(center=page_number) if rng.random() < 0.75 else FurnitureLine(right=page_number)
    if footer_text:
        return FurnitureLine(left=footer_text) if rng.random() < 0.5 else FurnitureLine(center=footer_text)
    return FurnitureLine()
