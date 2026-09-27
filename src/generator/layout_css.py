"""CSS for ``typography.table_style`` variants and ``page.columns`` layout.

Both knobs are consumed only by the HTML/Playwright renderers
(``HtmlMarkdownRenderer._build_html_document`` in ``markdown_renderers.py``);
the PIL ``MarkdownRenderer`` ignores them. Kept in a separate module per
AGENTS.md ("keep files focused ... rather than growing ... markdown_renderers.py").

Table styles build on the caller's ``rule_css`` (the ink colour, or the light
legacy default) so every style keeps the same "printed rules match the text
ink" behaviour as ``hr`` (see ``profile_application.plan_profile_render``'s
``rule_color``). None of these rules use ``!important``: python-markdown
emits an inline ``style="text-align: right;"`` on numeric columns (the
``---:`` separator, see ``table_generator.py``), and inline styles must keep
winning over these class-level rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from src.generator.markdown_render_utils import MarkdownStyle

# Valid ``typography.table_style`` values. Anything else (including unset)
# falls back to "web", the legacy look, byte-for-byte.
TABLE_STYLES = ("web", "grid", "header_shaded", "booktabs", "borderless")

# "web" keeps the original web-CSS padding; every other style uses tighter,
# printed-document padding.
_WEB_PADDING = "8px 12px"
_PRINTED_PADDING = "5px 8px"

# Fallback column gap (CSS px) when a MarkdownStyle is built directly with
# columns=2 and no explicit column_gap (profiles always set one).
_DEFAULT_COLUMN_GAP_PX = 24


def table_style_css(style: "MarkdownStyle", rule_css: str) -> str:
    """CSS block for ``.markdown-body table`` / ``th`` / ``td``, keyed by ``style.table_style``."""
    table_style = style.table_style or "web"
    if table_style not in TABLE_STYLES:
        table_style = "web"
    return _TABLE_STYLE_BUILDERS[table_style](rule_css)


def _web_css(rule_css: str) -> str:
    """Legacy look: light/ink borders, light header shading, zebra stripes."""
    return f"""
.markdown-body th, .markdown-body td {{
  border: 1px solid {rule_css};
  text-align: left;
  padding: {_WEB_PADDING};
  vertical-align: top;
  overflow-wrap: break-word;
  word-break: normal;
}}
.markdown-body th {{
  background: rgba(0, 0, 0, 0.06);
  font-weight: 600;
}}
.markdown-body tbody tr:nth-child(even) td {{
  background: rgba(0, 0, 0, 0.025);
}}
"""


def _grid_css(rule_css: str) -> str:
    """Solid ink borders on every cell, no zebra, bold centred header."""
    return f"""
.markdown-body th, .markdown-body td {{
  border: 1px solid {rule_css};
  text-align: left;
  padding: {_PRINTED_PADDING};
  vertical-align: top;
  overflow-wrap: break-word;
  word-break: normal;
}}
.markdown-body th {{
  font-weight: 700;
  text-align: center;
}}
"""


def _header_shaded_css(rule_css: str) -> str:
    """No vertical rules; a shaded, bold header row; horizontal row rules."""
    return f"""
.markdown-body th, .markdown-body td {{
  border: none;
  border-top: 1px solid {rule_css};
  border-bottom: 1px solid {rule_css};
  text-align: left;
  padding: {_PRINTED_PADDING};
  vertical-align: top;
  overflow-wrap: break-word;
  word-break: normal;
}}
.markdown-body thead th {{
  background: rgba(0, 0, 0, 0.12);
  font-weight: 700;
}}
"""


def _booktabs_css(rule_css: str) -> str:
    """Thick top/bottom table rules plus a header rule; no vertical lines."""
    return f"""
.markdown-body table {{
  border-top: 2px solid {rule_css};
  border-bottom: 2px solid {rule_css};
}}
.markdown-body th, .markdown-body td {{
  border: none;
  text-align: left;
  padding: {_PRINTED_PADDING};
  vertical-align: top;
  overflow-wrap: break-word;
  word-break: normal;
}}
.markdown-body thead th {{
  border-bottom: 1px solid {rule_css};
  font-weight: 700;
}}
"""


def _borderless_css(rule_css: str) -> str:
    """No table borders at all, except a single rule under the header row."""
    return f"""
.markdown-body th, .markdown-body td {{
  border: none;
  text-align: left;
  padding: {_PRINTED_PADDING};
  vertical-align: top;
  overflow-wrap: break-word;
  word-break: normal;
}}
.markdown-body thead th {{
  border-bottom: 1px solid {rule_css};
  font-weight: 600;
}}
"""


_TABLE_STYLE_BUILDERS = {
    "web": _web_css,
    "grid": _grid_css,
    "header_shaded": _header_shaded_css,
    "booktabs": _booktabs_css,
    "borderless": _borderless_css,
}


def columns_css(style: "MarkdownStyle") -> str:
    """CSS multi-column layout for ``.markdown-body``, or ``""`` for one column.

    Content flows column-major (top-to-bottom in column 1, then column 2, ...)
    in DOM order, so the rendered reading order matches ``GT_markdown``
    unchanged. ``column-gap`` is subtracted from the existing content width
    by the browser's column layout, not added to the page, so page width
    (``margin_left + content_width + margin_right``) stays constant. The
    document title (``h1``) spans every column; tables and image figures
    already set ``break-inside: avoid``, which also applies to column
    fragmentation, so they are never split across columns.
    """
    columns = int(style.columns or 1)
    if columns < 2:
        return ""
    gap = style.column_gap if style.column_gap else _DEFAULT_COLUMN_GAP_PX
    return f"""
.markdown-body {{
  column-count: {columns};
  column-gap: {int(gap)}px;
}}
.markdown-body h1 {{
  column-span: all;
}}
"""
