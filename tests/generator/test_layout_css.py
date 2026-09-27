"""``typography.table_style`` and ``page.columns`` CSS (src/generator/layout_css.py)."""

import re

import pytest

from src.generator.layout_css import TABLE_STYLES, columns_css, table_style_css
from src.generator.markdown_render_utils import MarkdownStyle


def _css_block(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, f"selector {selector!r} not found in:\n{css}"
    return match.group(1)


def test_default_table_style_matches_legacy_web_css() -> None:
    css = table_style_css(MarkdownStyle(), "rgba(0, 0, 0, 0.25)")

    cell_rule = _css_block(css, ".markdown-body th, .markdown-body td")
    assert "border: 1px solid rgba(0, 0, 0, 0.25)" in cell_rule
    assert "padding: 8px 12px" in cell_rule
    th_rule = _css_block(css, ".markdown-body th")
    assert "background: rgba(0, 0, 0, 0.06)" in th_rule
    assert "font-weight: 600" in th_rule
    assert "tbody tr:nth-child(even)" in css


def test_grid_style_has_solid_borders_no_zebra_bold_centered_header() -> None:
    css = table_style_css(MarkdownStyle(table_style="grid"), "rgb(10, 10, 10)")

    cell_rule = _css_block(css, ".markdown-body th, .markdown-body td")
    assert "border: 1px solid rgb(10, 10, 10)" in cell_rule
    th_rule = _css_block(css, ".markdown-body th")
    assert "font-weight: 700" in th_rule
    assert "text-align: center" in th_rule
    assert "tbody tr:nth-child(even)" not in css
    assert "background" not in th_rule  # no header shading


def test_header_shaded_style_has_no_vertical_borders_and_shaded_header() -> None:
    css = table_style_css(MarkdownStyle(table_style="header_shaded"), "rgb(20, 20, 20)")

    cell_rule = _css_block(css, ".markdown-body th, .markdown-body td")
    assert "border: none" in cell_rule
    assert "border-top: 1px solid rgb(20, 20, 20)" in cell_rule
    assert "border-bottom: 1px solid rgb(20, 20, 20)" in cell_rule
    thead_rule = _css_block(css, ".markdown-body thead th")
    assert "background: rgba(0, 0, 0, 0.12)" in thead_rule


def test_booktabs_style_has_top_bottom_and_header_rules_no_vertical_lines() -> None:
    css = table_style_css(MarkdownStyle(table_style="booktabs"), "rgb(0, 0, 0)")

    table_rule = _css_block(css, ".markdown-body table")
    assert "border-top: 2px solid rgb(0, 0, 0)" in table_rule
    assert "border-bottom: 2px solid rgb(0, 0, 0)" in table_rule
    cell_rule = _css_block(css, ".markdown-body th, .markdown-body td")
    assert "border: none" in cell_rule
    thead_rule = _css_block(css, ".markdown-body thead th")
    assert "border-bottom: 1px solid rgb(0, 0, 0)" in thead_rule


def test_borderless_style_has_header_underline_only() -> None:
    css = table_style_css(MarkdownStyle(table_style="borderless"), "rgb(5, 5, 5)")

    cell_rule = _css_block(css, ".markdown-body th, .markdown-body td")
    assert "border: none" in cell_rule
    assert "border-top" not in cell_rule
    assert ".markdown-body table {" not in css  # no table-level border rule
    thead_rule = _css_block(css, ".markdown-body thead th")
    assert "border-bottom: 1px solid rgb(5, 5, 5)" in thead_rule


def test_unknown_table_style_falls_back_to_web() -> None:
    css = table_style_css(MarkdownStyle(table_style="not-a-real-style"), "rgb(0, 0, 0)")

    th_rule = _css_block(css, ".markdown-body th")
    assert "background: rgba(0, 0, 0, 0.06)" in th_rule


@pytest.mark.parametrize("table_style", list(TABLE_STYLES))
def test_no_style_uses_important(table_style: str) -> None:
    # Inline text-align from the markdown `---:` numeric-alignment separator
    # must always win; a class rule using !important would break that.
    css = table_style_css(MarkdownStyle(table_style=table_style), "rgb(0, 0, 0)")

    assert "!important" not in css


def test_padding_is_tighter_for_printed_styles_than_web() -> None:
    web_css = table_style_css(MarkdownStyle(table_style="web"), "rgb(0, 0, 0)")
    grid_css = table_style_css(MarkdownStyle(table_style="grid"), "rgb(0, 0, 0)")

    assert "padding: 8px 12px" in _css_block(web_css, ".markdown-body th, .markdown-body td")
    assert "padding: 5px 8px" in _css_block(grid_css, ".markdown-body th, .markdown-body td")


def test_single_column_produces_no_css() -> None:
    assert columns_css(MarkdownStyle(columns=1)) == ""
    assert columns_css(MarkdownStyle()) == ""


def test_two_columns_sets_column_count_gap_and_h1_span() -> None:
    css = columns_css(MarkdownStyle(columns=2, column_gap=32))

    body_rule = _css_block(css, ".markdown-body")
    assert "column-count: 2" in body_rule
    assert "column-gap: 32px" in body_rule
    h1_rule = _css_block(css, ".markdown-body h1")
    assert "column-span: all" in h1_rule


def test_two_columns_without_explicit_gap_uses_a_default() -> None:
    css = columns_css(MarkdownStyle(columns=2))

    assert "column-gap: 24px" in css
