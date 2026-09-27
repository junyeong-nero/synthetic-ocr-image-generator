"""Tests for src/realism/layout_stats.py against synthetic drawn pages with
known line count, line height, margins and column structure."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from src.realism.layout_stats import compute_layout_stats


def _page(width: int = 1000, height: int = 1400) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), (255, 255, 255))
    return image, ImageDraw.Draw(image)


def _single_column_page(
    *,
    width: int = 1000,
    height: int = 1400,
    top: int = 140,
    left: int = 100,
    right: int = 900,
    line_height: int = 14,
    line_gap: int = 20,
    n_lines: int = 20,
) -> Image.Image:
    """A page with `n_lines` solid text-line bars of known height/gap."""
    image, draw = _page(width, height)
    y = top
    for _ in range(n_lines):
        draw.rectangle((left, y, right, y + line_height - 1), fill=(10, 10, 10))
        y += line_height + line_gap
    return image


def test_text_line_count_matches_drawn_lines() -> None:
    image = _single_column_page(n_lines=20)
    stats = compute_layout_stats(image)
    assert stats["text_line_count"] == 20


def test_median_line_height_matches_drawn_height() -> None:
    width = 1000
    line_height = 14
    image = _single_column_page(width=width, line_height=line_height, n_lines=20)
    stats = compute_layout_stats(image)
    assert stats["text_line_height_frac"] == pytest.approx(line_height / width, abs=1e-6)


def test_median_line_height_pt_is_positive_and_scales_with_height() -> None:
    short = _single_column_page(line_height=10, n_lines=20)
    tall = _single_column_page(line_height=20, n_lines=20)
    short_pt = compute_layout_stats(short)["text_line_height_pt"]
    tall_pt = compute_layout_stats(tall)["text_line_height_pt"]
    assert short_pt > 0
    assert tall_pt > short_pt


def test_ink_bbox_margins_match_drawn_bounds() -> None:
    width, height = 1000, 1400
    top, left, right = 140, 100, 900
    line_height = 14
    n_lines = 20
    line_gap = 20
    image = _single_column_page(
        width=width,
        height=height,
        top=top,
        left=left,
        right=right,
        line_height=line_height,
        line_gap=line_gap,
        n_lines=n_lines,
    )
    stats = compute_layout_stats(image)
    bottom = top + n_lines * (line_height + line_gap) - line_gap - 1

    assert stats["margin_top_frac"] == pytest.approx(top / height, abs=0.01)
    assert stats["margin_bottom_frac"] == pytest.approx((height - 1 - bottom) / height, abs=0.01)
    assert stats["margin_left_frac"] == pytest.approx(left / width, abs=0.01)
    assert stats["margin_right_frac"] == pytest.approx((width - 1 - (right - 1)) / width, abs=0.02)


def test_single_column_page_reports_one_column() -> None:
    image = _single_column_page(n_lines=20)
    stats = compute_layout_stats(image)
    assert stats["column_count"] == 1


def test_two_column_page_reports_two_columns() -> None:
    width, height = 1000, 1400
    image, draw = _page(width, height)
    # Two text blocks with a wide, persistent white gutter between them.
    for y in range(140, 1260, 34):
        draw.rectangle((80, y, 440, y + 14), fill=(10, 10, 10))
        draw.rectangle((560, y, 920, y + 14), fill=(10, 10, 10))
    stats = compute_layout_stats(image)
    assert stats["column_count"] == 2


def test_two_column_page_with_spanning_title_and_footer_reports_two_columns() -> None:
    """A full-width title and a centred page number both cross the gutter,
    as they typically do on real two-column pages (running head, page
    number). The gutter must still be detected from the much larger body."""
    width, height = 1000, 1600
    image, draw = _page(width, height)
    # Full-width title band, crossing the future gutter column range.
    draw.rectangle((80, 60, 920, 90), fill=(10, 10, 10))
    # Two-column body: 38 lines per column, gutter always empty.
    for y in range(140, 1420, 34):
        draw.rectangle((80, y, 440, y + 14), fill=(10, 10, 10))
        draw.rectangle((560, y, 920, y + 14), fill=(10, 10, 10))
    # Centred page-number band near the bottom, crossing the gutter too.
    draw.rectangle((460, 1460, 540, 1480), fill=(10, 10, 10))

    stats = compute_layout_stats(image)
    assert stats["column_count"] == 2


def test_blank_page_reports_zero_lines_and_full_margins() -> None:
    image, _ = _page()
    stats = compute_layout_stats(image)
    assert stats["text_line_count"] == 0
    assert stats["column_count"] == 1
    assert stats["text_area_frac"] == pytest.approx(0.0)


def test_text_area_fraction_is_between_zero_and_one() -> None:
    image = _single_column_page(n_lines=20)
    stats = compute_layout_stats(image)
    assert 0.0 < stats["text_area_frac"] < 1.0


def test_horizontal_rule_is_counted_and_distinguished_from_text_lines() -> None:
    width, height = 1000, 1400
    image, draw = _page(width, height)
    # Text lines: thick bars.
    for y in range(140, 500, 34):
        draw.rectangle((100, y, 900, y + 14), fill=(10, 10, 10))
    # A thin long horizontal rule well clear of the text lines.
    draw.rectangle((100, 700, 900, 701), fill=(10, 10, 10))
    for y in range(760, 1100, 34):
        draw.rectangle((100, y, 900, y + 14), fill=(10, 10, 10))
    stats = compute_layout_stats(image)
    assert stats["rule_count"] >= 1


def test_page_with_no_rule_reports_zero_rules() -> None:
    image = _single_column_page(n_lines=20)
    stats = compute_layout_stats(image)
    assert stats["rule_count"] == 0
