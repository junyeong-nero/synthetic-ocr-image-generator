"""Running headers, footers and page numbers drawn in the page margins."""

import random
import re
from pathlib import Path

import pytest

from src.generator.data_provider import DataProvider
from src.generator.markdown_render_utils import MarkdownStyle
from src.generator.markdown_renderers import HtmlMarkdownRenderer, PlaywrightMarkdownRenderer
from src.generator.page_furniture import (
    FurnitureLine,
    PageFurniture,
    apply_page_furniture,
    format_page_number,
    furniture_requested,
    sample_page_furniture,
)

_ALL_ON = {"header": True, "footer": True, "page_number": "dashed"}


def _data(tmp_path: Path, lang: str = "ko") -> DataProvider:
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / lang).mkdir(parents=True, exist_ok=True)
    return DataProvider(lang=lang, corpus_dir=corpus_dir)


def _furniture() -> PageFurniture:
    return PageFurniture(
        header=FurnitureLine(left="한국 & 기관", right="2024년 업무계획"),
        header_rule=True,
        footer=FurnitureLine(left="대외비", center="- 3 -"),
        footer_text="대외비",
        page_number="- 3 -",
    )


def test_furniture_requested_only_for_furniture_keys() -> None:
    assert furniture_requested({}) is False
    assert furniture_requested({"aspect_ratio": 1.414, "margins_mm": {"top": 20}}) is False
    assert furniture_requested({"header": False}) is True
    assert furniture_requested({"page_number": "none"}) is True


@pytest.mark.parametrize(
    "style, lang, expected",
    [
        ("dashed", "ko", "- 3 -"),
        ("plain", "ko", "3"),
        ("of_total", "ko", "3 / 12"),
        ("korean", "ko", "3쪽"),
        ("english", "ko", "Page 3"),
        ("korean", "en", "3"),
        ("none", "ko", ""),
        ("no-such-style", "ko", ""),
    ],
)
def test_format_page_number_styles(style: str, lang: str, expected: str) -> None:
    assert format_page_number(style, 3, 12, lang=lang) == expected


def test_sample_page_furniture_all_parts(tmp_path) -> None:
    random.seed(0)
    furniture = sample_page_furniture(_ALL_ON, _data(tmp_path), random.Random(1), lang="ko", continuation=False)

    assert furniture.has_content()
    assert furniture.header.parts()
    assert furniture.footer_text
    # A document's first page is page 1.
    assert furniture.page_number == "- 1 -"
    metadata = furniture.metadata()
    assert set(metadata) == {"header", "footer", "page_number"}
    assert metadata["header"] == "\n".join(furniture.header.parts())
    assert metadata["footer"] == furniture.footer_text
    assert metadata["page_number"] == "- 1 -"
    # Footer text and page number share the footer line.
    assert set(furniture.footer.parts()) == {furniture.footer_text, "- 1 -"}


def test_continuation_pages_get_later_page_numbers(tmp_path) -> None:
    data = _data(tmp_path)
    numbers = set()
    for seed in range(40):
        random.seed(seed)
        furniture = sample_page_furniture(
            {"page_number": "plain"}, data, random.Random(seed), lang="ko", continuation=True
        )
        numbers.add(int(furniture.page_number))
    assert min(numbers) >= 2
    assert len(numbers) > 5


def test_of_total_page_number_never_exceeds_total(tmp_path) -> None:
    data = _data(tmp_path)
    for seed in range(40):
        random.seed(seed)
        furniture = sample_page_furniture(
            {"page_number": "of_total"}, data, random.Random(seed), lang="ko", continuation=seed % 2 == 0
        )
        number, total = (int(part) for part in furniture.page_number.split(" / "))
        assert 1 <= number <= total


def test_absent_parts_are_empty_strings_not_none(tmp_path) -> None:
    random.seed(0)
    furniture = sample_page_furniture(
        {"header": False, "footer": False, "page_number": "none"},
        _data(tmp_path),
        random.Random(0),
        lang="ko",
        continuation=False,
    )

    assert not furniture.has_content()
    # Hub schemas are inferred from the first row: never None-typed values.
    assert furniture.metadata() == {"header": "", "footer": "", "page_number": ""}


def test_bare_probability_and_p_spec_are_accepted(tmp_path) -> None:
    random.seed(0)
    on = sample_page_furniture({"header": 1.0}, _data(tmp_path), random.Random(0), lang="ko", continuation=False)
    off = sample_page_furniture({"header": 0.0}, _data(tmp_path), random.Random(0), lang="ko", continuation=False)

    assert on.header.parts() and not off.header.parts()


def test_header_parts_are_short(tmp_path) -> None:
    data = _data(tmp_path)
    for seed in range(30):
        random.seed(seed)
        furniture = sample_page_furniture(_ALL_ON, data, random.Random(seed), lang="ko", continuation=False)
        for part in furniture.header.parts():
            assert 0 < len(part) <= 24
        assert 0 < len(furniture.footer_text) <= 24


@pytest.mark.parametrize("lang", ["ko", "en", "ja"])
def test_sampling_is_deterministic_per_seed(tmp_path, lang: str) -> None:
    data = _data(tmp_path, lang)

    def draw() -> PageFurniture:
        random.seed(5)
        data.faker.seed_instance(5)
        return sample_page_furniture(_ALL_ON, data, random.Random(5), lang=lang, continuation=True)

    assert draw() == draw()


def test_apply_page_furniture_sets_sheet_height_from_aspect() -> None:
    style = MarkdownStyle()
    apply_page_furniture(style, _furniture(), aspect_ratio=1.414)

    page_width = style.margin_left + style.content_width + style.margin_right
    assert style.page_furniture == _furniture()
    assert style.sheet_height == round(page_width * 1.414)


def test_apply_page_furniture_without_aspect_keeps_natural_height() -> None:
    style = MarkdownStyle()
    apply_page_furniture(style, _furniture(), aspect_ratio=None)

    assert style.page_furniture == _furniture()
    assert style.sheet_height is None


def test_apply_page_furniture_ignores_empty_furniture() -> None:
    style = MarkdownStyle()
    apply_page_furniture(style, PageFurniture(), aspect_ratio=1.414)

    assert style.page_furniture is None
    assert style.sheet_height is None


def test_apply_page_furniture_makes_room_in_tiny_margins() -> None:
    style = MarkdownStyle(margin_top=6, margin_bottom=6, body_font_size=14)
    apply_page_furniture(style, _furniture(), aspect_ratio=1.414)

    assert style.margin_top >= 20
    assert style.margin_bottom >= 20


def test_apply_page_furniture_keeps_realistic_margins() -> None:
    # 35mm / 15mm on a ~680px page: the header fits the HWP header zone.
    style = MarkdownStyle(margin_top=113, margin_bottom=49, body_font_size=12)
    apply_page_furniture(style, _furniture(), aspect_ratio=1.414)

    assert (style.margin_top, style.margin_bottom) == (113, 49)


def _html(style: MarkdownStyle, markdown: str = "# 제목\n\n본문입니다.\n") -> str:
    return HtmlMarkdownRenderer("body.ttf", style)._build_html_document(markdown)


def test_renderer_draws_furniture_outside_the_markdown_body() -> None:
    style = MarkdownStyle(margin_top=113, margin_bottom=49, body_font_size=12)
    apply_page_furniture(style, _furniture(), aspect_ratio=1.414)

    html = _html(style)

    body = html.split("<body>", 1)[1]
    assert '<div class="page-sheet">' in body
    header_at = body.index("page-header")
    content_at = body.index('<div class="markdown-body">')
    footer_at = body.index("page-footer")
    assert header_at < content_at < footer_at
    # Text is HTML-escaped and every part is drawn.
    assert "한국 &amp; 기관" in body and "2024년 업무계획" in body
    assert "대외비" in body and "- 3 -" in body
    # The markdown body itself carries none of it.
    content = body[content_at: body.index("</div>", content_at)]
    assert "대외비" not in content and "- 3 -" not in content
    assert f"min-height: {style.sheet_height}px" in html
    assert html.endswith("</div>\n</body>\n</html>")


def test_renderer_places_header_and_footer_inside_the_margins() -> None:
    style = MarkdownStyle(margin_top=113, margin_bottom=49, body_font_size=12)
    apply_page_furniture(style, _furniture(), aspect_ratio=1.414)

    html = _html(style)

    header_css = re.search(r"\.page-header \{([^}]*)\}", html).group(1)
    footer_css = re.search(r"\.page-footer \{([^}]*)\}", html).group(1)
    font_px = int(re.search(r"\.page-furniture \{[^}]*font-size: (\d+)px", html).group(1))
    top = int(re.search(r"top: (\d+)px", header_css).group(1))
    bottom = int(re.search(r"bottom: (\d+)px", footer_css).group(1))
    assert 0 < top and top + font_px * 1.25 < style.margin_top
    assert 0 < bottom and bottom + font_px * 1.25 < style.margin_bottom
    assert font_px < style.body_font_size
    # The optional rule under the header.
    assert "border-bottom" in header_css


def test_renderer_without_furniture_keeps_legacy_html() -> None:
    html = _html(MarkdownStyle())

    assert "page-sheet" not in html and "page-furniture" not in html
    assert '<body>\n  <div class="markdown-body">' in html


def test_capture_shell_wraps_legacy_document_exactly_as_before() -> None:
    html = _html(MarkdownStyle())

    wrapped = PlaywrightMarkdownRenderer._wrap_capture_shell(html, 680, 8)

    legacy = html.replace(
        '<body>\n  <div class="markdown-body">',
        (
            '<body>\n'
            '  <div class="capture-shell" style="padding: 8px; width: 696px; overflow: visible;">\n'
            '    <div class="markdown-body">'
        ),
        1,
    ).replace("</div>\n</body>", "</div>\n  </div>\n</body>", 1)
    assert wrapped == legacy


def test_capture_shell_wraps_the_page_sheet() -> None:
    style = MarkdownStyle()
    apply_page_furniture(style, _furniture(), aspect_ratio=1.414)
    html = _html(style)

    wrapped = PlaywrightMarkdownRenderer._wrap_capture_shell(html, 680, 8)

    body = wrapped.split("<body>", 1)[1]
    assert body.index('class="capture-shell"') < body.index('class="page-sheet"')
    assert wrapped.endswith("</div>\n  </div>\n</body>\n</html>")
    assert wrapped.count("</div>") == html.count("</div>") + 1
