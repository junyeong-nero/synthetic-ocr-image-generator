"""Korean document convention styling: heading numbering, 개조식 list markers,
law-style articles, and the two-parser (python-markdown / mistune) agreement
that keeps the rendered image, GT_markdown and GT_json in sync.
"""

import random

import markdown as markdown_pkg
import mistune
import pytest

from src.generator.data_provider import DataProvider
from src.generator.document_blocks import DocumentComposer
from src.generator.document_conventions import DocumentConventions
from src.generator.text_mutation import mutate_text_generator_sections


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def _data(lang: str = "ko") -> DataProvider:
    return DataProvider(lang=lang, mix_ratio=0.0, use_corpus=False)


def _render_html(markdown_text: str) -> str:
    return markdown_pkg.markdown(
        markdown_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists", "nl2br"],
    )


_MISTUNE_PARSER = mistune.create_markdown(
    renderer="ast", plugins=["table", "task_lists", "strikethrough"]
)


def _mistune_ast(markdown_text: str):
    return _MISTUNE_PARSER(markdown_text)


def _has_structural_list(ast) -> bool:
    """True if any AST node (recursively) is a mistune list block."""
    if isinstance(ast, dict):
        if ast.get("type") == "list":
            return True
        return any(_has_structural_list(value) for value in ast.values())
    if isinstance(ast, list):
        return any(_has_structural_list(item) for item in ast)
    return False


# ---------------------------------------------------------------------------
# Heading numbering
# ---------------------------------------------------------------------------


def test_heading_numbering_none_leaves_headings_unchanged() -> None:
    conventions = DocumentConventions(
        lang="ko",
        document_shape="business_report",
        content_specs={},
        data=_data(),
        clip_text=_clip,
    )

    assert conventions.plan.heading_numbering == "none"
    assert conventions.style_heading("개요") == "개요"
    assert conventions.style_heading("추진현황") == "추진현황"


def test_heading_numbering_arabic_prefixes_sequential_numbers() -> None:
    random.seed(1)
    conventions = DocumentConventions(
        lang="en",
        document_shape="business_report",
        content_specs={"heading_numbering": "arabic"},
        data=_data("en"),
        clip_text=_clip,
    )

    assert conventions.plan.heading_numbering == "arabic"
    headings = [conventions.style_heading(f"Section {i}") for i in range(1, 4)]
    # Either plain ("1.", "2.", "3.") or dotted ("1.1", "1.2", "1.3") -- both
    # increment and both start with a digit.
    prefixes = [heading.split(" ", 1)[0] for heading in headings]
    assert all(prefix[0].isdigit() for prefix in prefixes)
    assert len(set(prefixes)) == 3


def test_heading_numbering_roman_uses_ascii_roman_numerals() -> None:
    conventions = DocumentConventions(
        lang="en",
        document_shape="business_report",
        content_specs={"heading_numbering": "roman"},
        data=_data("en"),
        clip_text=_clip,
    )

    headings = [conventions.style_heading(f"Section {i}") for i in range(1, 4)]
    assert headings[0].startswith("I. ")
    assert headings[1].startswith("II. ")
    assert headings[2].startswith("III. ")


def test_heading_numbering_korean_admin_uses_pooled_markers_when_ko() -> None:
    random.seed(3)
    conventions = DocumentConventions(
        lang="ko",
        document_shape="business_report",
        content_specs={"heading_numbering": "korean_admin"},
        data=_data(),
        clip_text=_clip,
    )

    assert conventions.plan.heading_numbering == "korean_admin"
    heading = conventions.style_heading("개요")
    marker = heading.split(" ", 1)[0]
    assert marker[:-1] in (
        "Ⅰ", "Ⅱ", "Ⅲ",
        "1", "2", "3",
        "가", "나", "다",
    )


def test_heading_numbering_korean_admin_falls_back_to_arabic_outside_ko() -> None:
    conventions = DocumentConventions(
        lang="en",
        document_shape="business_report",
        content_specs={"heading_numbering": "korean_admin"},
        data=_data("en"),
        clip_text=_clip,
    )

    assert conventions.plan.heading_numbering == "arabic"
    heading = conventions.style_heading("Overview")
    marker = heading.split(" ", 1)[0]
    assert marker[0].isdigit()


# ---------------------------------------------------------------------------
# List style
# ---------------------------------------------------------------------------


def test_list_style_markdown_default_leaves_blocks_unchanged() -> None:
    conventions = DocumentConventions(
        lang="ko",
        document_shape="business_report",
        content_specs={},
        data=_data(),
        clip_text=_clip,
    )

    assert conventions.plan.list_style == "markdown"
    assert conventions.style_block("bullet_list", "- 항목 1\n- 항목 2") is None
    assert conventions.style_block("numbered_list", "1. 항목 1\n2. 항목 2") is None


def test_list_style_korean_admin_bullets_use_box_circle_dot_markers() -> None:
    random.seed(5)
    conventions = DocumentConventions(
        lang="ko",
        document_shape="business_report",
        content_specs={"list_style": "korean_admin"},
        data=_data(),
        clip_text=_clip,
    )

    styled = conventions.style_block("bullet_list", "- 항목 1\n- 항목 2\n- 항목 3")
    assert styled is not None
    body_lines = [line for line in styled.splitlines() if not line.startswith("※")]
    assert all(line[0] in "□○·" for line in body_lines)
    assert not any(line.startswith("- ") for line in styled.splitlines())


def test_list_style_korean_admin_numbered_uses_pooled_markers() -> None:
    random.seed(7)
    conventions = DocumentConventions(
        lang="ko",
        document_shape="business_report",
        content_specs={"list_style": "korean_admin"},
        data=_data(),
        clip_text=_clip,
    )

    styled = conventions.style_block("numbered_list", "1. 항목 1\n2. 항목 2\n3. 항목 3")
    assert styled is not None
    # Never plain "1." markdown ordered-list syntax.
    assert not any(
        line.split(" ", 1)[0].rstrip(".").isdigit() and line.split(" ", 1)[0].endswith(".")
        for line in styled.splitlines()
        if not line.startswith("※")
    )


def test_list_style_korean_admin_falls_back_to_markdown_outside_ko() -> None:
    conventions = DocumentConventions(
        lang="en",
        document_shape="business_report",
        content_specs={"list_style": "korean_admin"},
        data=_data("en"),
        clip_text=_clip,
    )

    assert conventions.plan.list_style == "markdown"
    assert conventions.style_block("bullet_list", "- a\n- b") is None


# ---------------------------------------------------------------------------
# Law articles
# ---------------------------------------------------------------------------


def test_law_articles_rewrites_numbered_list_as_articles_when_triggered() -> None:
    random.seed(9)
    conventions = DocumentConventions(
        lang="ko",
        document_shape="policy_document",
        content_specs={"law_articles": {"p": 1.0}},
        data=_data(),
        clip_text=_clip,
    )

    assert conventions.plan.law_articles_used is True
    styled = conventions.style_block(
        "numbered_list", "1. 목적 조항\n2. 세부 절차\n3. 예외 사항"
    )
    assert styled is not None
    lines = styled.splitlines()
    assert lines[0].startswith("제1조(")
    assert "목적 조항" in lines[0]
    assert lines[1].startswith("①")
    assert lines[2].startswith("②")


def test_law_articles_article_numbers_increment_across_blocks() -> None:
    random.seed(11)
    conventions = DocumentConventions(
        lang="ko",
        document_shape="policy_document",
        content_specs={"law_articles": {"p": 1.0}},
        data=_data(),
        clip_text=_clip,
    )

    first = conventions.style_block("numbered_list", "1. 가\n2. 나")
    second = conventions.style_block("numbered_list", "1. 다\n2. 라")

    assert first.splitlines()[0].startswith("제1조(")
    assert second.splitlines()[0].startswith("제2조(")


def test_law_articles_not_applied_outside_policy_document_shape() -> None:
    conventions = DocumentConventions(
        lang="ko",
        document_shape="business_report",
        content_specs={"law_articles": {"p": 1.0}},
        data=_data(),
        clip_text=_clip,
    )

    assert conventions.plan.law_articles_used is False
    assert conventions.style_block("numbered_list", "1. a\n2. b") is None


def test_law_articles_not_applied_outside_korean() -> None:
    conventions = DocumentConventions(
        lang="en",
        document_shape="policy_document",
        content_specs={"law_articles": {"p": 1.0}},
        data=_data("en"),
        clip_text=_clip,
    )

    assert conventions.plan.law_articles_used is False


def test_law_articles_probability_zero_never_triggers() -> None:
    random.seed(0)
    for _ in range(20):
        conventions = DocumentConventions(
            lang="ko",
            document_shape="policy_document",
            content_specs={"law_articles": {"p": 0.0}},
            data=_data(),
            clip_text=_clip,
        )
        assert conventions.plan.law_articles_used is False


# ---------------------------------------------------------------------------
# Two-parser agreement: every marker line stays plain text in both parsers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "markdown_text",
    [
        "□ 대분류 항목\n○ 하위 항목\n· 세부 항목",
        "□ 대분류 항목\n○ 하위 항목\n※ 참고 사항",
        "1\\) 첫째 항목\n2\\) 둘째 항목\n3\\) 셋째 항목",
        "가) 첫째 항목\n나) 둘째 항목\n다) 셋째 항목",
        "① 첫째 항목\n② 둘째 항목\n③ 셋째 항목",
        "제1조(목적) 이 규정은 목적을 정의한다.\n① 항목 1\n② 항목 2",
        "Ⅰ. 로마자 대제목",
        "가. 한글 세부 표제",
    ],
)
def test_marker_lines_agree_between_markdown_py_and_mistune(markdown_text: str) -> None:
    html = _render_html(markdown_text)
    ast = _mistune_ast(markdown_text)

    # Neither parser turns the marker into a <ol>/<ul> or a mistune list node.
    assert "<ol" not in html
    assert "<ul" not in html
    assert not _has_structural_list(ast)


def test_composer_with_korean_admin_and_law_articles_keeps_parsers_in_sync() -> None:
    random.seed(21)
    composer = DocumentComposer(
        data=_data(),
        clip_text=_clip,
        formula_supplier=lambda: "a=b",
        content_specs={
            "heading_numbering": "korean_admin",
            "list_style": "korean_admin",
            "law_articles": {"p": 1.0},
        },
    )

    for seed in range(21, 31):
        random.seed(seed)
        markdown_text, metadata = composer.compose(
            {
                "document_shape": "policy_document",
                "section_count": [5, 5],
                "blocks_per_section": [1, 1],
                "allowed_blocks": ["paragraph", "numbered_list", "bullet_list", "quote"],
                "required_blocks": ["numbered_list", "bullet_list"],
            }
        )

        html = _render_html(markdown_text)
        ast = _mistune_ast(markdown_text)

        html_has_list = "<ol" in html or "<ul" in html
        ast_has_list = _has_structural_list(ast)
        assert html_has_list == ast_has_list, markdown_text
        # korean_admin + law_articles never produce a real markdown list.
        assert not html_has_list
        assert not ast_has_list
        assert metadata.law_articles_used is True


# ---------------------------------------------------------------------------
# text_mutation.py interaction: korean_admin markers no longer look like
# "- item" / "N. item" to `_classify_markdown_chunk`, so the look-alike-typo
# mutation must skip them (not raise, not silently mangle the markers).
# ---------------------------------------------------------------------------


def test_mutate_text_generator_sections_skips_korean_admin_markers_safely() -> None:
    random.seed(31)
    composer = DocumentComposer(
        data=_data(),
        clip_text=_clip,
        formula_supplier=lambda: "a=b",
        content_specs={"list_style": "korean_admin"},
    )
    markdown_text, metadata = composer.compose(
        {
            "document_shape": "business_report",
            "section_count": [3, 3],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["bullet_list", "numbered_list"],
            "required_blocks": ["bullet_list", "numbered_list"],
        }
    )

    def fake_mutate(section_text: str, _ratio: float):
        return section_text.replace("가", "X"), 1

    mutated, mutation_count = mutate_text_generator_sections(
        markdown_text=markdown_text,
        ratio=0.5,
        merge_order=metadata.block_types,
        mutate_section=fake_mutate,
    )

    # The rich-block mutation path re-classifies each chunk from its literal
    # text and compares it against the planned block type; korean_admin
    # markers no longer match, so it aborts (unchanged text, 0 mutations)
    # instead of guessing -- no exception, no corrupted markers.
    assert mutated == markdown_text
    assert mutation_count == 0


def test_mutate_text_generator_sections_still_mutates_legacy_markdown_lists() -> None:
    random.seed(33)
    composer = DocumentComposer(
        data=_data(),
        clip_text=_clip,
        formula_supplier=lambda: "a=b",
        content_specs={},
    )
    markdown_text, metadata = composer.compose(
        {
            "document_shape": "business_report",
            "section_count": [2, 2],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["bullet_list", "paragraph"],
            "required_blocks": ["bullet_list", "paragraph"],
        }
    )

    def fake_mutate(section_text: str, _ratio: float):
        return section_text + "!", 1

    mutated, mutation_count = mutate_text_generator_sections(
        markdown_text=markdown_text,
        ratio=0.5,
        merge_order=metadata.block_types,
        mutate_section=fake_mutate,
    )

    assert mutation_count > 0
    assert mutated != markdown_text
