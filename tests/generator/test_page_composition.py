"""Continuation pages: drop the title and leading sections, start mid-paragraph."""

import random
import re

import pytest

from src.generator.data_provider import DataProvider
from src.generator.markdown_content import MarkdownDataGenerator
from src.generator.page_composition import compose_page
from src.generator.template_catalog import TemplateCatalog

_DOC = (
    "# 문서 제목\n\n"
    "## 1. 개요\n\n"
    "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.\n\n"
    "- 항목 하나\n- 항목 둘\n\n"
    "## 2. 현황\n\n"
    "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
    "## 3. 계획\n\n"
    "넷째 문장입니다. 다섯째 문장입니다.\n\n"
    "## 4. 기타\n\n"
    "여섯째 문장입니다. 일곱째 문장입니다.\n"
)
_MERGE = ["paragraph", "bullet_list", "table", "paragraph", "paragraph"]
_META = {
    "document_shape": "business_report",
    "block_types": list(_MERGE),
    "block_type_counts": {"paragraph": 3, "bullet_list": 1, "table": 1},
    "section_count": 4,
    "heading_numbering": "arabic",
}


def _chunks(markdown: str) -> list[str]:
    return [chunk for chunk in markdown.strip().split("\n\n") if chunk.strip()]


def _is_heading(chunk: str) -> bool:
    return chunk.lstrip().startswith("#") and "\n" not in chunk.strip()


def _assert_consistent(result) -> None:
    """merge_order / composition metadata describe exactly the page's blocks."""
    blocks = [chunk for chunk in _chunks(result.markdown) if not _is_heading(chunk)]
    assert len(blocks) == len(result.merge_order)
    meta = result.composition_metadata
    assert meta["block_types"] == result.merge_order
    assert meta["block_type_counts"] == {
        block_type: result.merge_order.count(block_type) for block_type in set(result.merge_order)
    }
    assert meta["section_count"] == sum(
        1 for line in result.markdown.splitlines() if line.startswith("## ")
    )


def test_compose_page_is_a_no_op_without_content_keys() -> None:
    rng = random.Random(3)
    state = rng.getstate()

    assert compose_page(_DOC, list(_MERGE), dict(_META), {"table_schemas": {"order": 1}}, rng) is None
    # Legacy runs must not consume random draws.
    assert rng.getstate() == state


def test_compose_page_keeps_first_page_when_not_continuation() -> None:
    result = compose_page(
        _DOC, list(_MERGE), dict(_META), {"continuation_page": {"p": 0.0}}, random.Random(1)
    )

    assert result is not None
    assert result.markdown == _DOC
    assert result.merge_order == _MERGE
    assert result.continuation_page is False
    assert result.start_mid_paragraph is False
    assert result.metadata() == {"continuation_page": False, "start_mid_paragraph": False}


@pytest.mark.parametrize("seed", range(20))
def test_continuation_page_drops_title_and_leading_sections(seed: int) -> None:
    result = compose_page(
        _DOC, list(_MERGE), dict(_META), {"continuation_page": {"p": 1.0}}, random.Random(seed)
    )

    assert result.continuation_page is True
    assert result.start_mid_paragraph is False
    chunks = _chunks(result.markdown)
    original = _chunks(_DOC)
    # No document title; the page starts at a later section heading.
    assert not any(chunk.startswith("# ") for chunk in chunks)
    assert chunks[0].startswith("## ") and chunks[0] != "## 1. 개요"
    # The page is the tail of the original document.
    assert chunks == original[len(original) - len(chunks):]
    assert result.markdown.endswith("\n")
    # Composition extras (e.g. Korean conventions) are carried over.
    assert result.composition_metadata["heading_numbering"] == "arabic"
    assert result.composition_metadata["document_shape"] == "business_report"
    _assert_consistent(result)


def test_continuation_page_reaches_every_later_section() -> None:
    starts = set()
    for seed in range(200):
        result = compose_page(
            _DOC, list(_MERGE), dict(_META), {"continuation_page": {"p": 1.0}}, random.Random(seed)
        )
        starts.add(_chunks(result.markdown)[0])

    assert starts == {"## 2. 현황", "## 3. 계획", "## 4. 기타"}


def test_single_section_continuation_drops_only_the_title() -> None:
    doc = "# 제목\n\n## 개요\n\n첫 문장입니다. 둘째 문장입니다.\n"
    result = compose_page(
        doc, ["paragraph"], {"block_types": ["paragraph"]}, {"continuation_page": {"p": 1.0}}, random.Random(0)
    )

    assert result.markdown == "## 개요\n\n첫 문장입니다. 둘째 문장입니다.\n"
    assert result.merge_order == ["paragraph"]
    _assert_consistent(result)


@pytest.mark.parametrize("seed", range(40))
def test_start_mid_paragraph_opens_with_a_sentence_boundary_tail(seed: int) -> None:
    result = compose_page(
        _DOC,
        list(_MERGE),
        dict(_META),
        {"continuation_page": {"p": 1.0}, "start_mid_paragraph": {"p": 1.0}},
        random.Random(seed),
    )

    _assert_consistent(result)
    first = _chunks(result.markdown)[0]
    if not result.start_mid_paragraph:
        # Only possible when neither neighbouring paragraph can be cut.
        assert first.startswith("## ")
        return
    assert not _is_heading(first)
    assert result.merge_order[0] == "paragraph"
    original_paragraphs = [
        chunk for chunk, block_type in zip(
            [c for c in _chunks(_DOC) if not _is_heading(c)], _MERGE
        ) if block_type == "paragraph"
    ]
    source = next(p for p in original_paragraphs if p.endswith(first))
    # A proper tail that starts right after a sentence end.
    assert first != source
    assert re.search(r"[.!?。！？]\s+$", source[: len(source) - len(first)])


def test_start_mid_paragraph_picks_a_break_next_to_a_paragraph() -> None:
    firsts = set()
    for seed in range(100):
        result = compose_page(
            _DOC,
            list(_MERGE),
            dict(_META),
            {"continuation_page": {"p": 1.0}, "start_mid_paragraph": {"p": 1.0}},
            random.Random(seed),
        )
        # Starting at "## 2. 현황" (after a list, before a table) cannot start
        # mid-paragraph, so the break is placed where it can: the requested
        # rate is honoured whenever the document has a cuttable paragraph.
        assert result.start_mid_paragraph is True
        firsts.add(_chunks(result.markdown)[0])
    # Both the kept section's own first paragraph (its heading went to the
    # previous page) and the previous section's last paragraph are used.
    assert "다섯째 문장입니다." in firsts
    assert "일곱째 문장입니다." in firsts


def test_start_mid_paragraph_never_starts_with_block_markup() -> None:
    # A tail starting "1. ..." or "# ..." would render as a list or heading,
    # not as the paragraph merge_order says it is.
    doc = (
        "# 제목\n\n## 개요\n\n서론입니다. 1. 번호로 시작합니다. # 샵으로 시작합니다.\n\n"
        "## 본론\n\n본론입니다.\n"
    )
    for seed in range(30):
        result = compose_page(
            doc,
            ["paragraph", "paragraph"],
            {},
            {"continuation_page": {"p": 1.0}, "start_mid_paragraph": {"p": 1.0}},
            random.Random(seed),
        )
        first = _chunks(result.markdown)[0]
        assert not re.match(r"^\s*(?:[#>|*+\-]+|\d+[.)])\s+", first), first
        _assert_consistent(result)


def test_start_mid_paragraph_needs_a_continuation_page() -> None:
    result = compose_page(
        _DOC,
        list(_MERGE),
        dict(_META),
        {"continuation_page": {"p": 0.0}, "start_mid_paragraph": {"p": 1.0}},
        random.Random(0),
    )

    assert result.markdown == _DOC
    assert result.start_mid_paragraph is False


def test_compose_page_leaves_unaligned_merge_order_untouched() -> None:
    # Legacy (non-composer) documents list one merge_order entry per section,
    # not per block; cutting them could desynchronise merge_order and GT.
    result = compose_page(
        _DOC, ["text", "table"], dict(_META), {"continuation_page": {"p": 1.0}}, random.Random(0)
    )

    assert result.markdown == _DOC
    assert result.merge_order == ["text", "table"]
    assert result.continuation_page is False


def test_compose_page_is_deterministic_for_a_seed() -> None:
    specs = {"continuation_page": {"p": 1.0}, "start_mid_paragraph": {"p": 0.5}}
    first = compose_page(_DOC, list(_MERGE), dict(_META), specs, random.Random(7))
    second = compose_page(_DOC, list(_MERGE), dict(_META), specs, random.Random(7))

    assert first == second


def test_compose_page_bare_float_is_a_probability() -> None:
    kept = compose_page(_DOC, list(_MERGE), dict(_META), {"continuation_page": 0.0}, random.Random(0))
    cut = compose_page(_DOC, list(_MERGE), dict(_META), {"continuation_page": 1.0}, random.Random(0))

    assert kept.continuation_page is False
    assert cut.continuation_page is True


@pytest.mark.parametrize("template", ["business_report", "policy_document", "meeting_minutes"])
def test_compose_page_keeps_real_composer_output_consistent(tmp_path, template: str) -> None:
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "ko").mkdir(parents=True)
    (corpus_dir / "ko" / "paragraphs.txt").write_text(
        "첫 번째 문단입니다. 두 번째 문장입니다. 세 번째 문장입니다.\n"
        "네 번째 문단입니다. 다섯 번째 문장입니다.\n",
        encoding="utf-8",
    )
    data_generator = MarkdownDataGenerator(
        "ko", data_provider=DataProvider(lang="ko", corpus_dir=corpus_dir)
    )
    data_generator.content_specs = {
        "heading_numbering": "korean_admin",
        "list_style": {"choices": ["korean_admin", "markdown"]},
        "extra_blocks_per_section": 1,
    }
    spec = TemplateCatalog().get(template)
    specs = {"continuation_page": {"p": 1.0}, "start_mid_paragraph": {"p": 0.7}}

    for seed in range(15):
        random.seed(seed)
        markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
        merge_order = data_generator.pop_merge_order()
        metadata = data_generator.pop_composition_metadata()

        result = compose_page(markdown, merge_order, metadata, specs, random.Random(seed))

        assert result.continuation_page is True
        assert not result.markdown.startswith("# ")
        _assert_consistent(result)
