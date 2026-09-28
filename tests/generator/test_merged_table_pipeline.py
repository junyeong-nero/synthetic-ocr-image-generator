"""Merged-cell HTML tables through composition, sheet fitting, mutation, GT_json and rendering."""

import random

from src.generator.data_provider import DataProvider
from src.generator.document_blocks import DocumentComposer
from src.generator.generator import Generator
from src.generator.markdown_render_utils import MarkdownStyle
from src.generator.markdown_renderers import HtmlMarkdownRenderer
from src.generator.profile_application import fit_markdown_to_sheet
from src.generator.table_schemas import SCHEMAS
from src.generator.text_mutation import _classify_markdown_chunk, mutate_text_generator_sections
from src.utils import markdown_to_json_ast

_BLUEPRINT = {
    "document_shape": "report",
    "section_count": [4, 4],
    "blocks_per_section": [2, 2],
    "allowed_blocks": ["paragraph", "table"],
    "required_blocks": ["table", "paragraph", "table"],
    "table": {"rows": [2, 5], "columns": [3, 5]},
}
_MERGED_CONTENT = {"table_schemas": {name: 1.0 for name in SCHEMAS}, "merged_table_ratio": 1.0}


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def _compose(content_specs, seed: int = 3):
    random.seed(seed)
    composer = DocumentComposer(
        data=DataProvider(lang="ko", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "x = y",
        content_specs=content_specs,
    )
    return composer.compose(_BLUEPRINT)


def _chunks(markdown: str) -> list[str]:
    return [chunk for chunk in markdown.strip().split("\n\n") if chunk.strip()]


def _content_chunks(markdown: str) -> list[str]:
    return [chunk for chunk in _chunks(markdown) if not chunk.startswith("#")]


def _html_tables(markdown: str) -> list[str]:
    return [chunk for chunk in _chunks(markdown) if chunk.startswith("<table>")]


def test_sample_content_defaults_merged_table_ratio_to_zero() -> None:
    composer = DocumentComposer(
        data=DataProvider(lang="ko", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "x",
        content_specs={"section_count_scale": 1.0},
    )
    assert composer._sample_content()["merged_table_ratio"] == 0.0
    composer.content_specs = {"merged_table_ratio": 0.4}
    assert composer._sample_content()["merged_table_ratio"] == 0.4


def test_content_merged_table_ratio_emits_html_table_blocks() -> None:
    markdown, metadata = _compose(_MERGED_CONTENT)

    tables = _html_tables(markdown)
    assert len(tables) == metadata.block_type_counts["table"] >= 2
    assert "| --- |" not in markdown and "---:" not in markdown


def test_without_merged_table_ratio_composition_is_unchanged() -> None:
    schemas_only = {"table_schemas": _MERGED_CONTENT["table_schemas"]}
    with_zero = dict(schemas_only, merged_table_ratio=0.0)

    assert _compose(schemas_only) == _compose(with_zero)
    assert not _html_tables(_compose(schemas_only)[0])


def test_composition_metadata_counts_each_html_table_as_one_table_block() -> None:
    markdown, metadata = _compose(_MERGED_CONTENT)

    chunks = _content_chunks(markdown)
    assert len(chunks) == len(metadata.block_types)
    for chunk, block_type in zip(chunks, metadata.block_types):
        assert _classify_markdown_chunk(chunk) == block_type
        if block_type == "table":
            assert chunk.startswith("<table>") and chunk.endswith("</table>")


def test_fit_markdown_to_sheet_keeps_html_tables_whole_and_counts_aligned() -> None:
    markdown, metadata = _compose(_MERGED_CONTENT)

    for overflow in (1.3, 1.8, 3.0):
        trimmed, kept = fit_markdown_to_sheet(markdown, overflow)
        assert trimmed != markdown
        chunks = _content_chunks(trimmed)
        assert kept == len(chunks)
        trimmed_meta = Generator._trim_composition_metadata(metadata.to_dict(), trimmed, kept)
        assert trimmed_meta["block_types"] == metadata.block_types[:kept]
        for chunk, block_type in zip(chunks, trimmed_meta["block_types"]):
            assert _classify_markdown_chunk(chunk) == block_type
        for table in _html_tables(trimmed):
            assert table in _html_tables(markdown)  # never cut mid-table


def test_text_mutation_leaves_html_tables_untouched_and_still_mutates_prose() -> None:
    markdown, metadata = _compose(_MERGED_CONTENT)

    def mutate(text: str, ratio: float) -> tuple[str, int]:
        _ = ratio
        return "MUTATED " + text, 1

    mutated, count = mutate_text_generator_sections(markdown, 0.5, metadata.block_types, mutate)

    assert count == metadata.block_type_counts["paragraph"] > 0
    assert _html_tables(mutated) == _html_tables(markdown)
    assert "MUTATED" not in "".join(_html_tables(mutated))


def test_gt_json_has_one_block_html_token_per_table() -> None:
    markdown, _ = _compose(_MERGED_CONTENT)

    tokens = markdown_to_json_ast(markdown)
    html_blocks = [token["raw"].strip() for token in tokens if token["type"] == "block_html"]
    assert html_blocks == _html_tables(markdown)
    assert not any(token["type"] == "table" for token in tokens)


def test_python_markdown_renders_html_table_verbatim() -> None:
    markdown, _ = _compose(_MERGED_CONTENT)
    renderer = HtmlMarkdownRenderer("missing-font.ttf", MarkdownStyle())

    rendered = renderer._coerce_markdown_html(renderer._prepare_component_markdown(markdown))

    for table in _html_tables(markdown):
        assert table in rendered
    assert "<p><table" not in rendered
    assert rendered.count("<table>") == len(_html_tables(markdown))
