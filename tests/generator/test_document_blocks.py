import random
from collections import Counter
from pathlib import Path

from generator.data_provider import DataProvider
from generator.document_blocks import (
    DEFAULT_BLOCK_TYPES,
    GENRE_BY_DOCUMENT_SHAPE,
    DocumentComposer,
    normalize_block_types,
)


def _clip(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def test_normalize_block_types_ignores_unknown_values() -> None:
    assert normalize_block_types(["paragraph", "unknown", "table"]) == ["paragraph", "table"]


def test_normalize_block_types_falls_back_when_all_values_invalid() -> None:
    assert normalize_block_types(["unknown"]) == list(DEFAULT_BLOCK_TYPES)


def test_composer_satisfies_required_blocks_and_records_metadata() -> None:
    random.seed(11)
    composer = DocumentComposer(
        data=DataProvider(lang="en", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "E = mc^2",
    )
    markdown, metadata = composer.compose(
        {
            "document_shape": "technical_manual",
            "section_count": [4, 4],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["paragraph", "bullet_list", "code", "table"],
            "required_blocks": ["code", "table"],
            "table": {"rows": [2, 2], "columns": [3, 3]},
        }
    )

    assert markdown.startswith("# ")
    assert "```" in markdown
    assert "| " in markdown
    assert metadata.document_shape == "technical_manual"
    assert metadata.section_count == 4
    assert "code" in metadata.block_types
    assert "table" in metadata.block_types
    assert metadata.block_type_counts["code"] == 1
    assert metadata.block_type_counts["table"] == 1


def test_table_only_document_has_single_section_heading() -> None:
    random.seed(29)
    composer = DocumentComposer(
        data=DataProvider(lang="en", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "E = mc^2",
    )
    markdown, metadata = composer.compose(
        {
            "document_shape": "table_only",
            "section_count": [1, 1],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["table"],
            "required_blocks": ["table"],
            "table": {"rows": [2, 2], "columns": [3, 3]},
        }
    )

    section_headings = [
        line for line in markdown.splitlines() if line.startswith("## ")
    ]
    assert len(section_headings) == 1
    assert metadata.section_count == 1
    assert metadata.block_types == ["table"]


def test_composer_emits_each_supported_block_type() -> None:
    composer = DocumentComposer(
        data=DataProvider(lang="en", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "x^2 + y^2 = z^2",
    )

    markdown, metadata = composer.compose(
        {
            "document_shape": "all_blocks",
            "section_count": [11, 11],
            "blocks_per_section": [1, 1],
            "allowed_blocks": list(DEFAULT_BLOCK_TYPES),
            "required_blocks": list(DEFAULT_BLOCK_TYPES),
            "table": {"rows": [2, 2], "columns": [3, 3]},
        }
    )

    counts = Counter(metadata.block_types)
    assert set(DEFAULT_BLOCK_TYPES).issubset(counts)
    assert "- " in markdown
    assert "1. " in markdown
    assert "- [ ]" in markdown or "- [x]" in markdown
    assert "> " in markdown
    assert "```" in markdown
    assert "![Figure]" in markdown
    assert "---" in markdown
    assert "$$ x^2 + y^2 = z^2 $$" in markdown


def test_content_specs_table_schemas_plumb_into_table_blocks() -> None:
    random.seed(4)
    composer = DocumentComposer(
        data=DataProvider(lang="ko", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "x=y",
        content_specs={"table_schemas": {"roster": 1.0}},
    )
    markdown, metadata = composer.compose(
        {
            "document_shape": "table_only",
            "section_count": [1, 1],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["table"],
            "required_blocks": ["table"],
            "table": {"rows": [2, 2], "columns": [4, 4]},
        }
    )

    assert metadata.block_types == ["table"]
    assert "성명" in markdown
    assert "소속" in markdown
    assert "직위" in markdown
    assert "연락처" in markdown


def test_content_specs_without_table_schemas_keeps_legacy_table_output() -> None:
    random.seed(4)
    baseline_composer = DocumentComposer(
        data=DataProvider(lang="ko", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "x=y",
    )
    baseline_markdown, _ = baseline_composer.compose(
        {
            "document_shape": "table_only",
            "section_count": [1, 1],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["table"],
            "required_blocks": ["table"],
            "table": {"rows": [2, 2], "columns": [4, 4]},
        }
    )

    random.seed(4)
    scaled_composer = DocumentComposer(
        data=DataProvider(lang="ko", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "x=y",
        content_specs={"section_count_scale": 1.0},
    )
    scaled_markdown, _ = scaled_composer.compose(
        {
            "document_shape": "table_only",
            "section_count": [1, 1],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["table"],
            "required_blocks": ["table"],
            "table": {"rows": [2, 2], "columns": [4, 4]},
        }
    )

    assert baseline_markdown == scaled_markdown


def _write_corpus_file(corpus_dir: Path, lang: str, filename: str, lines: list) -> None:
    lang_dir = corpus_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    (lang_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


_GENRE_BLUEPRINT = {
    "section_count": [1, 1],
    "blocks_per_section": [1, 1],
    "allowed_blocks": ["paragraph"],
    "required_blocks": ["paragraph"],
}


def test_composer_uses_genre_corpus_when_mapped_shape_and_probability_hit(tmp_path) -> None:
    document_shape, genre = next(iter(GENRE_BY_DOCUMENT_SHAPE.items()))
    _write_corpus_file(tmp_path, "ko", "paragraphs.txt", ["일반 위키 문단입니다."])
    _write_corpus_file(tmp_path, "ko", f"{genre}.txt", ["장르 코퍼스 문장입니다."])

    random.seed(1)
    composer = DocumentComposer(
        data=DataProvider(lang="ko", mix_ratio=0.0, corpus_dir=tmp_path),
        clip_text=_clip,
        formula_supplier=lambda: "x=y",
        content_specs={"genre_corpus": {"p": 1.0}},
    )
    markdown, metadata = composer.compose({**_GENRE_BLUEPRINT, "document_shape": document_shape})
    body_lines = [line for line in markdown.splitlines() if not line.startswith("#")]

    assert any("장르 코퍼스 문장입니다" in line for line in body_lines)
    assert not any("일반 위키 문단입니다" in line for line in body_lines)
    assert metadata.content_genre == genre


def test_composer_falls_back_to_general_corpus_when_genre_file_absent(tmp_path) -> None:
    document_shape, genre = next(iter(GENRE_BY_DOCUMENT_SHAPE.items()))
    _write_corpus_file(tmp_path, "ko", "paragraphs.txt", ["일반 위키 문단입니다."])

    random.seed(1)
    composer = DocumentComposer(
        data=DataProvider(lang="ko", mix_ratio=0.0, corpus_dir=tmp_path),
        clip_text=_clip,
        formula_supplier=lambda: "x=y",
        content_specs={"genre_corpus": {"p": 1.0}},
    )
    markdown, metadata = composer.compose({**_GENRE_BLUEPRINT, "document_shape": document_shape})

    assert "일반 위키 문단입니다" in markdown
    assert metadata.content_genre is None


def test_composer_without_content_specs_never_uses_genre_corpus(tmp_path) -> None:
    """Legacy unchanged: no `content_specs` at all (no profile) must behave
    exactly as before genre corpora existed, even when a genre file exists
    for a mapped document shape."""
    document_shape, genre = next(iter(GENRE_BY_DOCUMENT_SHAPE.items()))
    _write_corpus_file(tmp_path, "ko", "paragraphs.txt", ["일반 위키 문단입니다."])
    _write_corpus_file(tmp_path, "ko", f"{genre}.txt", ["장르 코퍼스 문장입니다."])

    random.seed(1)
    composer = DocumentComposer(
        data=DataProvider(lang="ko", mix_ratio=0.0, corpus_dir=tmp_path),
        clip_text=_clip,
        formula_supplier=lambda: "x=y",
    )
    markdown, metadata = composer.compose({**_GENRE_BLUEPRINT, "document_shape": document_shape})

    assert "일반 위키 문단입니다" in markdown
    assert metadata.content_genre is None


def test_composer_ignores_genre_corpus_for_unmapped_document_shape(tmp_path) -> None:
    _write_corpus_file(tmp_path, "ko", "paragraphs.txt", ["일반 위키 문단입니다."])
    for genre in set(GENRE_BY_DOCUMENT_SHAPE.values()):
        _write_corpus_file(tmp_path, "ko", f"{genre}.txt", ["장르 코퍼스 문장입니다."])

    random.seed(1)
    composer = DocumentComposer(
        data=DataProvider(lang="ko", mix_ratio=0.0, corpus_dir=tmp_path),
        clip_text=_clip,
        formula_supplier=lambda: "x=y",
        content_specs={"genre_corpus": {"p": 1.0}},
    )
    markdown, metadata = composer.compose(
        {**_GENRE_BLUEPRINT, "document_shape": "technical_manual"}
    )

    assert "일반 위키 문단입니다" in markdown
    assert metadata.content_genre is None


def test_composer_output_is_deterministic_with_seed() -> None:
    blueprint = {
        "document_shape": "release_note",
        "section_count": [5, 5],
        "blocks_per_section": [1, 1],
        "allowed_blocks": ["paragraph", "bullet_list", "checklist", "rule"],
        "required_blocks": ["checklist"],
    }

    random.seed(123)
    first_composer = DocumentComposer(
        data=DataProvider(lang="en", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "a=b",
    )
    first_markdown, first_metadata = first_composer.compose(blueprint)

    random.seed(123)
    second_composer = DocumentComposer(
        data=DataProvider(lang="en", mix_ratio=0.0, use_corpus=False),
        clip_text=_clip,
        formula_supplier=lambda: "a=b",
    )
    second_markdown, second_metadata = second_composer.compose(blueprint)

    assert first_markdown == second_markdown
    assert first_metadata.to_dict() == second_metadata.to_dict()
