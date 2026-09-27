"""Tests for src/realism/text_stats.py: character-class shares, markdown
stripping, and CLI-facing text iterators."""

from __future__ import annotations

import json

from src.realism.text_stats import (
    classify_char,
    compute_text_stats,
    iter_jsonl_texts,
    iter_metadata_texts,
    iter_text_dir,
    strip_markdown_syntax,
    summarize_text_stats,
    top_symbols,
)


def test_classify_char_covers_expected_classes() -> None:
    assert classify_char("가") == "hangul"
    assert classify_char("漢") == "hanja"
    assert classify_char("あ") == "kana"
    assert classify_char("カ") == "kana"
    assert classify_char("a") == "latin"
    assert classify_char("A") == "latin"
    assert classify_char("5") == "digit"
    assert classify_char(".") == "punctuation"
    assert classify_char(",") == "punctuation"
    assert classify_char(" ") == "whitespace"
    assert classify_char("\n") == "whitespace"
    assert classify_char("□") == "symbol"
    assert classify_char("○") == "symbol"
    assert classify_char("※") == "symbol"
    assert classify_char("①") == "symbol"


def test_compute_text_stats_known_string() -> None:
    text = "가나다 abc 123.!"
    stats = compute_text_stats(text)
    total = len(text)
    assert stats["chars_per_page"] == total
    assert stats["hangul_share"] == 3 / total
    assert stats["latin_share"] == 3 / total
    assert stats["digit_share"] == 3 / total
    # '.' and '!' are punctuation, two spaces are whitespace.
    assert stats["punctuation_share"] == 2 / total
    assert stats["whitespace_share"] == 2 / total


def test_compute_text_stats_mean_line_length() -> None:
    text = "abcde\nab\n\nabcdefgh"
    stats = compute_text_stats(text)
    # Blank lines are ignored; ("abcde"=5, "ab"=2, "abcdefgh"=8) -> mean 5.
    assert stats["mean_line_length"] == 5.0
    assert stats["line_count"] == 3


def test_compute_text_stats_empty_string_does_not_crash() -> None:
    stats = compute_text_stats("")
    assert stats["chars_per_page"] == 0
    assert stats["mean_line_length"] == 0.0
    for key, value in stats.items():
        if key.endswith("_share"):
            assert value == 0.0


def test_strip_markdown_syntax_removes_common_syntax() -> None:
    markdown = (
        "# Heading\n"
        "\n"
        "Some **bold** and *italic* text with a [link](http://x) and `code`.\n"
        "\n"
        "- item one\n"
        "- item two\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
    )
    stripped = strip_markdown_syntax(markdown)
    assert "#" not in stripped
    assert "**" not in stripped
    assert "[link]" not in stripped
    assert "http://x" not in stripped
    assert "|" not in stripped
    assert "Heading" in stripped
    assert "bold" in stripped
    assert "item one" in stripped


def test_top_symbols_ranks_by_frequency() -> None:
    texts = ["※ 주의 ※ ※", "○ 항목 ○"]
    top = top_symbols(texts, top_k=5)
    assert top[0] == ("※", 3)
    assert ("○", 2) in top


def test_summarize_text_stats_matches_compare_compatible_shape() -> None:
    rows = [compute_text_stats("가나다 abc"), compute_text_stats("라마바 def")]
    summary = summarize_text_stats(rows, source="unit-test", top_symbols=[("※", 2)])
    assert summary["source"] == "unit-test"
    assert summary["count"] == 2
    assert "hangul_share" in summary["metrics"]
    assert "quantiles" in summary["metrics"]["hangul_share"]
    assert summary["top_symbols"] == [{"char": "※", "count": 2}]


def test_iter_metadata_texts_strips_markdown(tmp_path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    rows = [
        {"GT_markdown": "# Title\n\nBody **text** here."},
        {"GT_markdown": "- a\n- b\n"},
    ]
    metadata.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    texts = list(iter_metadata_texts(metadata))
    assert len(texts) == 2
    assert "#" not in texts[0]
    assert "**" not in texts[0]
    assert "Title" in texts[0]


def test_iter_jsonl_texts_reads_field(tmp_path) -> None:
    path = tmp_path / "texts.jsonl"
    rows = [{"text": "hello world"}, {"text": "다른 문장"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    texts = list(iter_jsonl_texts(path, field="text"))
    assert texts == ["hello world", "다른 문장"]


def test_iter_text_dir_reads_txt_files(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("first page", encoding="utf-8")
    (tmp_path / "b.txt").write_text("second page", encoding="utf-8")
    texts = sorted(iter_text_dir(tmp_path))
    assert texts == ["first page", "second page"]
