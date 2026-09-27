"""Tests for the `distribution text-stats` CLI subcommand."""

from __future__ import annotations

import argparse
import json

from src.cli import distribution


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    distribution.configure_parser(parser)
    return parser.parse_args(argv)


def test_text_stats_subcommand_is_registered() -> None:
    args = _parse(["text-stats", "--text-dir", ".", "--output", "out.json"])
    assert args.handler is distribution.run_text_stats


def test_run_text_stats_from_metadata(tmp_path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    rows = [
        {"GT_markdown": "# 제목\n\n본문 **강조** 문장입니다."},
        {"GT_markdown": "- 항목 1\n- 항목 2\n"},
    ]
    metadata.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "stats.json"

    args = _parse(["text-stats", "--metadata", str(metadata), "--output", str(output)])
    rc = distribution.run_text_stats(args)
    assert rc == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["count"] == 2
    assert "hangul_share" in summary["metrics"]
    assert "#" not in json.dumps(summary)  # markdown syntax stripped before counting


def test_run_text_stats_from_texts_jsonl(tmp_path) -> None:
    texts_path = tmp_path / "texts.jsonl"
    rows = [{"text": "hello world"}, {"text": "다른 문장 입니다"}]
    texts_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "stats.json"

    args = _parse(
        ["text-stats", "--texts", str(texts_path), "--field", "text", "--output", str(output)]
    )
    rc = distribution.run_text_stats(args)
    assert rc == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["count"] == 2


def test_run_text_stats_from_text_dir(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("first document text", encoding="utf-8")
    (tmp_path / "b.txt").write_text("second document text", encoding="utf-8")
    output = tmp_path / "stats.json"

    args = _parse(["text-stats", "--text-dir", str(tmp_path), "--output", str(output)])
    rc = distribution.run_text_stats(args)
    assert rc == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["count"] == 2


def test_run_text_stats_compare_roundtrip(tmp_path) -> None:
    from src.cli.distribution import run_compare

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    (ref_dir / "a.txt").write_text("가나다라 한국어 문서 텍스트", encoding="utf-8")
    cand_dir = tmp_path / "cand"
    cand_dir.mkdir()
    (cand_dir / "a.txt").write_text("plain latin lorem ipsum text", encoding="utf-8")

    ref_out = tmp_path / "ref.json"
    cand_out = tmp_path / "cand.json"
    distribution.run_text_stats(
        _parse(["text-stats", "--text-dir", str(ref_dir), "--output", str(ref_out)])
    )
    distribution.run_text_stats(
        _parse(["text-stats", "--text-dir", str(cand_dir), "--output", str(cand_out)])
    )

    compare_args = _parse(
        ["compare", "--reference", str(ref_out), "--candidate", str(cand_out)]
    )
    rc = run_compare(compare_args)
    assert rc == 0
