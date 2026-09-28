"""Tests for `distribution measure --save-rows`: per-image rows + paths."""

from __future__ import annotations

import argparse
import json

from PIL import Image, ImageDraw

from src.cli import distribution


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    distribution.configure_parser(parser)
    return parser.parse_args(argv)


def _write_page(path) -> None:
    image = Image.new("RGB", (400, 500), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for row in range(40, 460, 20):
        draw.rectangle((40, row, 360, row + 6), fill=(10, 10, 10))
    image.save(path)


def test_measure_without_save_rows_omits_rows(tmp_path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_page(images_dir / "a.png")
    _write_page(images_dir / "b.png")
    output = tmp_path / "stats.json"

    args = _parse(["measure", "--images", str(images_dir), "--output", str(output)])
    rc = distribution.run_measure(args)
    assert rc == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert "rows" not in summary


def test_measure_with_save_rows_includes_rows_and_paths(tmp_path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_page(images_dir / "a.png")
    _write_page(images_dir / "b.png")
    output = tmp_path / "stats.json"

    args = _parse(
        ["measure", "--images", str(images_dir), "--output", str(output), "--save-rows"]
    )
    rc = distribution.run_measure(args)
    assert rc == 0

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert "rows" in summary
    assert len(summary["rows"]) == 2
    paths = {row["path"] for row in summary["rows"]}
    assert paths == {str(images_dir / "a.png"), str(images_dir / "b.png")}
    for row in summary["rows"]:
        assert "est_dpi_a4" in row
        assert "text_line_count" in row

    # `path` must not leak into the numeric metric summary.
    assert "path" not in summary["metrics"]
