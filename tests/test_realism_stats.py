import json
import random

import pytest
from PIL import Image, ImageDraw

from src.generation.distribution_summary import (
    format_distribution_tables,
    summarize_metadata_distribution,
)
from src.generator.degradation import apply_capture_degradation
from src.realism.distribution_stats import compare_summaries, format_comparison_markdown, summarize_stats
from src.realism.image_stats import compute_image_stats, estimate_noise_sigma


def _text_page() -> Image.Image:
    image = Image.new("RGB", (800, 1000), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for row in range(80, 920, 28):
        draw.rectangle((80, row, 720, row + 10), fill=(15, 15, 15))
    return image


@pytest.mark.parametrize("angle", [-2.0, 0.0, 1.5])
def test_skew_estimate_recovers_applied_rotation(angle: float) -> None:
    rotated = apply_capture_degradation(_text_page(), {"skew_deg": angle}, random.Random(0))
    stats = compute_image_stats(rotated)
    assert stats["skew_deg"] == pytest.approx(angle, abs=0.3)


def test_noise_estimate_increases_with_noise() -> None:
    clean = compute_image_stats(_text_page())["noise_sigma"]
    noisy_image = apply_capture_degradation(_text_page(), {"noise_sigma": 10.0}, random.Random(0))
    noisy = compute_image_stats(noisy_image)["noise_sigma"]
    assert noisy > clean + 3.0


def test_noise_estimate_handles_tiny_images() -> None:
    import numpy as np

    assert estimate_noise_sigma(np.zeros((2, 2), dtype=np.uint8)) == 0.0


def test_binary_and_grayscale_flags() -> None:
    stats = compute_image_stats(apply_capture_degradation(_text_page(), {"binarize": True}, random.Random(0)))
    assert stats["is_binary"] == 1.0
    assert stats["is_grayscale"] == 1.0


def test_compare_summaries_flags_distribution_shift() -> None:
    reference_rows = [compute_image_stats(_text_page()) for _ in range(3)]
    shifted_rows = [
        compute_image_stats(apply_capture_degradation(_text_page(), {"skew_deg": 3.0, "noise_sigma": 8.0}, random.Random(i)))
        for i in range(3)
    ]
    reference = summarize_stats(reference_rows, source="real")
    candidate = summarize_stats(shifted_rows, source="synthetic")
    rows = {row["metric"]: row for row in compare_summaries(reference, candidate)}
    assert rows["abs_skew_deg"]["candidate_is"] == "higher"
    assert rows["noise_sigma"]["verdict"] != "close"
    report = format_comparison_markdown(list(rows.values()), "real", "synthetic")
    assert "| abs_skew_deg |" in report


def test_suggested_specs_are_valid_distribution_specs() -> None:
    from src.generator.distribution_profile import sample_value

    rows = [compute_image_stats(apply_capture_degradation(_text_page(), {"skew_deg": a}, random.Random(0))) for a in (-1, 0, 1)]
    specs = summarize_stats(rows)["suggested_profile_specs"]
    rng = random.Random(0)
    for spec in specs.values():
        sample_value(spec, rng)


def test_metadata_distribution_summary(tmp_path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    rows = [
        {"document_family": "forms", "capture_channel": "scanned", "block_type_counts": {"table": 2}},
        {"document_family": "business", "capture_channel": "scanned", "block_type_counts": {"paragraph": 3}},
    ]
    metadata.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    summary = summarize_metadata_distribution(metadata)
    assert summary["capture_channel"] == {"scanned": 2}
    assert summary["block_type"] == {"paragraph": 3, "table": 2}
    tables = "\n".join(format_distribution_tables(summary))
    assert "| scanned | 2 | 100.0% |" in tables
    assert summarize_metadata_distribution(tmp_path / "missing.jsonl") == {}
