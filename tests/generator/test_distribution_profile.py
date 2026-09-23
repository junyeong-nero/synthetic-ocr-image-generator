import random
from collections import Counter

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.generator.degradation import apply_capture_degradation
from src.generator.distribution_profile import (
    DistributionProfile,
    available_profiles,
    load_distribution_profile,
    points_to_css_px,
    render_scale_for_dpi,
    sample_value,
)
from src.generator.document_blocks import DocumentComposer, parse_block_blueprint
from src.generator.markdown_render_utils import MarkdownStyle
from src.generator.profile_application import finalize_profile_image, plan_profile_render
from src.generation.options import GenerationOptions


def _page(width: int = 400, height: int = 520) -> Image.Image:
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for row in range(40, height - 40, 24):
        draw.rectangle((40, row, width - 40, row + 8), fill=(20, 20, 20))
    return image


def test_sample_value_supports_all_spec_forms() -> None:
    rng = random.Random(0)
    assert sample_value(3, rng) == 3
    assert sample_value(None, rng) is None
    assert 1.0 <= sample_value([1, 2], rng) <= 2.0
    assert sample_value({"choices": ["a"], "weights": [1]}, rng) == "a"
    assert isinstance(sample_value({"p": 0.5}, rng), bool)
    assert -1.0 <= sample_value({"normal": [0, 10], "clip": [-1, 1]}, rng) <= 1.0
    assert isinstance(sample_value({"lognormal": [0, 0.5], "round": 0}, rng), int)
    assert 2.0 <= sample_value({"beta": [2, 2], "range": [2, 3]}, rng) <= 3.0
    hist = {"histogram": {"bins": [0, 1, 2], "weights": [0, 1]}}
    assert all(1.0 <= sample_value(hist, rng) <= 2.0 for _ in range(50))


def test_sample_value_rejects_malformed_specs() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError):
        sample_value({"unknown": 1}, rng)
    with pytest.raises(ValueError):
        sample_value({"histogram": {"bins": [0, 1], "weights": [1, 2]}}, rng)


@pytest.mark.parametrize("name", ["real_world_v1", "ko_admin_scan_v1"])
def test_bundled_profiles_load_and_sample(name: str) -> None:
    assert name in available_profiles()
    profile = load_distribution_profile(name)
    assert profile.profile_id == name
    assert abs(sum(profile.family_mix.values()) - 1.0) < 1e-9
    rng = random.Random(1)
    channels = Counter(profile.sample_capture(rng).channel for _ in range(400))
    assert set(channels) <= {channel.name for channel in profile.capture_channels}
    typography = profile.sample_typography(rng)
    assert 8.0 <= typography["body_font_pt"] <= 14.0


def test_channel_mix_follows_weights() -> None:
    profile = DistributionProfile.from_dict(
        {
            "id": "t",
            "capture_channels": {
                "a": {"weight": 0.8, "dpi": 100},
                "b": {"weight": 0.2, "dpi": 300, "degradations": {"skew_deg": 1.5}},
            },
        }
    )
    rng = random.Random(3)
    samples = [profile.sample_capture(rng) for _ in range(2000)]
    share_a = sum(sample.channel == "a" for sample in samples) / len(samples)
    assert 0.75 < share_a < 0.85
    b = next(sample for sample in samples if sample.channel == "b")
    assert b.dpi == 300 and b.params == {"skew_deg": 1.5}


def test_profile_requires_capture_channels() -> None:
    with pytest.raises(ValueError):
        DistributionProfile.from_dict({"id": "empty"})


def test_render_scale_and_point_conversion() -> None:
    width_css = 700
    assert render_scale_for_dpi(300, width_css) == pytest.approx(300 / (700 / 8.27))
    assert render_scale_for_dpi(10_000, width_css) == 4.0
    # 10.5pt on a ~85 dpi virtual page is ~12 CSS px.
    assert points_to_css_px(10.5, width_css) == 12


def test_degradation_is_deterministic_and_changes_geometry() -> None:
    params = {
        "skew_deg": 2.0,
        "blur_sigma": 0.8,
        "noise_sigma": 4.0,
        "paper_tint": True,
        "jpeg_quality": 70,
    }
    first = apply_capture_degradation(_page(), params, random.Random(9), channel="scanned")
    second = apply_capture_degradation(_page(), params, random.Random(9), channel="scanned")
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert first.size != (400, 520)  # rotation expands the canvas


def test_degradation_binarize_outputs_two_levels() -> None:
    image = apply_capture_degradation(_page(), {"binarize": True}, random.Random(0))
    assert set(np.unique(np.asarray(image.convert("L")))) <= {0, 255}


def test_degradation_ignores_empty_params() -> None:
    page = _page()
    assert np.array_equal(np.asarray(apply_capture_degradation(page, {}, random.Random(0))), np.asarray(page))


def test_plan_profile_render_updates_style_and_metadata() -> None:
    profile = load_distribution_profile("real_world_v1")
    style = MarkdownStyle(add_noise=True, add_blur=True, background_color=(200, 230, 210))
    plan = plan_profile_render(profile, style, random.Random(5))
    assert style.add_noise is False and style.add_blur is False
    assert style.render_scale == pytest.approx(plan.render_scale)
    metadata = plan.metadata()
    assert metadata["distribution_profile"] == "real_world_v1"
    assert metadata["visual_difficulty"] in {"easy", "medium", "hard"}
    if not metadata["colored_background"]:
        assert style.background_color == (255, 255, 255)

    image = finalize_profile_image(_page(), plan, renderer_applied_scale=False)
    assert image.width > 0 and image.height > 0


def test_block_weights_bias_filler_selection() -> None:
    parsed = parse_block_blueprint(
        {"allowed_blocks": ["paragraph", "table"], "section_count": [1, 1]}
    )
    random.seed(0)
    plan = DocumentComposer._plan_block_types(
        parsed,
        total_slots=500,
        block_weights={"paragraph": 9.0, "table": 1.0},
    )
    share = Counter(plan)["paragraph"] / len(plan)
    assert 0.85 < share < 0.95


def test_block_weights_ignored_when_all_zero() -> None:
    parsed = parse_block_blueprint({"allowed_blocks": ["paragraph", "table"]})
    random.seed(0)
    plan = DocumentComposer._plan_block_types(parsed, total_slots=50, block_weights={"code": 1.0})
    assert set(plan) == {"paragraph", "table"}


def test_generation_options_round_trip_distribution_profile() -> None:
    options = GenerationOptions(distribution_profile="real_world_v1")
    restored = GenerationOptions.from_dict(options.to_dict())
    assert restored.distribution_profile == "real_world_v1"
    assert options.to_generator_kwargs()["distribution_profile"] == "real_world_v1"

