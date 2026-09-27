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
    mm_to_css_px,
    points_to_css_px,
    render_scale_for_dpi,
    sample_value,
)
from src.generator.document_blocks import DocumentComposer, parse_block_blueprint
from src.generator.markdown_render_utils import MarkdownStyle
from src.generator.profile_application import (
    finalize_profile_image,
    plan_profile_render,
    select_render_fonts,
)
from src.generation.options import GenerationOptions

_CHANNELS = {"capture_channels": {"born_digital": {"weight": 1.0, "dpi": 150}}}


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



def test_fit_markdown_to_sheet_drops_trailing_blocks() -> None:
    from src.generator.profile_application import fit_markdown_to_sheet

    markdown = "# Title\n\n## A\n\npara one\n\n## B\n\npara two\n\n## C\n\n- item\n- item\n"
    trimmed, kept = fit_markdown_to_sheet(markdown, overflow_ratio=2.0)
    assert trimmed.startswith("# Title")
    assert not trimmed.rstrip().splitlines()[-1].startswith("#")
    assert kept == sum(1 for chunk in trimmed.strip().split("\n\n") if not chunk.startswith("#"))
    assert len(trimmed) < len(markdown)
    assert fit_markdown_to_sheet(markdown, overflow_ratio=0.9)[0] == markdown


def test_pad_to_aspect_extends_short_pages_only() -> None:
    from src.generator.profile_application import pad_to_aspect

    short = pad_to_aspect(_page(400, 200), 1.414)
    assert short.size == (400, 566)
    tall = _page(400, 900)
    assert pad_to_aspect(tall, 1.414).size == (400, 900)


def test_fit_markdown_to_sheet_keeps_first_content_block() -> None:
    from src.generator.profile_application import fit_markdown_to_sheet

    markdown = "# Title\n\n## Section\n\n" + ("very long paragraph " * 200) + "\n\n## Next\n\nmore\n"
    trimmed, kept = fit_markdown_to_sheet(markdown, overflow_ratio=6.0)
    assert kept == 1
    assert "very long paragraph" in trimmed
    assert "## Next" not in trimmed


def test_mm_to_css_px_scales_with_page_width() -> None:
    # Half of A4's 210mm width maps to half the CSS page width.
    assert mm_to_css_px(105, 210) == 105
    assert mm_to_css_px(210, 210) == 210
    # 30mm HWP left/right margin on a 680px virtual page.
    assert mm_to_css_px(30, 680) == round(30 / 210 * 680)
    assert mm_to_css_px(0, 680) == 0


def test_plan_profile_render_converts_margins_mm_to_css_px() -> None:
    profile = DistributionProfile.from_dict(
        {
            "id": "t",
            "page": {
                "margins_mm": {"top": 35, "bottom": 15, "left": 30, "right": 30},
            },
            **_CHANNELS,
        }
    )
    style = MarkdownStyle(margin_top=40, margin_bottom=40, margin_left=40, margin_right=40, content_width=600)
    width_css = 40 + 600 + 40  # page width captured before margins mutate it

    plan_profile_render(profile, style, random.Random(0))

    assert style.margin_top == mm_to_css_px(35, width_css)
    assert style.margin_bottom == mm_to_css_px(15, width_css)
    assert style.margin_left == mm_to_css_px(30, width_css)
    assert style.margin_right == mm_to_css_px(30, width_css)


def test_plan_profile_render_keeps_page_width_constant_after_margins_mm() -> None:
    # render_scale_for_dpi and the body-font-size conversion are both derived
    # from the page width captured *before* margins_mm is applied; left/right
    # margins must shrink content_width by the same amount they grow, or the
    # page comes out wider than the DPI scale assumes (and body text ends up
    # physically smaller than body_font_pt).
    profile = DistributionProfile.from_dict(
        {
            "id": "t",
            "page": {"margins_mm": {"top": 35, "bottom": 15, "left": 30, "right": 30}},
            **_CHANNELS,
        }
    )
    style = MarkdownStyle(margin_top=40, margin_bottom=40, margin_left=34, margin_right=34, content_width=620)
    width_before = style.margin_left + style.content_width + style.margin_right

    plan_profile_render(profile, style, random.Random(0))

    assert style.margin_left + style.content_width + style.margin_right == width_before
    assert style.margin_left == mm_to_css_px(30, width_before)
    assert style.margin_right == mm_to_css_px(30, width_before)


def test_plan_profile_render_clamps_absurd_margins_without_negative_content_width() -> None:
    profile = DistributionProfile.from_dict(
        {
            "id": "t",
            "page": {"margins_mm": {"left": 500, "right": 500}},
            **_CHANNELS,
        }
    )
    style = MarkdownStyle(margin_left=34, margin_right=34, content_width=620)
    width_before = style.margin_left + style.content_width + style.margin_right

    plan_profile_render(profile, style, random.Random(0))

    assert style.content_width > 0
    assert style.margin_left + style.content_width + style.margin_right == width_before


def test_plan_profile_render_final_pixel_width_matches_target_dpi() -> None:
    # With the page-width invariant held, the rendered device-pixel width is
    # target_dpi * A4_WIDTH_INCH regardless of the base style's page width in
    # CSS px (e.g. 300 dpi -> ~2481px), matching what a real A4 sheet at that
    # DPI would produce.
    profile = DistributionProfile.from_dict(
        {
            "id": "t",
            "page": {"margins_mm": {"left": 30, "right": 30}},
            "capture_channels": {"born_digital": {"weight": 1.0, "dpi": 300}},
        }
    )
    style = MarkdownStyle(margin_left=34, margin_right=34, content_width=620)

    plan_profile_render(profile, style, random.Random(0))

    page_width_css = style.margin_left + style.content_width + style.margin_right
    device_px_width = page_width_css * style.render_scale
    assert device_px_width == pytest.approx(300 * 8.27, abs=2)


def test_plan_profile_render_leaves_margins_unchanged_without_margins_mm() -> None:
    profile = load_distribution_profile("real_world_v1")
    style = MarkdownStyle(margin_top=40, margin_bottom=40, margin_left=34, margin_right=34, content_width=620)

    plan_profile_render(profile, style, random.Random(0))

    assert (style.margin_top, style.margin_bottom, style.margin_left, style.margin_right) == (40, 40, 34, 34)


def test_plan_profile_render_sets_text_align_and_word_break_from_typography() -> None:
    profile = DistributionProfile.from_dict(
        {"id": "t", "typography": {"text_align": "justify", "word_break": "keep-all"}, **_CHANNELS}
    )
    style = MarkdownStyle()

    plan_profile_render(profile, style, random.Random(0))

    assert style.text_align == "justify"
    assert style.word_break == "keep-all"


def test_plan_profile_render_leaves_text_align_and_word_break_unset_without_keys() -> None:
    profile = load_distribution_profile("real_world_v1")
    style = MarkdownStyle()

    plan_profile_render(profile, style, random.Random(0))

    assert style.text_align is None
    assert style.word_break is None


def test_choose_font_picks_group_by_weight_then_file_uniformly() -> None:
    profile = DistributionProfile.from_dict({"id": "t", **_CHANNELS})
    paths = [
        "/fonts/NanumMyeongjo.ttf",
        "/fonts/NanumMyeongjoBold.ttf",
        "/fonts/batang-Regular.ttf",
        "/fonts/NanumGothic.ttf",
        "/fonts/NanumSquareB.ttf",
    ]
    groups = {"Myeongjo": 3, "batang": 1}
    rng = random.Random(7)

    picks = [profile.choose_font(paths, groups, rng) for _ in range(2000)]

    assert all(pick is not None for pick in picks)
    assert set(picks) <= {
        "/fonts/NanumMyeongjo.ttf",
        "/fonts/NanumMyeongjoBold.ttf",
        "/fonts/batang-Regular.ttf",
    }
    myeongjo_share = sum(1 for p in picks if "Myeongjo" in p) / len(picks)
    assert 0.68 < myeongjo_share < 0.82  # weight 3 of (3 + 1)


def test_choose_font_returns_none_when_no_group_matches() -> None:
    profile = DistributionProfile.from_dict({"id": "t", **_CHANNELS})
    paths = ["/fonts/NanumGothic.ttf", "/fonts/NanumSquareB.ttf"]

    assert profile.choose_font(paths, {"Ipsum": 1}, random.Random(0)) is None
    assert profile.choose_font(paths, {}, random.Random(0)) is None


def test_choose_font_apply_exclude_flag_controls_fonts_exclude() -> None:
    profile = DistributionProfile.from_dict(
        {"id": "t", "fonts": {"exclude": ["D2Coding"]}, **_CHANNELS}
    )
    paths = ["/fonts/D2Coding-Regular.ttf", "/fonts/NanumGothic.ttf"]
    groups = {"D2Coding": 1}

    assert profile.choose_font(paths, groups, random.Random(0), apply_exclude=True) is None
    picked = profile.choose_font(paths, groups, random.Random(0), apply_exclude=False)
    assert picked == "/fonts/D2Coding-Regular.ttf"


def test_select_render_fonts_legacy_choice_without_profile() -> None:
    paths = ["/fonts/A.ttf", "/fonts/B.ttf"]

    body, heading, code = select_render_fonts(None, paths, random.Random(0))

    assert body in paths
    assert heading is None
    assert code is None


def test_select_render_fonts_legacy_choice_without_font_groups() -> None:
    profile = DistributionProfile.from_dict({"id": "t", **_CHANNELS})
    paths = ["/fonts/A.ttf", "/fonts/B.ttf"]

    body, heading, code = select_render_fonts(profile, paths, random.Random(0))

    assert body in paths
    assert heading is None
    assert code is None


def test_select_render_fonts_uses_body_heading_code_groups() -> None:
    profile = DistributionProfile.from_dict(
        {
            "id": "t",
            "fonts": {
                "body": {"Myeongjo": 1},
                "heading": {"Gothic": 1},
                "code": ["D2Coding"],
            },
            **_CHANNELS,
        }
    )
    paths = [
        "/fonts/NanumMyeongjo.ttf",
        "/fonts/NanumGothic.ttf",
        "/fonts/D2Coding-Regular.ttf",
    ]

    body, heading, code = select_render_fonts(profile, paths, random.Random(0))

    assert body == "/fonts/NanumMyeongjo.ttf"
    assert heading == "/fonts/NanumGothic.ttf"
    assert code == "/fonts/D2Coding-Regular.ttf"


def test_paragraph_builder_neutralizes_markdown_prefixes() -> None:
    from src.generator.document_blocks import _neutralize_block_markup

    assert _neutralize_block_markup("# 수행 유도하기 예") == "수행 유도하기 예"
    assert _neutralize_block_markup("> - 1. 인용") == "인용"
    assert _neutralize_block_markup("2009년 7월") == "2009년 7월"
