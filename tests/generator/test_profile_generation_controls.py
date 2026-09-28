"""Distribution-profile controls on template selection, text source and rules."""

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.cli import generate as generate_cli
from src.generation.options import GenerationOptions, GenerationTaskContext, PublishOptions
from src.generation.readme_builder import build_dataset_readme
from src.generator.data_provider import DataProvider
from src.generator.distribution_profile import DistributionProfile, mm_to_css_px
from src.generator.generator import Generator
from src.generator.markdown_content import MarkdownDataGenerator
from src.generator.markdown_render_utils import MarkdownStyle
from src.generator.markdown_renderers import HtmlMarkdownRenderer, MarkdownRenderer
from src.generator.profile_application import plan_profile_render

_CHANNELS = {"capture_channels": {"born_digital": {"weight": 1.0, "dpi": 150}}}


def _write_profile(tmp_path: Path, body: str) -> str:
    path = tmp_path / "profile.yaml"
    path.write_text(
        body + "\ncapture_channels:\n  born_digital:\n    weight: 1.0\n    dpi: 150\n",
        encoding="utf-8",
    )
    return str(path)


def _generator(tmp_path: Path, *, with_corpus: bool = True) -> Generator:
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "Body.ttf").write_bytes(b"")
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "ko").mkdir(parents=True)
    if with_corpus:
        (corpus_dir / "ko" / "paragraphs.txt").write_text(
            "첫 번째 문단입니다. 두 번째 문장입니다.\n세 번째 문단입니다.\n",
            encoding="utf-8",
        )
    generator = Generator(output_dir=str(tmp_path / "out"), font_dir=str(font_dir), lang="ko")
    generator.data_generator = MarkdownDataGenerator(
        "ko", data_provider=DataProvider(lang="ko", corpus_dir=corpus_dir)
    )
    return generator


def _select_many(generator: Generator, count: int) -> tuple[Counter, Counter]:
    """Drive template selection the way ``generate_single`` updates its counters."""
    random.seed(0)
    for _ in range(count):
        spec, _ = generator._select_template_spec()
        generator.template_counts[spec.template_id] += 1
        generator.family_counts[spec.family] += 1
    return generator.family_counts, generator.template_counts


def test_profile_family_mix_sets_family_shares(tmp_path) -> None:
    generator = _generator(tmp_path)
    generator._configure_generation(seed=1, distribution_profile="real_world_v2")

    families, _ = _select_many(generator, 3000)

    shares = {family: count / 3000 for family, count in families.items()}
    targets = {"business": 0.38, "technical": 0.20, "academic": 0.17, "forms": 0.15, "operations": 0.10}
    assert set(shares) == set(targets)
    for family, target in targets.items():
        assert shares[family] == pytest.approx(target, abs=0.03), family


def test_explicit_coverage_targets_override_profile_family_mix(tmp_path) -> None:
    generator = _generator(tmp_path)
    generator._configure_generation(
        seed=1,
        distribution_profile="real_world_v2",
        coverage_targets=["sections=0.5"],
    )

    families, _ = _select_many(generator, 300)

    # `sections` is not in the profile mix, so it only appears because the
    # explicit --coverage-target replaced the profile's family sampling.
    assert families["sections"] > 0


def test_profile_template_weight_zero_excludes_template(tmp_path) -> None:
    profile_path = _write_profile(
        tmp_path,
        "id: t\nfamily_mix: {academic: 1.0}\ntemplate_weights: {formula_heavy: 0}\n",
    )
    generator = _generator(tmp_path)
    generator._configure_generation(seed=1, distribution_profile=profile_path)

    _, templates = _select_many(generator, 200)

    assert templates == Counter({"academic_note": 200})


def test_profile_template_weights_set_shares_within_family(tmp_path) -> None:
    # academic_note has catalog weight 1.0, formula_heavy 0.8 x 0.25 = 0.2,
    # so formula_heavy should get 0.2 / 1.2 of the academic pages.
    profile_path = _write_profile(
        tmp_path,
        "id: t\nfamily_mix: {academic: 1.0}\ntemplate_weights: {formula_heavy: 0.25}\n",
    )
    generator = _generator(tmp_path)
    generator._configure_generation(seed=1, distribution_profile=profile_path)

    _, templates = _select_many(generator, 3000)

    assert templates["formula_heavy"] / 3000 == pytest.approx(0.2 / 1.2, abs=0.03)


def test_explicit_template_request_wins_over_profile_exclusion(tmp_path) -> None:
    profile_path = _write_profile(
        tmp_path,
        "id: t\nfamily_mix: {academic: 1.0}\ntemplate_weights: {formula_heavy: 0}\n",
    )
    generator = _generator(tmp_path)
    generator._configure_generation(seed=1, distribution_profile=profile_path, template="formula_heavy")

    _, templates = _select_many(generator, 20)

    assert templates == Counter({"formula_heavy": 20})


def test_profile_without_paragraph_corpus_is_rejected(tmp_path) -> None:
    generator = _generator(tmp_path, with_corpus=False)

    with pytest.raises(RuntimeError, match="corpus"):
        generator._configure_generation(seed=1, distribution_profile="real_world_v2")


def test_missing_corpus_is_allowed_without_profile(tmp_path) -> None:
    generator = _generator(tmp_path, with_corpus=False)

    generator._configure_generation(seed=1)

    assert generator.distribution_profile is None


@pytest.mark.parametrize(
    ("profile_ratio", "requested", "expected"),
    [
        (0.0, None, 0.0),  # profile value applies when the CLI flag is not given
        (0.0, 0.2, 0.2),  # an explicit flag wins over the profile
        (None, None, 0.08),  # neither: legacy default
    ],
)
def test_similar_char_ratio_resolution(tmp_path, profile_ratio, requested, expected) -> None:
    content = "" if profile_ratio is None else f"content: {{similar_char_ratio: {profile_ratio}}}\n"
    profile_path = _write_profile(tmp_path, "id: t\n" + content)
    generator = _generator(tmp_path)

    generator._configure_generation(
        seed=1,
        distribution_profile=profile_path,
        similar_char_ratio=requested,
    )

    assert generator.similar_char_ratio == pytest.approx(expected)


def test_similar_char_ratio_defaults_without_profile(tmp_path) -> None:
    generator = _generator(tmp_path)

    generator._configure_generation(seed=1, similar_char_ratio=None)

    assert generator.similar_char_ratio == pytest.approx(0.08)


def test_generate_cli_leaves_similar_char_ratio_unset_by_default() -> None:
    parser = argparse.ArgumentParser()
    generate_cli.add_arguments(parser)

    context = generate_cli.build_context_from_args(parser.parse_args(["--lang", "ko", "--size", "1"]))

    assert context.generation.similar_char_ratio is None
    restored = GenerationOptions.from_dict(context.generation.to_dict())
    assert restored.similar_char_ratio is None


def test_dataset_card_omits_unset_similar_char_ratio_flag() -> None:
    context = GenerationTaskContext(
        lang="ko",
        size=10,
        generation=GenerationOptions(distribution_profile="real_world_v2"),
        publish=PublishOptions(repo_id="u/x"),
    )

    card = build_dataset_readme("u/x", context, 10, {"train": 10})

    assert "--similar-char-ratio" not in card
    assert "--distribution-profile real_world_v2" in card


def _css_block(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, selector
    return match.group(1)


@pytest.mark.parametrize(
    ("typography", "expected_rule"),
    [
        ({"ink_gray": 0}, "rgb(0, 0, 0)"),  # profile ink colour
        ({}, "rgb(12, 34, 56)"),  # no ink_gray: the sampled text colour
    ],
)
def test_profile_rules_use_text_ink_colour(typography, expected_rule) -> None:
    profile = DistributionProfile.from_dict({"id": "t", "typography": typography, **_CHANNELS})
    style = MarkdownStyle(text_color=(12, 34, 56))
    plan_profile_render(profile, style, random.Random(0))

    css = HtmlMarkdownRenderer("font.ttf", style)._build_html_document("| a |\n| --- |\n| b |\n\n---\n")

    assert f"border: 1px solid {expected_rule}" in _css_block(css, ".markdown-body th, .markdown-body td")
    assert f"border-top: 1px solid {expected_rule}" in _css_block(css, ".markdown-body hr")


def test_default_style_keeps_light_table_rules() -> None:
    css = HtmlMarkdownRenderer("font.ttf", MarkdownStyle())._build_html_document("| a |\n| --- |\n| b |\n")

    assert "border: 1px solid rgba(0, 0, 0, 0.25)" in _css_block(css, ".markdown-body th, .markdown-body td")


def test_pil_renderer_draws_hr_in_rule_colour() -> None:
    style = MarkdownStyle(add_noise=False, rule_color=(0, 0, 0))
    image = MarkdownRenderer("missing-font.ttf", style).render("---")

    assert int(np.asarray(image.convert("L")).min()) == 0


def test_heading_and_code_font_faces_emitted_when_distinct_paths_given() -> None:
    renderer = HtmlMarkdownRenderer(
        "body.ttf",
        MarkdownStyle(),
        heading_font_path="heading.ttf",
        code_font_path="code.ttf",
    )

    css = renderer._build_html_document("# Title\n\n```\ncode\n```\n")

    assert css.count("@font-face") == 3
    assert "font-family: 'RenderFontHeading';" in css
    assert "font-family: 'RenderFontCode';" in css
    heading_rule = _css_block(css, ".markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body th")
    assert "'RenderFontHeading'" in heading_rule
    code_rule = _css_block(css, ".markdown-body pre, .markdown-body code")
    assert "'RenderFontCode'" in code_rule


def test_default_font_faces_reuse_render_font_without_new_keys() -> None:
    renderer = HtmlMarkdownRenderer("body.ttf", MarkdownStyle())

    css = renderer._build_html_document("# Title\n")

    assert css.count("@font-face") == 1
    assert "RenderFontHeading" not in css
    assert "RenderFontCode" not in css


def test_text_align_and_word_break_css_from_style() -> None:
    style = MarkdownStyle(text_align="justify", word_break="keep-all")
    css = HtmlMarkdownRenderer("font.ttf", style)._build_html_document("para\n")

    body_rule = _css_block(css, ".markdown-body")
    assert "text-align: justify;" in body_rule
    assert "word-break: keep-all;" in body_rule


def test_default_style_keeps_legacy_word_break_and_no_text_align() -> None:
    css = HtmlMarkdownRenderer("font.ttf", MarkdownStyle())._build_html_document("para\n")

    body_rule = _css_block(css, ".markdown-body")
    assert "word-break: break-word;" in body_rule
    assert "text-align:" not in body_rule


def test_profile_table_style_applied_to_style() -> None:
    profile = DistributionProfile.from_dict(
        {"id": "t", "typography": {"table_style": "grid"}, **_CHANNELS}
    )
    style = MarkdownStyle()

    plan_profile_render(profile, style, random.Random(0))

    assert style.table_style == "grid"


def test_profile_without_table_style_key_leaves_legacy_default() -> None:
    profile = DistributionProfile.from_dict({"id": "t", "typography": {}, **_CHANNELS})
    style = MarkdownStyle()

    plan_profile_render(profile, style, random.Random(0))

    assert style.table_style is None


def test_profile_columns_and_gap_applied_to_style() -> None:
    profile = DistributionProfile.from_dict(
        {"id": "t", "page": {"columns": 2, "column_gap_mm": 10.5}, **_CHANNELS}
    )
    style = MarkdownStyle()
    width_css = style.margin_left + style.content_width + style.margin_right

    plan_profile_render(profile, style, random.Random(0))

    assert style.columns == 2
    assert style.column_gap == mm_to_css_px(10.5, width_css)


def test_profile_without_columns_key_leaves_single_column() -> None:
    profile = DistributionProfile.from_dict({"id": "t", "page": {}, **_CHANNELS})
    style = MarkdownStyle()

    plan_profile_render(profile, style, random.Random(0))

    assert style.columns == 1
    assert style.column_gap is None


def test_profile_page_width_unchanged_when_columns_set() -> None:
    profile = DistributionProfile.from_dict(
        {"id": "t", "page": {"columns": 2, "column_gap_mm": 8}, **_CHANNELS}
    )
    style = MarkdownStyle()
    width_before = style.margin_left + style.content_width + style.margin_right

    plan_profile_render(profile, style, random.Random(0))

    width_after = style.margin_left + style.content_width + style.margin_right
    assert width_after == width_before


def test_profile_columns_family_conditioning_overrides_default_share() -> None:
    profile = DistributionProfile.from_dict(
        {
            "id": "t",
            "page": {
                "columns": {
                    "default": {"choices": [1, 2], "weights": [1.0, 0.0]},
                    "by_family": {"academic": {"choices": [1, 2], "weights": [0.0, 1.0]}},
                }
            },
            **_CHANNELS,
        }
    )

    academic_columns = set()
    other_columns = set()
    for seed in range(50):
        style = MarkdownStyle()
        plan_profile_render(profile, style, random.Random(seed), family="academic")
        academic_columns.add(style.columns)

        style = MarkdownStyle()
        plan_profile_render(profile, style, random.Random(seed), family="forms")
        other_columns.add(style.columns)

    assert academic_columns == {2}
    assert other_columns == {1}


def test_family_kwarg_is_optional_and_uses_default_share() -> None:
    profile = DistributionProfile.from_dict(
        {
            "id": "t",
            "page": {"columns": {"default": {"choices": [1, 2], "weights": [0.0, 1.0]}, "by_family": {}}},
            **_CHANNELS,
        }
    )
    style = MarkdownStyle()

    plan_profile_render(profile, style, random.Random(0))

    assert style.columns == 2


def test_distribution_profile_content_keys_drive_korean_conventions(tmp_path) -> None:
    profile_path = _write_profile(
        tmp_path,
        "id: t\nfamily_mix: {business: 1.0}\n"
        "content:\n"
        "  heading_numbering: korean_admin\n"
        "  list_style: korean_admin\n"
        "  law_articles: {p: 1.0}\n",
    )
    generator = _generator(tmp_path)
    generator._configure_generation(
        seed=1, distribution_profile=profile_path, template="policy_document"
    )

    spec, _ = generator._select_template_spec()
    random.seed(2)
    markdown_text = generator.data_generator.generate_markdown(
        template_id=spec.template_id, template_spec=spec
    )
    metadata = generator.data_generator.pop_composition_metadata()

    assert metadata["heading_numbering"] == "korean_admin"
    assert metadata["list_style"] == "korean_admin"
    assert metadata["law_articles_used"] is True
    assert "제1조(" in markdown_text


def test_without_distribution_profile_korean_conventions_stay_legacy(tmp_path) -> None:
    generator = _generator(tmp_path)
    generator._configure_generation(seed=1, template="policy_document")

    spec, _ = generator._select_template_spec()
    random.seed(2)
    generator.data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    metadata = generator.data_generator.pop_composition_metadata()

    assert metadata["heading_numbering"] == "none"
    assert metadata["list_style"] == "markdown"
    assert metadata["law_articles_used"] is False
