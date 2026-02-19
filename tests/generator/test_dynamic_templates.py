import random
import sys
import importlib
import re
from datetime import datetime
from types import ModuleType

from PIL import Image


stub_character_similarity = ModuleType("character_similarity")


def _stub_find_similar_chars(_char: str, _db, top_n: int = 5):
    return []


setattr(stub_character_similarity, "find_similar_chars", _stub_find_similar_chars)
sys.modules.setdefault("character_similarity", stub_character_similarity)


stub_faker = ModuleType("faker")
stub_faker_config = ModuleType("faker.config")
setattr(stub_faker_config, "AVAILABLE_LOCALES", ["en_US", "ko_KR"])


class _DummyFaker:
    def __init__(self, _locale: str = "en_US"):
        self.locale = _locale

    @staticmethod
    def seed(_seed: int) -> None:
        return None

    def seed_instance(self, _seed: int) -> None:
        return None

    def name(self) -> str:
        return "Jane Doe"

    def first_name(self) -> str:
        return "Jane"

    def last_name(self) -> str:
        return "Doe"

    def address(self) -> str:
        return "100 Main St"

    def city(self) -> str:
        return "Seoul"

    def street_address(self) -> str:
        return "100 Main St"

    def postcode(self) -> str:
        return "00000"

    def company(self) -> str:
        return "Acme Corp"

    def company_suffix(self) -> str:
        return "Inc"

    def phone_number(self) -> str:
        return "010-0000-0000"

    def email(self) -> str:
        return "jane@example.com"

    def date(self, pattern: str = "%Y-%m-%d") -> str:
        return datetime(2024, 1, 1).strftime(pattern)

    def time(self, pattern: str = "%H:%M") -> str:
        return datetime(2024, 1, 1, 9, 0).strftime(pattern)

    def date_time(self) -> datetime:
        return datetime(2024, 1, 1, 9, 0)

    def paragraph(self, nb_sentences: int = 3) -> str:
        return " ".join(["Sample sentence."] * max(1, nb_sentences))

    def sentence(self) -> str:
        return "Sample sentence."

    def sentences(self, nb: int = 3):
        return ["Sample sentence."] * max(1, nb)

    def text(self, max_nb_chars: int = 200) -> str:
        return "x" * min(max_nb_chars, 20)

    def word(self) -> str:
        return "token"

    def words(self, nb: int = 5):
        return ["token"] * max(1, nb)

    def domain_name(self) -> str:
        return "example.com"

    def job(self) -> str:
        return "Engineer"


setattr(stub_faker, "Faker", _DummyFaker)
sys.modules.setdefault("faker", stub_faker)
sys.modules.setdefault("faker.config", stub_faker_config)

generator_module = importlib.import_module("generator.generator")

A4_MAX_HEIGHT_PX = generator_module.A4_MAX_HEIGHT_PX
A4_MAX_WIDTH_PX = generator_module.A4_MAX_WIDTH_PX
Generator = generator_module.Generator
HARD_CODED_FORMULA_EXPRESSIONS = generator_module.HARD_CODED_FORMULA_EXPRESSIONS
HtmlMarkdownRenderer = generator_module.HtmlMarkdownRenderer
MarkdownDataGenerator = generator_module.MarkdownDataGenerator
TemplateCatalog = generator_module.TemplateCatalog
TemplateSpec = generator_module.TemplateSpec
parse_coverage_targets = generator_module.parse_coverage_targets
parse_markdown_formula_line = generator_module.parse_markdown_formula_line


def test_parse_coverage_targets_accepts_list_and_dict() -> None:
    parsed_from_list = parse_coverage_targets(["sections=0.6", "incident:0.2"])
    assert parsed_from_list == {"sections": 0.6, "incident": 0.2}

    parsed_from_dict = parse_coverage_targets({"sections": 0.8, "ops": 2.0})
    assert parsed_from_dict == {"sections": 0.8, "ops": 1.0}


def test_template_catalog_loads_section_schema(tmp_path) -> None:
    config_path = tmp_path / "templates.yaml"
    config_path.write_text(
        """
version: 2
id: default
mode: sections
text:
  section_count: [2, 2]
table:
  section_count: [1, 1]
  rows: [2, 2]
  columns: [3, 3]
formula:
  section_count: [1, 1]
""".strip(),
        encoding="utf-8",
    )

    catalog = TemplateCatalog(config_dir=str(tmp_path))

    resolved = catalog.resolve("default", None, None, None)
    assert len(resolved) == 1
    assert resolved[0].template_id == "default"
    assert resolved[0].mode == "sections"
    assert resolved[0].blueprint["text"]["section_count"] == [2, 2]

    filtered = catalog.resolve(None, "sections", None, None)
    assert filtered
    assert all(spec.family == "sections" for spec in filtered)


def test_sections_generation_composes_text_table_formula_blocks() -> None:
    random.seed(17)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="sections_test",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "text": {"section_count": [2, 2]},
            "table": {"section_count": [1, 1], "rows": [2, 2], "columns": [3, 3]},
            "formula": {"section_count": [1, 1]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    merge_order = data_generator.pop_merge_order()
    signature_tokens = set(Generator._structure_signature(markdown).split("|"))

    assert len(merge_order) == 4
    assert merge_order.count("text") == 2
    assert merge_order.count("table") == 1
    assert merge_order.count("formula") == 1
    assert "table" in signature_tokens
    assert "formula" in signature_tokens


def test_hardcoded_formula_pool_has_100_plus_entries() -> None:
    assert len(HARD_CODED_FORMULA_EXPRESSIONS) >= 100
    assert any("\\int" in formula for formula in HARD_CODED_FORMULA_EXPRESSIONS)


def test_hardcoded_formula_pool_normalizes_relation_aliases() -> None:
    assert all(
        re.search(r"\\ge(?![A-Za-z])", formula) is None
        for formula in HARD_CODED_FORMULA_EXPRESSIONS
    )
    assert all(
        re.search(r"\\le(?![A-Za-z])", formula) is None
        for formula in HARD_CODED_FORMULA_EXPRESSIONS
    )


def test_hardcoded_formula_pool_includes_llm_objectives() -> None:
    required_tokens = (
        r"\mathcal{L}_{\operatorname{SFT}}",
        r"\mathcal{L}_{\operatorname{DPO}}",
        r"\mathcal{L}_{\operatorname{PPO}}",
        r"\mathcal{L}_{\operatorname{KD}}",
        r"\mathcal{L}_{\operatorname{InfoNCE}}",
    )
    assert all(
        any(token in formula for formula in HARD_CODED_FORMULA_EXPRESSIONS)
        for token in required_tokens
    )


def test_normalize_formula_text_converts_ge_le_aliases() -> None:
    data_generator = MarkdownDataGenerator(lang="en")
    assert data_generator._normalize_formula_text(
        r"\operatorname{CRLB}(\hat{\theta})\ge \frac{1}{nI(\theta)}"
    ) == r"\operatorname{CRLB}(\hat{\theta})\geq \frac{1}{nI(\theta)}"
    assert data_generator._normalize_formula_text(
        r"|x+y| \le |x| + |y|"
    ) == r"|x+y| \leq |x| + |y|"


def test_grammar_formula_generator_emits_math_like_expression() -> None:
    random.seed(41)
    data_generator = MarkdownDataGenerator(lang="en")

    samples = [data_generator._build_grammar_formula_expression() for _ in range(20)]

    assert all(sample.strip() for sample in samples)
    assert any(
        token in sample
        for sample in samples
        for token in ("\\frac", "\\sum", "\\int", "\\lim", "\\log", "\\sin", "\\cos", "=")
    )


def test_synthetic_formula_generation_uses_hybrid_branches(monkeypatch) -> None:
    data_generator = MarkdownDataGenerator(lang="en")

    monkeypatch.setattr(data_generator, "_build_grammar_formula_expression", lambda: "GRAMMAR_EXPR")
    monkeypatch.setattr(data_generator, "_build_parametric_synthetic_formula_expression", lambda: "PARAM_EXPR")
    monkeypatch.setattr(data_generator, "_build_hard_coded_formula_expression", lambda: "HARD_EXPR")

    monkeypatch.setattr(random, "choices", lambda *args, **kwargs: ["grammar"])
    assert data_generator._build_synthetic_formula_expression() == "GRAMMAR_EXPR"

    monkeypatch.setattr(random, "choices", lambda *args, **kwargs: ["parametric"])
    assert data_generator._build_synthetic_formula_expression() == "PARAM_EXPR"

    monkeypatch.setattr(random, "choices", lambda *args, **kwargs: ["hardcoded"])
    assert data_generator._build_synthetic_formula_expression() == "HARD_EXPR"


def test_sections_generation_emits_expected_block_types() -> None:
    random.seed(7)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="sections_required_test",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "text": {"section_count": [1, 1]},
            "table": {"section_count": [1, 1], "rows": [2, 2], "columns": [3, 3]},
            "formula": {"section_count": [1, 1]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    merge_order = data_generator.pop_merge_order()
    signature_tokens = set(Generator._structure_signature(markdown).split("|"))

    assert merge_order.count("text") == 1
    assert merge_order.count("table") == 1
    assert merge_order.count("formula") == 1
    assert "table" in signature_tokens
    assert "formula" in signature_tokens


def test_sections_mode_ignores_blueprint_only_controls() -> None:
    random.seed(19)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="sections_ignore_blueprint_controls",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "text": {"section_count": [1, 1]},
            "table": {"section_count": [0, 0]},
            "formula": {"section_count": [1, 1]},
            "allowed_blocks": ["contents", "image", "checklist"],
            "required_blocks": ["contents", "image", "checklist"],
            "frontmatter_probability": 1.0,
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)

    assert "$$" in markdown
    assert "## Contents" not in markdown
    assert "![" not in markdown


def test_formula_dataset_mode_uses_dataset_entries(tmp_path) -> None:
    formula_path = tmp_path / "formulas.txt"
    formula_path.write_text(
        "\n".join(
            [
                "x^2 + y^2 = z^2",
                "\\frac{a+b}{c}",
            ]
        ),
        encoding="utf-8",
    )

    random.seed(21)
    data_generator = MarkdownDataGenerator(lang="en")
    data_generator.configure_content_sources(
        formula_source_mode="dataset",
        formula_dataset_path=str(formula_path),
    )
    spec = TemplateSpec(
        template_id="formula_dataset_test",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "text": {"section_count": [0, 0]},
            "table": {"section_count": [0, 0]},
            "formula": {"section_count": [1, 1]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    formula_lines = [
        parse_markdown_formula_line(line)
        for line in markdown.splitlines()
        if parse_markdown_formula_line(line)
    ]

    assert formula_lines
    assert formula_lines[0] in {"x^2 + y^2 = z^2", "\\frac{a+b}{c}"}


def test_sections_mode_does_not_emit_image_markdown() -> None:
    random.seed(33)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="image_dataset_test",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "text": {"section_count": [1, 1]},
            "table": {"section_count": [0, 0]},
            "formula": {"section_count": [0, 0]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)

    assert "![" not in markdown


def test_mixed_source_mode_falls_back_without_dataset_entries() -> None:
    random.seed(55)
    data_generator = MarkdownDataGenerator(lang="en")
    data_generator.configure_content_sources(
        formula_source_mode="mixed",
        formula_dataset_path="/tmp/not-exist-formulas.txt",
    )
    spec = TemplateSpec(
        template_id="mixed_fallback_test",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "text": {"section_count": [0, 0]},
            "table": {"section_count": [0, 0]},
            "formula": {"section_count": [2, 2]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)

    assert "$$" in markdown
    assert "![" not in markdown


def test_sample_seed_derivation_is_deterministic() -> None:
    generator = Generator.__new__(Generator)
    generator.base_seed = 100

    assert generator._derive_sample_seed(2, 0) == 2118
    assert generator._derive_sample_seed(2, 1) == 11294


def test_text_section_typos_apply_only_to_text_generator_sections() -> None:
    generator = Generator.__new__(Generator)

    def fake_mutate(section_text: str, _ratio: float):
        return section_text.replace("Alpha", "A1pha"), 1

    generator._mutate_similar_text = fake_mutate

    markdown = (
        "# Report\n\n"
        "## Text Block\n"
        "Alpha paragraph content.\n\n"
        "## Table Block\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n\n"
        "## Formula Block\n"
        "$$ x^2 + y^2 = z^2 $$"
    )

    mutated, mutation_count = generator._mutate_text_generator_sections(
        markdown,
        0.2,
        ["text", "table", "formula"],
    )

    assert "A1pha paragraph content." in mutated
    assert "| A | B |" in mutated
    assert "$$ x^2 + y^2 = z^2 $$" in mutated
    assert mutation_count == 1


def test_text_section_typos_skip_when_merge_order_is_empty() -> None:
    generator = Generator.__new__(Generator)
    called = {"value": False}

    def fake_mutate(section_text: str, _ratio: float):
        called["value"] = True
        return section_text, 1

    generator._mutate_similar_text = fake_mutate

    markdown = "# Title\n\nPlain markdown content."
    mutated, mutation_count = generator._mutate_text_generator_sections(markdown, 0.2, [])

    assert mutated == markdown
    assert mutation_count == 0
    assert called["value"] is False


def test_html_renderer_component_preprocessing_handles_image_and_formula() -> None:
    markdown = "![Chart](placeholder://chart-101)\n$$ E = mc^2 $$"
    renderer = HtmlMarkdownRenderer(font_path="/tmp/does-not-need-to-exist.ttf")
    prepared = renderer._prepare_component_markdown(markdown, image_assets={})

    assert "md-image-placeholder" in prepared
    assert "md-formula" in prepared
    assert "placeholder://" not in prepared


def test_html_renderer_formula_css_is_center_aligned() -> None:
    renderer = HtmlMarkdownRenderer(font_path="/tmp/does-not-need-to-exist.ttf")
    html_doc = renderer._build_html_document("$$ E = mc^2 $$", image_assets={})

    assert "text-align: center;" in html_doc
    assert "align-items: center;" in html_doc
    assert "margin: 0 auto;" in html_doc


def test_html_renderer_css_prevents_horizontal_overflow_clipping() -> None:
    renderer = HtmlMarkdownRenderer(font_path="/tmp/does-not-need-to-exist.ttf")
    html_doc = renderer._build_html_document("![Wide](placeholder://wide)", image_assets={})

    assert "box-sizing: border-box;" in html_doc
    assert ".markdown-body .md-image-rendered" in html_doc
    assert "width: auto;" in html_doc
    assert "max-width: 100%;" in html_doc
    assert "object-fit: contain;" in html_doc


def test_html_renderer_component_preprocessing_embeds_real_image_when_asset_exists() -> None:
    markdown = "![Chart](placeholder://chart-101)"
    renderer = HtmlMarkdownRenderer(font_path="/tmp/does-not-need-to-exist.ttf")
    fake_asset = Image.new("RGB", (120, 80), color=(20, 140, 220))

    prepared = renderer._prepare_component_markdown(
        markdown,
        image_assets={"placeholder://chart-101": fake_asset},
    )

    assert "md-image-rendered" in prepared
    assert "data:image/png;base64," in prepared


def test_sections_generation_respects_configured_section_counts() -> None:
    random.seed(31)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="section_count_test",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "text": {"section_count": [3, 3]},
            "table": {"section_count": [2, 2], "rows": [2, 2], "columns": [3, 3]},
            "formula": {"section_count": [1, 1]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    merge_order = data_generator.pop_merge_order()
    assert markdown
    assert len(merge_order) == 6
    assert merge_order.count("text") == 3
    assert merge_order.count("table") == 2
    assert merge_order.count("formula") == 1


def test_fit_image_to_a4_keeps_original_size_without_clipping() -> None:
    generator = Generator.__new__(Generator)
    generator.max_render_width = A4_MAX_WIDTH_PX
    generator.max_render_height = A4_MAX_HEIGHT_PX
    generator.max_render_aspect_ratio = 2.0

    large = Image.new("RGB", (3200, 4200), color=(255, 255, 255))
    fitted, clipped = generator._fit_image_to_a4(large)
    assert clipped is False
    assert fitted.size == large.size

    tall = Image.new("RGB", (656, 3508), color=(255, 255, 255))
    fitted_tall, clipped_tall = generator._fit_image_to_a4(tall)
    assert clipped_tall is False
    assert fitted_tall.size == tall.size

    small = Image.new("RGB", (900, 1200), color=(255, 255, 255))
    fitted_small, clipped_small = generator._fit_image_to_a4(small)
    assert clipped_small is False
    assert fitted_small.size == small.size


def test_fit_image_to_a4_does_not_enforce_aspect_ratio() -> None:
    generator = Generator.__new__(Generator)
    generator.max_render_width = A4_MAX_WIDTH_PX
    generator.max_render_height = A4_MAX_HEIGHT_PX
    generator.max_render_aspect_ratio = 2.0

    wide = Image.new("RGB", (1200, 300), color=(250, 250, 250))
    fitted, clipped = generator._fit_image_to_a4(wide)

    assert clipped is False
    assert fitted.width == 1200
    assert fitted.height == 300
