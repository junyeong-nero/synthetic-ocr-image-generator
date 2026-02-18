import random
import sys
import importlib
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
HtmlMarkdownRenderer = generator_module.HtmlMarkdownRenderer
MarkdownDataGenerator = generator_module.MarkdownDataGenerator
TemplateCatalog = generator_module.TemplateCatalog
TemplateSpec = generator_module.TemplateSpec
parse_coverage_targets = generator_module.parse_coverage_targets


def test_parse_coverage_targets_accepts_list_and_dict() -> None:
    parsed_from_list = parse_coverage_targets(["legacy=0.6", "incident:0.2"])
    assert parsed_from_list == {"legacy": 0.6, "incident": 0.2}

    parsed_from_dict = parse_coverage_targets({"legacy": 0.8, "ops": 2.0})
    assert parsed_from_dict == {"legacy": 0.8, "ops": 1.0}


def test_template_catalog_resolve_supports_alias_and_filters(tmp_path) -> None:
    config_path = tmp_path / "templates.yaml"
    config_path.write_text(
        """
version: 1
templates:
  - id: dynamic_ops_brief
    family: operations
    complexity: 3
    weight: 1.2
    mode: blueprint
    aliases: [ops-brief]
    blueprint:
      section_count: [1, 1]
      paragraphs_per_section: [1, 1]
      blocks_per_section: [1, 1]
      allowed_blocks: [code]
""".strip(),
        encoding="utf-8",
    )

    catalog = TemplateCatalog(config_dir=str(tmp_path))

    alias_resolved = catalog.resolve("ops-brief", None, None, None)
    assert len(alias_resolved) == 1
    assert alias_resolved[0].template_id == "dynamic_ops_brief"

    filtered = catalog.resolve(None, "legacy", 3, None)
    assert filtered
    assert all(spec.family == "legacy" for spec in filtered)
    assert all(spec.complexity >= 3 for spec in filtered)


def test_blueprint_generation_emits_required_block_types() -> None:
    random.seed(7)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="dynamic_test",
        family="procedural",
        complexity=3,
        mode="blueprint",
        blueprint={
            "title_prefix": "Dynamic Test",
            "section_count": [1, 1],
            "paragraphs_per_section": [1, 1],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["table", "code", "bullet_list"],
            "required_blocks": ["table", "code"],
            "table_rows": [2, 2],
            "frontmatter_probability": 0.0,
            "section_rule_probability": 0.0,
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    signature_tokens = set(Generator._structure_signature(markdown).split("|"))

    assert "table" in signature_tokens
    assert "code" in signature_tokens


def test_blueprint_generation_supports_component_blocks_formula_image_contents() -> None:
    random.seed(19)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="dynamic_components",
        family="components",
        complexity=2,
        mode="blueprint",
        blueprint={
            "title_prefix": "Component Canvas",
            "section_count": [2, 2],
            "paragraphs_per_section": [0, 0],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["contents", "bullet_points", "formula", "image", "checklist"],
            "required_blocks": ["contents", "formula", "image", "checklist"],
            "frontmatter_probability": 0.0,
            "section_rule_probability": 0.0,
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    signature_tokens = set(Generator._structure_signature(markdown).split("|"))

    assert "## Contents" in markdown
    assert "$$" in markdown
    assert "![" in markdown
    assert "(placeholder://" in markdown
    assert "formula" in signature_tokens
    assert "image" in signature_tokens


def test_sample_seed_derivation_is_deterministic() -> None:
    generator = Generator.__new__(Generator)
    generator.base_seed = 100

    assert generator._derive_sample_seed(2, 0) == 2118
    assert generator._derive_sample_seed(2, 1) == 11294


def test_html_renderer_component_preprocessing_handles_image_and_formula() -> None:
    markdown = "![Chart](placeholder://chart-101)\n$$ E = mc^2 $$"
    prepared = HtmlMarkdownRenderer._prepare_component_markdown(markdown)

    assert "md-image-placeholder" in prepared
    assert "md-formula" in prepared
    assert "placeholder://" not in prepared


def test_blueprint_generation_respects_max_total_lines() -> None:
    random.seed(31)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="line_budget_test",
        family="components",
        complexity=3,
        mode="blueprint",
        blueprint={
            "title_prefix": "Line Budget",
            "section_count": [4, 4],
            "paragraphs_per_section": [2, 2],
            "blocks_per_section": [3, 3],
            "allowed_blocks": ["subtitle", "bullet_points", "checklist", "table", "formula", "image"],
            "required_blocks": ["contents", "formula", "image", "checklist"],
            "max_total_lines": 45,
            "max_paragraph_chars": 120,
            "frontmatter_probability": 1.0,
            "section_rule_probability": 0.5,
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    assert len(markdown.splitlines()) <= 45


def test_fit_image_to_a4_resizes_large_images() -> None:
    generator = Generator.__new__(Generator)
    generator.max_render_width = A4_MAX_WIDTH_PX
    generator.max_render_height = A4_MAX_HEIGHT_PX

    large = Image.new("RGB", (3200, 4200), color=(255, 255, 255))
    fitted, scaled = generator._fit_image_to_a4(large)
    assert scaled is True
    assert fitted.width <= A4_MAX_WIDTH_PX
    assert fitted.height <= A4_MAX_HEIGHT_PX

    small = Image.new("RGB", (900, 1200), color=(255, 255, 255))
    fitted_small, scaled_small = generator._fit_image_to_a4(small)
    assert scaled_small is False
    assert fitted_small.size == small.size
