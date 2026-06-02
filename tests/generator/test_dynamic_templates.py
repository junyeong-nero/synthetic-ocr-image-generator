import random
import sys
import importlib
import re
from collections import Counter, deque
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

Generator = generator_module.Generator
HARD_CODED_FORMULA_EXPRESSIONS = generator_module.HARD_CODED_FORMULA_EXPRESSIONS
HtmlMarkdownRenderer = generator_module.HtmlMarkdownRenderer
MarkdownDataGenerator = generator_module.MarkdownDataGenerator
PlaywrightMarkdownRenderer = generator_module.PlaywrightMarkdownRenderer
TextGenerator = generator_module.TextGenerator
TemplateCatalog = generator_module.TemplateCatalog
TemplateSpec = generator_module.TemplateSpec
parse_coverage_targets = generator_module.parse_coverage_targets
parse_markdown_formula_line = generator_module.parse_markdown_formula_line
normalize_chained_scripts = generator_module._normalize_chained_scripts


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


def test_rich_blueprint_generation_records_block_metadata() -> None:
    random.seed(101)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="rich_blocks_test",
        family="technical",
        complexity=3,
        mode="sections",
        blueprint={
            "document_shape": "technical_manual",
            "section_count": [4, 4],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["paragraph", "code", "table", "checklist"],
            "required_blocks": ["code", "table"],
            "table": {"rows": [2, 2], "columns": [3, 3]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    merge_order = data_generator.pop_merge_order()
    composition = data_generator.pop_composition_metadata()

    assert "```" in markdown
    assert "| " in markdown
    assert len(merge_order) == 4
    assert "code" in merge_order
    assert "table" in merge_order
    assert composition["document_shape"] == "technical_manual"
    assert composition["block_type_counts"]["code"] == 1
    assert composition["block_type_counts"]["table"] == 1


def test_legacy_sections_generation_keeps_existing_merge_order() -> None:
    random.seed(102)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="legacy_sections_still_work",
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
    composition = data_generator.pop_composition_metadata()

    assert "$$" in markdown
    assert "| " in markdown
    assert sorted(merge_order) == ["formula", "table", "text"]
    assert composition["document_shape"] == "sections"
    assert composition["block_type_counts"] == {"text": 1, "table": 1, "formula": 1}


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


def test_normalize_chained_scripts_rewrites_double_superscript_expression() -> None:
    expression = r"\lim_{n\to 1} \alpha^{2m}^{n}"
    assert normalize_chained_scripts(expression) == r"\lim_{n\to 1} \alpha^{2m^{n}}"


def test_normalize_chained_scripts_rewrites_double_subscript_expression() -> None:
    expression = r"x_{a}_{b}"
    assert normalize_chained_scripts(expression) == r"x_{a_{b}}"


def test_normalize_chained_scripts_rewrites_mixed_script_duplicates() -> None:
    expression_sup = r"x^{a}_{b}^{c}"
    expression_sub = r"x_{a}^{b}_{c}"
    assert normalize_chained_scripts(expression_sup) == r"x^{a^{c}}_{b}"
    assert normalize_chained_scripts(expression_sub) == r"x_{a_{c}}^{b}"


def test_normalize_chained_scripts_handles_simple_script_tokens() -> None:
    expression = r"x^a_b^c"
    assert normalize_chained_scripts(expression) == r"x^{a^{c}}_{b}"


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


def test_sections_mode_honors_rich_blueprint_controls() -> None:
    random.seed(19)
    data_generator = MarkdownDataGenerator(lang="en")
    spec = TemplateSpec(
        template_id="sections_honor_blueprint_controls",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "document_shape": "mixed_controls",
            "section_count": [3, 3],
            "blocks_per_section": [1, 1],
            "allowed_blocks": ["paragraph", "image", "checklist"],
            "required_blocks": ["image", "checklist"],
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)
    composition = data_generator.pop_composition_metadata()

    assert "![" in markdown
    assert "- [ ]" in markdown or "- [x]" in markdown
    assert composition["document_shape"] == "mixed_controls"
    assert "image" in composition["block_types"]
    assert "checklist" in composition["block_types"]


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
    assert "min-width:" in html_doc
    assert "overflow: visible;" in html_doc
    assert ".markdown-body .md-image-rendered" in html_doc
    assert "width: auto;" in html_doc
    assert "max-width: 100%;" in html_doc
    assert "object-fit: contain;" in html_doc


def test_html_renderer_table_css_uses_roomier_document_defaults() -> None:
    renderer = HtmlMarkdownRenderer(font_path="/tmp/does-not-need-to-exist.ttf")
    html_doc = renderer._build_html_document("| A | B |\n| --- | --- |\n| 1 | 2 |", image_assets={})

    assert "@page {" in html_doc
    assert "print-color-adjust: exact;" in html_doc
    assert "table-layout: auto;" in html_doc
    assert "padding: 8px 12px;" in html_doc
    assert "vertical-align: top;" in html_doc
    assert "tbody tr:nth-child(even) td" in html_doc


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


def test_html_renderer_converts_markdown_hard_breaks_to_br_tags() -> None:
    html = HtmlMarkdownRenderer._coerce_markdown_html("Alpha  \nBeta")

    assert "Alpha<br" in html
    assert "Beta" in html


def test_playwright_renderer_uses_headless_chromium(monkeypatch) -> None:
    renderers_module = importlib.import_module("generator.markdown_renderers")
    real_import_module = renderers_module.importlib.import_module
    calls = {}

    class _FakeLocator:
        def screenshot(self, path: str, animations: str) -> None:
            calls["screenshot"] = {"path": path, "animations": animations}
            Image.new("RGB", (64, 96), color=(255, 255, 255)).save(path)

    class _FakePage:
        def goto(self, url: str, wait_until: str) -> None:
            calls["goto"] = {"url": url, "wait_until": wait_until}

        def wait_for_function(self, script: str) -> None:
            calls["wait_for_function"] = script

        def evaluate(self, script: str):
            calls["evaluate"] = script
            return True

        def locator(self, selector: str) -> _FakeLocator:
            calls["selector"] = selector
            return _FakeLocator()

    class _FakeBrowser:
        def new_page(self, viewport, device_scale_factor: int) -> _FakePage:
            calls["new_page"] = {
                "viewport": viewport,
                "device_scale_factor": device_scale_factor,
            }
            return _FakePage()

        def close(self) -> None:
            calls["browser_closed"] = True

    class _FakeChromium:
        def launch(self, headless: bool, args):
            calls["launch"] = {"headless": headless, "args": args}
            return _FakeBrowser()

    class _FakePlaywrightContext:
        def __enter__(self):
            return type("_FakePlaywright", (), {"chromium": _FakeChromium()})()

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_import_module(name: str):
        if name == "playwright.sync_api":
            return type(
                "_FakePlaywrightModule",
                (),
                {"sync_playwright": staticmethod(lambda: _FakePlaywrightContext())},
            )()
        return real_import_module(name)

    monkeypatch.setattr(renderers_module.importlib, "import_module", _fake_import_module)

    renderer = PlaywrightMarkdownRenderer(font_path="/tmp/does-not-need-to-exist.ttf")
    image = renderer.render("# Title\n\nBody text")

    assert image.size == (64, 96)
    assert calls["launch"]["headless"] is True
    assert "--hide-scrollbars" in calls["launch"]["args"]
    assert calls["selector"] == ".capture-shell"
    assert calls["new_page"]["viewport"]["width"] == (
        renderer.style.margin_left
        + renderer.style.content_width
        + renderer.style.margin_right
        + (renderer._CAPTURE_PADDING_PX * 2)
    )
    assert calls["screenshot"]["animations"] == "disabled"
    assert calls["browser_closed"] is True


def test_playwright_renderer_reports_missing_dependency(monkeypatch) -> None:
    renderers_module = importlib.import_module("generator.markdown_renderers")
    real_import_module = renderers_module.importlib.import_module

    def _fake_import_module(name: str):
        if name == "playwright.sync_api":
            raise ImportError("missing playwright")
        return real_import_module(name)

    monkeypatch.setattr(renderers_module.importlib, "import_module", _fake_import_module)

    renderer = PlaywrightMarkdownRenderer(font_path="/tmp/does-not-need-to-exist.ttf")

    try:
        renderer.render("Simple body")
    except RuntimeError as exc:
        assert "playwright package is required" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when Playwright is unavailable")


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


def test_sections_generation_respects_configured_text_wrap_width(tmp_path) -> None:
    data_generator = MarkdownDataGenerator(lang="en")
    corpus_lang_dir = tmp_path / "en"
    corpus_lang_dir.mkdir(parents=True, exist_ok=True)
    corpus_lang_dir.joinpath("paragraphs.txt").write_text(
        "Wrapping width should come from the template blueprint so generated text sections break earlier than the default width allows.\n",
        encoding="utf-8",
    )
    data_generator.data = importlib.import_module("generator.data_provider").DataProvider(
        lang="en",
        mix_ratio=0.0,
        corpus_dir=tmp_path,
        use_corpus=True,
    )
    spec = TemplateSpec(
        template_id="text_wrap_width_test",
        family="sections",
        complexity=2,
        mode="sections",
        blueprint={
            "text": {"section_count": [1, 1], "max_line_chars": 24},
            "table": {"section_count": [0, 0]},
            "formula": {"section_count": [0, 0]},
        },
    )

    markdown = data_generator.generate_markdown(template_id=spec.template_id, template_spec=spec)

    assert "  \n" in markdown
    assert "Wrapping width should" in markdown
    assert "from the template" in markdown


def test_text_generator_uses_language_aware_sentence_content() -> None:
    random.seed(73)
    data_provider_module = importlib.import_module("generator.data_provider")
    data = data_provider_module.DataProvider(lang="en", mix_ratio=0.0, use_corpus=False)
    text_generator = TextGenerator(
        data=data,
        clip_text=lambda text, max_len: text if len(text) <= max_len else text[:max_len],
        max_paragraph_chars=220,
    )

    sections = text_generator.generate_sections(section_count=6)
    markdown = "\n\n".join(sections)

    assert "- Person:" not in markdown
    assert "- Install:" not in markdown
    assert any(
        sentence in markdown
        for sentence in [
            "This project is designed to enhance user productivity.",
            "It provides various features with an extensible architecture.",
            "Easy to install and well-documented for quick onboarding.",
            "Continuously improved with community support.",
        ]
    )


def test_text_generator_prefers_paragraph_corpus_sentences(tmp_path) -> None:
    data_provider_module = importlib.import_module("generator.data_provider")
    corpus_lang_dir = tmp_path / "en"
    corpus_lang_dir.mkdir(parents=True, exist_ok=True)
    (corpus_lang_dir / "paragraphs.txt").write_text(
        "Corpus sentences should drive generated sections. Follow-up corpus sentence here.\n",
        encoding="utf-8",
    )

    data = data_provider_module.DataProvider(lang="en", mix_ratio=0.0, corpus_dir=tmp_path, use_corpus=True)
    text_generator = TextGenerator(
        data=data,
        clip_text=lambda text, max_len: text if len(text) <= max_len else text[:max_len],
        max_paragraph_chars=220,
    )

    sections = text_generator.generate_sections(section_count=1)
    markdown = "\n\n".join(sections)

    assert "Corpus sentences should drive generated sections." in markdown or "Follow-up corpus sentence here." in markdown


def test_text_generator_wraps_long_corpus_paragraphs_into_markdown_line_breaks(tmp_path) -> None:
    data_provider_module = importlib.import_module("generator.data_provider")
    corpus_lang_dir = tmp_path / "en"
    corpus_lang_dir.mkdir(parents=True, exist_ok=True)
    long_paragraph = (
        "This corpus paragraph is intentionally long so the text generator inserts markdown line breaks "
        "before it turns into one oversized text section line for OCR rendering fidelity."
    )
    (corpus_lang_dir / "paragraphs.txt").write_text(f"{long_paragraph}\n", encoding="utf-8")

    data = data_provider_module.DataProvider(lang="en", mix_ratio=0.0, corpus_dir=tmp_path, use_corpus=True)
    text_generator = TextGenerator(
        data=data,
        clip_text=lambda text, max_len: text if len(text) <= max_len else text[:max_len],
        max_paragraph_chars=220,
        max_line_chars=48,
    )

    sections = text_generator.generate_sections(section_count=1)
    markdown = "\n\n".join(sections)

    assert "  \n" in markdown
    assert "This corpus paragraph is intentionally long" in markdown
    assert "before it turns into one oversized text section" in markdown
    assert "line for OCR rendering fidelity." in markdown


def test_table_generator_prefers_paragraph_corpus_headers_and_cells(tmp_path) -> None:
    data_provider_module = importlib.import_module("generator.data_provider")
    table_module = importlib.import_module("generator.table_generator")

    corpus_lang_dir = tmp_path / "en"
    corpus_lang_dir.mkdir(parents=True, exist_ok=True)
    (corpus_lang_dir / "paragraphs.txt").write_text(
        "Quarterly revenue insights improve planning. Customer retention metrics guide roadmap.\n",
        encoding="utf-8",
    )

    data = data_provider_module.DataProvider(lang="en", mix_ratio=0.0, corpus_dir=tmp_path, use_corpus=True)
    table_generator = table_module.TableGenerator(
        data=data,
        clip_text=lambda text, max_len: text if len(text) <= max_len else text[:max_len],
    )

    sections = table_generator.generate_sections(section_count=1, row_range=(2, 2), column_range=(3, 3))
    markdown = "\n\n".join(sections)

    assert "| Product | Category | Price |" not in markdown
    assert "| Column 1 |" not in markdown
    assert "Quarterly revenue" in markdown or "Customer retention" in markdown


def test_template_catalog_builtin_default_limits_table_columns() -> None:
    catalog = TemplateCatalog(config_dir="/tmp/does-not-exist")

    resolved = catalog.resolve("default", None, None, None)

    assert resolved
    assert resolved[0].blueprint["table"]["columns"] == [3, 4]


def test_balanced_style_sampler_preserves_roomier_minimum_widths(monkeypatch) -> None:
    style_sampler_module = importlib.import_module("generator.style_sampler")

    monkeypatch.setattr(style_sampler_module.random, "choice", lambda values: values[0])
    monkeypatch.setattr(style_sampler_module.random, "randint", lambda low, _high: low)
    monkeypatch.setattr(style_sampler_module.random, "uniform", lambda low, _high: low)

    style = style_sampler_module.random_style("balanced")

    assert style.margin_left >= 28
    assert style.margin_right >= 28
    assert style.content_width >= 500


def test_generate_single_metadata_does_not_include_a4_clipping_flags(monkeypatch) -> None:
    generator = Generator.__new__(Generator)
    generator.template_specs = [
        TemplateSpec(
            template_id="test-template",
            family="text",
            mode="generated",
            complexity=1,
            source="test",
            weight=1.0,
            version="1",
            blueprint={},
        )
    ]
    generator.template_catalog = None
    generator.template_counts = Counter()
    generator.family_counts = Counter()
    generator.novelty_window = 8
    generator.novelty_threshold = 1.0
    generator.novelty_max_attempts = 1
    generator._recent_signatures = deque(maxlen=generator.novelty_window)
    generator.base_seed = None
    generator.noise_ratio = 0.0
    generator.blur_ratio = 0.0
    generator.style_profile = "balanced"
    generator.markdown_renderer = "pil"
    generator.similar_char_ratio = 0.0
    generator._seed_for_sample = lambda _seed: None
    generator._derive_sample_seed = lambda _sample_index, _attempt: None
    generator._select_template_spec = lambda: (generator.template_specs[0], 1.0)
    generator._mutate_text_generator_sections = lambda markdown, _ratio, merge_order: (markdown, 0)

    class _StubDataGenerator:
        @staticmethod
        def generate_markdown(template_id: str, template_spec: TemplateSpec) -> str:
            assert template_id == "test-template"
            assert template_spec.template_id == "test-template"
            return "# Heading\n\nBody"

        @staticmethod
        def pop_merge_order() -> list[str]:
            return ["text"]

    generator.data_generator = _StubDataGenerator()

    monkeypatch.setattr(generator_module, "random_style", lambda _profile: generator_module.MarkdownStyle())
    monkeypatch.setattr(generator_module, "markdown_to_json_ast", lambda markdown_text: [{"raw": markdown_text}])

    class _StubRenderer:
        def __init__(self, _font_path, style):
            self.style = style

        def render(self, markdown_text: str):
            assert markdown_text == "# Heading\n\nBody"
            return Image.new("RGB", (1234, 2345), color=(255, 255, 255))

    monkeypatch.setattr(generator_module, "MarkdownRenderer", _StubRenderer)
    generator.font_paths = ["/tmp/dummy-font.ttf"]

    _image, metadata = generator.generate_single(sample_index=7)

    assert metadata["image_width"] == 1234
    assert metadata["image_height"] == 2345
    assert "a4_scaled" not in metadata
    assert "a4_clipped" not in metadata
