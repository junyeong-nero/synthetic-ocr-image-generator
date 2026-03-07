import importlib
import sys
from datetime import datetime
from types import ModuleType


def _load_generator_module():
    stub_character_similarity = ModuleType("character_similarity")
    setattr(stub_character_similarity, "find_similar_chars", lambda _char, _db, top_n=5: [])
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

        def date_time(self) -> datetime:
            return datetime(2024, 1, 1, 9, 0)

        def __getattr__(self, _name: str):
            return lambda *args, **kwargs: "stub"

    setattr(stub_faker, "Faker", _DummyFaker)
    sys.modules.setdefault("faker", stub_faker)
    sys.modules.setdefault("faker.config", stub_faker_config)

    return importlib.import_module("generator.generator")


def test_normalize_chained_scripts_rewrites_chained_markers() -> None:
    module = _load_generator_module()

    assert module._normalize_chained_scripts(r"x_{a}_{b}") == r"x_{a_{b}}"
    assert module._normalize_chained_scripts(r"x^a_b^c") == r"x^{a^{c}}_{b}"


def test_random_style_keeps_content_width_in_bounds() -> None:
    module = _load_generator_module()
    generator = module.Generator.__new__(module.Generator)
    generator.style_profile = "aggressive"

    style = generator._random_style()

    assert 460 <= style.content_width <= 720
    assert style.margin_left >= 20
    assert style.margin_right >= 20


def test_mutate_text_generator_sections_only_mutates_text_sections() -> None:
    module = _load_generator_module()
    generator = module.Generator.__new__(module.Generator)
    generator._mutate_similar_text = lambda section_text, _ratio: (section_text.replace("Alpha", "A1pha"), 1)

    markdown = "# Report\n\n## Text\nAlpha paragraph.\n\n## Table\n| A | B |\n"
    mutated, mutation_count = generator._mutate_text_generator_sections(markdown, 0.2, ["text", "table"])

    assert "A1pha paragraph." in mutated
    assert "| A | B |" in mutated
    assert mutation_count == 1
