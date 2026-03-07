import importlib
import sys
from datetime import datetime
from types import ModuleType


def _load_generator_module():
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

    return importlib.import_module("generator.generator")


def test_resolve_effect_settings_respects_explicit_toggle() -> None:
    module = _load_generator_module()
    generator = module.Generator.__new__(module.Generator)

    enabled, ratio = generator._resolve_effect_settings(
        enabled_key="add_noise",
        ratio_key="noise_ratio",
        enabled_default=True,
        ratio_default=0.1,
        kwargs={"add_noise": False, "noise_ratio": 0.7},
    )

    assert enabled is False
    assert ratio == 0.0


def test_normalize_choice_falls_back_to_default() -> None:
    module = _load_generator_module()
    normalized = module.Generator._normalize_choice(
        "unexpected-mode",
        {"pil", "html2image", "playwright"},
        "pil",
        "markdown renderer",
    )

    assert normalized == "pil"
