import sys
from types import ModuleType
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
GENERATOR_DIR = SRC / "generator"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

generator_stub = ModuleType("generator")
generator_stub.__path__ = [str(GENERATOR_DIR)]
sys.modules.setdefault("generator", generator_stub)

faker_stub = ModuleType("faker")


class _DummyFaker:
    def __init__(self, _locale: str = "en_US"):
        self.locale = _locale

    @staticmethod
    def seed(_seed: int) -> None:
        return None

    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: "stub"


setattr(faker_stub, "Faker", _DummyFaker)
sys.modules.setdefault("faker", faker_stub)

faker_config_stub = ModuleType("faker.config")
setattr(
    faker_config_stub,
    "AVAILABLE_LOCALES",
    ["en_US", "ko_KR", "ja_JP", "hi_IN", "zh_CN"],
)
sys.modules.setdefault("faker.config", faker_config_stub)

def _load_provider_symbols():
    from generator.data_provider import DataProvider
    from generator.faker_locales import base_lang_code, normalize_lang_code, resolve_faker_locale

    return DataProvider, base_lang_code, normalize_lang_code, resolve_faker_locale


def test_faker_locale_helpers_normalize_and_resolve() -> None:
    _, base_lang_code, normalize_lang_code, resolve_faker_locale = _load_provider_symbols()

    assert normalize_lang_code("EN_us") == "en-us"
    assert base_lang_code("zh-CN") == "zh"
    assert resolve_faker_locale("ko") == "ko_KR"


def test_data_provider_falls_back_to_english_hardcoded_data() -> None:
    DataProvider, _, _, _ = _load_provider_symbols()

    provider = DataProvider(lang="xx-YY", use_corpus=False)

    assert provider.currency == "$"
    assert provider.base_lang == "xx"
    assert provider.normalized_lang == "xx-yy"


def test_data_provider_loads_corpus_with_dedup_and_rotation(tmp_path: Path) -> None:
    DataProvider, _, _, _ = _load_provider_symbols()

    corpus_lang_dir = tmp_path / "en"
    corpus_lang_dir.mkdir(parents=True, exist_ok=True)
    (corpus_lang_dir / "product_names.txt").write_text(
        "alpha\nalpha\nbeta\n",
        encoding="utf-8",
    )

    provider = DataProvider(lang="en", seed=7, corpus_dir=tmp_path, use_corpus=True)

    assert provider.has_corpus("product_names") is True
    assert provider.corpus_size("product_names") == 2

    first = provider.product_name()
    second = provider.product_name()
    third = provider.product_name()

    assert {first, second} == {"alpha", "beta"}
    assert third in {"alpha", "beta"}


def test_data_provider_builds_sentences_from_paragraph_corpus(tmp_path: Path) -> None:
    DataProvider, _, _, _ = _load_provider_symbols()

    corpus_lang_dir = tmp_path / "ko"
    corpus_lang_dir.mkdir(parents=True, exist_ok=True)
    (corpus_lang_dir / "paragraphs.txt").write_text(
        "첫 번째 문장입니다. 두 번째 문장입니다.\n세 번째 문장입니다! 네 번째 문장입니다?\n",
        encoding="utf-8",
    )

    provider = DataProvider(lang="ko", seed=11, corpus_dir=tmp_path, use_corpus=True)

    first = provider.sentence()
    second = provider.sentence()
    third = provider.sentence()

    expected = {
        "첫 번째 문장입니다.",
        "두 번째 문장입니다.",
        "세 번째 문장입니다!",
        "네 번째 문장입니다?",
    }
    assert first in expected
    assert second in expected
    assert third in expected


def test_title_and_feature_fall_back_to_paragraph_corpus_fragments(tmp_path: Path) -> None:
    DataProvider, _, _, _ = _load_provider_symbols()

    corpus_lang_dir = tmp_path / "en"
    corpus_lang_dir.mkdir(parents=True, exist_ok=True)
    (corpus_lang_dir / "paragraphs.txt").write_text(
        "Paragraph driven titles should stay language aware. Another supporting sentence.\n",
        encoding="utf-8",
    )

    provider = DataProvider(lang="en", seed=5, corpus_dir=tmp_path, use_corpus=True)

    title = provider.title()
    feature = provider.feature()

    assert title
    assert feature
    assert title != "Getting Started"
    assert feature != "Fast performance"
    assert "language aware" in title.lower() or "paragraph driven" in title.lower()
    assert "language aware" in feature.lower() or "paragraph driven" in feature.lower()
