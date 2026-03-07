import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_main_module():
    return importlib.import_module("main")


def _load_pipeline_module():
    stub_character_similarity = ModuleType("character_similarity")

    def _stub_find_similar_chars(_char: str, _db, top_n: int = 5):
        return []

    setattr(stub_character_similarity, "find_similar_chars", _stub_find_similar_chars)
    sys.modules.setdefault("character_similarity", stub_character_similarity)

    stub_faker = ModuleType("faker")

    class _DummyFaker:
        def __init__(self, _locale: str = "en_US"):
            self.locale = _locale

        @staticmethod
        def seed(_seed: int) -> None:
            return None

        def seed_instance(self, _seed: int) -> None:
            return None

    setattr(stub_faker, "Faker", _DummyFaker)
    sys.modules.setdefault("faker", stub_faker)

    stub_faker_config = ModuleType("faker.config")
    setattr(stub_faker_config, "AVAILABLE_LOCALES", ["en_US", "ko_KR"])
    sys.modules.setdefault("faker.config", stub_faker_config)

    return importlib.import_module("pipeline")


class DummyModelSpecificConfig:
    backend = "openai"
    tensor_parallel_size = 2
    api_base = "https://example.com"
    timeout = 60
    max_retries = 5
    device = "cuda"
    dtype = "bfloat16"
    rate_limit_rpm = 120

    @staticmethod
    def get_temperature() -> float:
        return 0.2

    @staticmethod
    def get_max_tokens() -> int:
        return 1024

    @staticmethod
    def get_batch_size() -> int:
        return 4


def test_resolve_execution_mode_priority() -> None:
    main_module = _load_main_module()

    args = argparse.Namespace(inference_only=True, evaluate_only=True)
    assert main_module._resolve_execution_mode(args).value == "inference_only"

    args = argparse.Namespace(inference_only=False, evaluate_only=True)
    assert main_module._resolve_execution_mode(args).value == "evaluate_only"

    args = argparse.Namespace(inference_only=False, evaluate_only=False)
    assert main_module._resolve_execution_mode(args).value == "all"


def test_resolve_evaluation_runtime_applies_cli_overrides() -> None:
    main_module = _load_main_module()

    args = argparse.Namespace(
        backend=None,
        temperature=0.9,
        max_tokens=2048,
        batch_size=8,
        tensor_parallel=3,
        api_base="https://override.example.com",
    )

    runtime = main_module._resolve_evaluation_runtime(args, DummyModelSpecificConfig())

    assert runtime["backend"] == "openai"
    assert runtime["temperature"] == 0.9
    assert runtime["max_tokens"] == 2048
    assert runtime["batch_size"] == 8
    assert runtime["tensor_parallel_size"] == 3
    assert runtime["api_base"] == "https://override.example.com"


def test_build_generation_kwargs_includes_optional_toggles_only_when_set() -> None:
    pipeline_module = _load_pipeline_module()
    kwargs = pipeline_module._build_generation_kwargs(
        template="readme",
        template_family=None,
        min_template_complexity=None,
        max_template_complexity=None,
        template_config_dir=None,
        markdown_renderer="pil",
        style_profile="balanced",
        coverage_targets=None,
        novelty_window=80,
        novelty_threshold=0.95,
        novelty_max_attempts=4,
        similar_char_ratio=0.08,
        similarity_db_path=None,
        formula_source_mode="mixed",
        formula_dataset_path=None,
        formula_dataset_weight=0.45,
        formula_random_weight=0.30,
        formula_synthetic_weight=0.25,
        seed=7,
        add_noise=None,
        add_blur=True,
        sample_start_index=15,
    )

    assert "add_noise" not in kwargs
    assert kwargs["add_blur"] is True
    assert kwargs["seed"] == 7
    assert kwargs["sample_start_index"] == 15


def test_build_generate_pipeline_args_maps_expected_keys() -> None:
    main_module = _load_main_module()

    args = argparse.Namespace(
        repo_id="repo",
        output_dir="./data",
        lang="ko",
        size=10,
        template="readme",
        template_family="sections",
        min_template_complexity=1,
        max_template_complexity=3,
        template_config_dir="configs/generator/templates",
        markdown_renderer="pil",
        style_profile="balanced",
        coverage_target=["sections=0.5"],
        novelty_window=80,
        novelty_threshold=0.95,
        novelty_max_attempts=4,
        similar_char_ratio=0.08,
        similarity_db_path=None,
        formula_source_mode="mixed",
        formula_dataset_path=None,
        formula_dataset_weight=0.45,
        formula_random_weight=0.30,
        formula_synthetic_weight=0.25,
        add_noise=True,
        add_blur=False,
        mixed=False,
        train_ratio=0.9,
        test_ratio=0.1,
        seed=123,
        shard_size=250,
        max_shards=3,
        resume=True,
    )

    pipeline_args = main_module._build_generate_pipeline_args(args)

    assert pipeline_args["repo_id"] == "repo"
    assert pipeline_args["coverage_targets"] == ["sections=0.5"]
    assert pipeline_args["seed"] == 123
    assert pipeline_args["shard_size"] == 250
    assert pipeline_args["max_shards"] == 3
    assert pipeline_args["resume"] is True
    assert len(pipeline_args) == len(main_module.GENERATE_ARG_TO_PIPELINE_KEY)


def test_add_optional_generation_effect_argument_preserves_bool_optional_behavior() -> None:
    main_module = _load_main_module()

    parser = argparse.ArgumentParser()
    main_module._add_optional_generation_effect_argument(
        parser,
        "--add-noise",
        "Enable or disable noise effect",
    )

    default_args = parser.parse_args([])
    enabled_args = parser.parse_args(["--add-noise"])
    disabled_args = parser.parse_args(["--no-add-noise"])

    assert default_args.add_noise is None
    assert enabled_args.add_noise is True
    assert disabled_args.add_noise is False


def test_configure_generate_parser_wires_defaults_and_effect_flags() -> None:
    main_module = _load_main_module()

    parser = argparse.ArgumentParser()
    main_module._configure_generate_parser(parser)

    parsed = parser.parse_args([
        "--repo-id",
        "demo/repo",
        "--shard-size",
        "200",
        "--max-shards",
        "2",
        "--resume",
        "--no-add-noise",
        "--add-blur",
    ])

    assert parsed.repo_id == "demo/repo"
    assert parsed.markdown_renderer == "pil"
    assert parsed.style_profile == "balanced"
    assert parsed.shard_size == 200
    assert parsed.max_shards == 2
    assert parsed.resume is True
    assert parsed.add_noise is False
    assert parsed.add_blur is True
