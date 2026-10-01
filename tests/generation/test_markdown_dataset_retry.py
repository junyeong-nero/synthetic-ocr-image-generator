import json
import time
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from src.generation.markdown_dataset import MarkdownDatasetGenerator
from src.generation.options import GenerationOptions


class StubGenerator:
    """Stands in for the real Generator; ``failures[i]`` is how many times sample i raises before succeeding."""

    def __init__(self, output_dir: Path, failures: dict[int, int]):
        self.output_dir = output_dir
        self.failures = failures
        self.calls: Counter = Counter()
        output_dir.mkdir(parents=True, exist_ok=True)

    def _configure_generation(self, **kwargs) -> None:
        pass

    def generate_single(self, sample_index: int = 0):
        self.calls[sample_index] += 1
        if self.calls[sample_index] <= self.failures.get(sample_index, 0):
            raise RuntimeError("render failed")
        return Image.new("RGB", (4, 4)), {"GT_markdown": f"# {sample_index}", "sample_index": sample_index}

    def save_image(self, image, filename: str) -> None:
        image.save(self.output_dir / filename)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


def _generator(tmp_path: Path, failures: dict[int, int]) -> tuple[MarkdownDatasetGenerator, StubGenerator]:
    dataset_generator = MarkdownDatasetGenerator(str(tmp_path / "out"), "fonts/ko", "ko")
    stub = StubGenerator(tmp_path / "out" / "markdown", failures)
    dataset_generator._markdown_generator = stub
    return dataset_generator, stub


def _indices(tmp_path: Path) -> list[int]:
    lines = (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["sample_index"] for line in lines]


def test_a_transient_render_failure_is_retried_with_the_same_sample(tmp_path: Path) -> None:
    dataset_generator, stub = _generator(tmp_path, {1: 2})

    result = dataset_generator.run(3, GenerationOptions())

    assert result is not None
    assert _indices(tmp_path) == [0, 1, 2]
    assert stub.calls[1] == 3
    assert dataset_generator.skipped_samples == []


def test_a_persistent_failure_aborts_the_run_by_default(tmp_path: Path) -> None:
    dataset_generator, _ = _generator(tmp_path, {1: 99})

    assert dataset_generator.run(3, GenerationOptions()) is None


def test_persistent_failures_are_skipped_within_the_budget(tmp_path: Path) -> None:
    dataset_generator, _ = _generator(tmp_path, {101: 99})

    result = dataset_generator.run(3, GenerationOptions(), sample_start_index=100, skip_failed_samples=1)

    assert result is not None
    assert _indices(tmp_path) == [100, 102]
    assert dataset_generator.skipped_samples == [101]
    assert not (tmp_path / "out" / "markdown" / "markdown_00101.png").exists()


def test_exceeding_the_skip_budget_aborts_the_run(tmp_path: Path) -> None:
    dataset_generator, _ = _generator(tmp_path, {0: 99, 1: 99})

    assert dataset_generator.run(3, GenerationOptions(), skip_failed_samples=1) is None
    assert dataset_generator.skipped_samples == [0]
