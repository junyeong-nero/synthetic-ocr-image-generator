import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

from PIL import Image


def _install_generator_stubs() -> None:
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

        def __getattr__(self, _name: str):
            return lambda *args, **kwargs: "stub"

    setattr(stub_faker, "Faker", _DummyFaker)
    sys.modules.setdefault("faker", stub_faker)
    sys.modules.setdefault("faker.config", stub_faker_config)


_install_generator_stubs()

MarkdownDatasetGenerator = importlib.import_module("generation.markdown_dataset").MarkdownDatasetGenerator
BaseGenerator = importlib.import_module("generator.base").BaseGenerator
realism_stats_module = importlib.import_module("generator.realism_stats")
RealismStatsAccumulator = realism_stats_module.RealismStatsAccumulator
compute_realism_stats = realism_stats_module.compute_realism_stats


class DummyStreamingGenerator(BaseGenerator):
    def generate_single(self, **kwargs):
        raise NotImplementedError

    def generate(self, num_images: int, **kwargs) -> int:
        metadata_handle = kwargs["metadata_handle"]
        stats_accumulator = kwargs["stats_accumulator"]

        for idx in range(num_images):
            image = Image.new("RGB", (idx + 10, idx + 20), color="white")
            filename = f"sample_{idx:03d}.png"
            self.save_image(image, filename)
            metadata = {
                "file_name": str(self.output_dir / filename),
                "GT_markdown": f"sample-{idx}",
                "GT_json": {"idx": idx},
                "image_width": image.width,
                "image_height": image.height,
            }
            self.append_metadata(metadata_handle, stats_accumulator, metadata)

        return num_images


def test_realism_stats_accumulator_matches_batch_stats() -> None:
    metadata = [
        {
            "format": "markdown",
            "GT_markdown": "# Title\n\nBody",
            "GT_json": [{"type": "heading"}],
            "image_width": 640,
            "image_height": 480,
            "merge_order": ["text", "table"],
            "entities": {"name": "alice"},
        },
        {
            "format": "markdown",
            "GT_markdown": "## Subtitle",
            "GT_json": [{"type": "heading"}],
            "image_width": 800,
            "image_height": 600,
            "merge_order": ["formula"],
            "entities": {"name": "bob", "date": "2024-01-01"},
        },
    ]

    accumulator = RealismStatsAccumulator(format_name="markdown")
    for item in metadata:
        accumulator.update(item)

    assert accumulator.finalize() == compute_realism_stats(metadata, format_name="markdown")


def test_base_generator_run_streams_metadata_and_writes_stats(tmp_path: Path) -> None:
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "dummy.ttf").write_bytes(b"font")

    output_dir = tmp_path / "generated"
    generator = DummyStreamingGenerator(output_dir=str(output_dir), font_dir=str(font_dir), lang="ko")

    result = generator.run(num_images=3)

    assert result == str(output_dir)

    metadata_path = output_dir / "metadata.jsonl"
    stats_path = output_dir / "realism_stats.json"
    assert metadata_path.exists()
    assert stats_path.exists()

    lines = metadata_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    rows = [json.loads(line) for line in lines]
    assert rows[0]["GT_markdown"] == "sample-0"
    assert rows[-1]["image_height"] == 22

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["total_samples"] == 3
    assert stats["field_presence"]["GT_markdown"] == 3
    assert stats["numeric_field_stats"]["image_width"]["max"] == 12.0


class FakeMarkdownGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _configure_generation(self, **kwargs) -> None:
        return None

    def generate_single(self, sample_index: int = 0, **kwargs):
        image = Image.new("RGB", (100 + sample_index, 200 + sample_index), color="white")
        metadata = {
            "GT_markdown": f"# Sample {sample_index}",
            "GT_json": [{"type": "heading", "index": sample_index}],
            "format": "markdown",
            "sample_index": sample_index,
        }
        return image, metadata

    def save_image(self, image: Image.Image, filename: str) -> Path:
        destination = self.output_dir / filename
        image.save(destination)
        return destination


def test_markdown_dataset_generator_run_streams_top_level_metadata_and_stats(tmp_path: Path, monkeypatch) -> None:
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "dummy.ttf").write_bytes(b"font")

    dataset_generator = MarkdownDatasetGenerator(
        output_dir=str(tmp_path / "markdown_dataset"),
        font_dir=str(font_dir),
        lang="ko",
    )
    dataset_generator._markdown_generator = FakeMarkdownGenerator(dataset_generator.output_dir / "markdown")
    monkeypatch.setattr(
        "generation.markdown_dataset.attach_unified_ground_truth",
        lambda _fmt, meta: dict(meta, GT_json={"kind": "markdown"}),
    )

    result = dataset_generator.run(num_images=2, sample_start_index=10)

    assert result == str(dataset_generator.output_dir)

    metadata_path = dataset_generator.output_dir / "metadata.jsonl"
    stats_path = dataset_generator.output_dir / "realism_stats.json"
    assert metadata_path.exists()
    assert stats_path.exists()

    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["GT_markdown"] == "# Sample 10"
    assert rows[0]["sample_index"] == 10
    assert rows[1]["file_name"].endswith("markdown_00011.png")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["total_samples"] == 2
    assert stats["format"] == "markdown"
    assert stats["format_counts"]["markdown"] == 2
