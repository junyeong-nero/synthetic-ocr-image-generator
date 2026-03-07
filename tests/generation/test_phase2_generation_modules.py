import json
from pathlib import Path

from generation.sharding import RunManifest
from generation.git_metadata import normalize_github_url
from generation.ground_truth import attach_unified_ground_truth
from generation.hub_upload import upload_mixed_format_to_hub
from generation.readme_builder import build_dataset_readme


def test_normalize_github_url_handles_ssh_and_git_suffix() -> None:
    assert normalize_github_url("git@github.com:org/repo.git") == "https://github.com/org/repo"
    assert normalize_github_url("ssh://git@github.com/org/repo.git") == "https://github.com/org/repo"
    assert normalize_github_url("http://github.com/org/repo") == "https://github.com/org/repo"


def test_attach_unified_ground_truth_removes_legacy_keys_for_non_markdown() -> None:
    payload = {
        "ground_truth": "value",
        "markdown": "old",
        "json": {"old": True},
        "extra": 1,
    }
    updated = attach_unified_ground_truth("table", payload)

    assert "ground_truth" not in updated
    assert "markdown" not in updated
    assert "json" not in updated
    assert updated["extra"] == 1
    assert updated["GT_json"] == {"ground_truth": "value", "format": "table"}
    assert "```json" in updated["GT_markdown"]


def test_build_dataset_readme_contains_expected_sections(monkeypatch) -> None:
    monkeypatch.setattr(
        "generation.readme_builder.resolve_git_metadata",
        lambda: {
            "github_url": "https://github.com/org/repo",
            "commit": "abc123",
            "branch": "main",
        },
    )

    readme = build_dataset_readme(
        repo_id="org/dataset",
        lang="ko",
        size=10,
        generated_count=10,
        template="readme",
        template_family="sections",
        min_template_complexity=1,
        max_template_complexity=3,
        template_config_dir="configs/generator/templates",
        markdown_renderer="pil",
        style_profile="balanced",
        coverage_targets={"sections": 0.5},
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
        seed=7,
        split_counts={"train": 10},
    )

    assert "# Synthetic OCR Dataset" in readme
    assert "https://huggingface.co/datasets/org/dataset" in readme
    assert "https://github.com/org/repo" in readme
    assert "`abc123`" in readme
    assert "`main`" in readme
    assert "--template readme" in readme


def test_upload_mixed_format_to_hub_splits_and_uploads(monkeypatch, tmp_path: Path) -> None:
    records = []
    for idx in range(4):
        image_path = tmp_path / f"sample_{idx}.png"
        image_path.write_bytes(b"img")
        records.append(
            {
                "file_name": str(image_path),
                "GT_markdown": f"sample-{idx}",
                "GT_json": {"idx": idx},
            }
        )

    metadata_path = tmp_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    calls = []

    def _stub_upload_subset_to_hub(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("generation.hub_upload.upload_subset_to_hub", _stub_upload_subset_to_hub)

    counts = upload_mixed_format_to_hub(
        repo_id="org/dataset",
        output_dir=tmp_path,
        train_ratio=0.75,
        test_ratio=0.25,
    )

    assert counts == {"train": 3, "test": 1}
    assert len(calls) == 2
    assert {call["split"] for call in calls} == {"train", "test"}
    assert all(call["config_name"] == "default" for call in calls)
    assert all(call["reuse_existing_schema"] is True for call in calls)


def test_publish_pipeline_uses_manifest_context(monkeypatch, tmp_path: Path) -> None:
    from pipeline import publish_pipeline

    generated_path = tmp_path / "images_markdown"
    generated_path.mkdir()
    manifest = RunManifest.create(
        path=generated_path / "run_manifest.json",
        generator_name="markdown",
        size=12,
        shard_size=4,
        mixed=False,
        lang="ko",
        seed=7,
        repo_id="demo/repo",
        generation_config={
            "lang": "ko",
            "size": 12,
            "template": "readme",
            "template_family": "sections",
            "min_template_complexity": 1,
            "max_template_complexity": 3,
            "template_config_dir": "configs/generator/templates",
            "markdown_renderer": "pil",
            "style_profile": "balanced",
            "coverage_targets": ["sections=0.5"],
            "novelty_window": 80,
            "novelty_threshold": 0.95,
            "novelty_max_attempts": 4,
            "similar_char_ratio": 0.08,
            "similarity_db_path": None,
            "formula_source_mode": "mixed",
            "formula_dataset_path": None,
            "formula_dataset_weight": 0.45,
            "formula_random_weight": 0.30,
            "formula_synthetic_weight": 0.25,
            "add_noise": True,
            "add_blur": False,
            "mixed": False,
            "train_ratio": 0.9,
            "test_ratio": 0.1,
            "seed": 7,
        },
    )
    manifest.mark_finished()

    calls = []

    def _stub_upload_generated_dataset(**kwargs):
        calls.append(kwargs)
        return {"train": 12}

    monkeypatch.setattr("pipeline.upload_generated_dataset", _stub_upload_generated_dataset)

    counts = publish_pipeline(generated_path=str(generated_path))

    assert counts == {"train": 12}
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "demo/repo"
    assert calls[0]["generated_path"] == generated_path
    assert calls[0]["template"] == "readme"
    assert calls[0]["seed"] == 7
