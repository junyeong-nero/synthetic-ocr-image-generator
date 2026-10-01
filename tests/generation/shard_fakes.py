"""In-memory stand-ins for the generator and the Hub, shared by the shard publishing tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

from PIL import Image


def write_fake_shard(
    shard_dir: Path,
    start_index: int,
    count: int,
    skipped: Iterable[int] = (),
    extra_row_fields: Optional[dict] = None,
) -> None:
    """Write what MarkdownDatasetGenerator leaves in a shard dir: PNGs under markdown/ and metadata.jsonl."""
    images_dir = shard_dir / "markdown"
    images_dir.mkdir(parents=True, exist_ok=True)
    skipped = set(skipped)
    rows = []
    for offset in range(count):
        sample_index = start_index + offset
        if sample_index in skipped:
            continue
        image_path = images_dir / f"markdown_{sample_index:05d}.png"
        Image.new("RGB", (16 + offset % 5, 12), (offset * 7 % 255, 20, 30)).save(image_path)
        rows.append(
            {
                "template": "readme",
                "document_family": "technical" if offset % 2 else "business",
                "block_types": ["heading", "paragraph"],
                "block_type_counts": {"heading": 1, "paragraph": 1},
                "section_count": 2,
                "law_articles_used": False,
                "content_genre": None,
                "template_weight": 1.0,
                "GT_markdown": f"# document {sample_index}",
                "GT_json": [{"type": "heading", "text": f"document {sample_index}"}],
                "sample_index": sample_index,
                "sample_seed": 42 + sample_index,
                "novelty_score": 0.25,
                "image_width": 16,
                "image_height": 12,
                "file_name": str(image_path),
                **(extra_row_fields or {}),
            }
        )
    with open(shard_dir / "metadata.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def install_fake_generator(monkeypatch, *, fail_starts: Iterable[int] = (), skip_indices: Iterable[int] = ()):
    """Replace MarkdownDatasetGenerator in the runner; returns the list of shard start indices it generated."""
    generated: list[int] = []
    fail_starts = set(fail_starts)
    skip_indices = set(skip_indices)

    class FakeGenerator:
        def __init__(self, output_dir, font_dir, lang):
            self.output_dir = Path(output_dir)
            self.skipped_samples: list[int] = []

        def run(self, num_images, options, sample_start_index=0, skip_failed_samples=0):
            if sample_start_index in fail_starts:
                return None
            generated.append(sample_start_index)
            self.skipped_samples = [
                i for i in sorted(skip_indices) if sample_start_index <= i < sample_start_index + num_images
            ]
            write_fake_shard(self.output_dir, sample_start_index, num_images, self.skipped_samples)
            return str(self.output_dir)

    monkeypatch.setattr("src.generation.shard_runner.MarkdownDatasetGenerator", FakeGenerator)
    return generated


class FakeHubApi:
    """The slice of HfApi the shard publisher uses, backed by a dict of path -> size."""

    def __init__(self, files: Optional[dict[str, int]] = None, *, exists: bool = True, fail_commits: int = 0):
        self.repo_files = dict(files or {})
        self.exists = exists
        self.fail_commits = fail_commits
        self.size_delta = 0
        self.created: list[tuple[str, Optional[bool]]] = []
        self.commits: list[tuple[str, list[str]]] = []
        self.calls = 0

    def repo_exists(self, repo_id, repo_type=None):
        self.calls += 1
        return self.exists

    def create_repo(self, repo_id, repo_type=None, private=None, **kwargs):
        self.calls += 1
        self.exists = True
        self.created.append((repo_id, private))

    def list_repo_files(self, repo_id, repo_type=None):
        self.calls += 1
        return sorted(self.repo_files)

    def create_commit(self, repo_id, operations, *, commit_message, repo_type=None, **kwargs):
        self.calls += 1
        if self.fail_commits > 0:
            self.fail_commits -= 1
            raise ConnectionError("network unreachable")
        operations = list(operations)
        for operation in operations:
            self.repo_files[operation.path_in_repo] = os.path.getsize(operation.path_or_fileobj) + self.size_delta
        self.commits.append((commit_message, [operation.path_in_repo for operation in operations]))
        return SimpleNamespace(oid=f"commit{len(self.commits)}")

    def get_paths_info(self, repo_id, paths, repo_type=None):
        self.calls += 1
        return [SimpleNamespace(path=path, size=self.repo_files[path]) for path in paths if path in self.repo_files]
