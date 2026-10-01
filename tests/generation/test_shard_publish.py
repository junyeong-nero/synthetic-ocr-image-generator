import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from datasets import Dataset
from PIL import Image as PILImage
from shard_fakes import FakeHubApi, write_fake_shard

from src.generation import shard_publish as sp
from src.generation.hub_upload import plan_split_indices


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


def _stage(tmp_path: Path, shard_index: int = 3, count: int = 25, features_path: Path | None = None, **row_fields):
    shard_dir = tmp_path / f"shard-{shard_index:06d}"
    write_fake_shard(shard_dir, shard_index * 10_000, count, extra_row_fields=row_fields)
    summary = sp.stage_shard(
        shard_dir=shard_dir,
        features_path=features_path or tmp_path / sp.FEATURES_NAME,
        shard_index=shard_index,
        train_ratio=0.9,
        expected_rows=count,
    )
    return shard_dir, summary


def test_shard_repo_paths_are_zero_padded_per_split() -> None:
    assert sp.shard_repo_paths(42) == {
        "train": "data/train-000042.parquet",
        "test": "data/test-000042.parquet",
    }


def test_classify_repo_data_files_reports_published_shards_and_foreign_files() -> None:
    published, foreign = sp.classify_repo_data_files(
        [
            ".gitattributes",
            "README.md",
            "data/train-000000.parquet",
            "data/test-000000.parquet",
            "data/train-000007.parquet",
            "data/test-000009.parquet",
            "data/train-00000-of-00001.parquet",
        ]
    )

    # Only train files count: both splits of a shard share one commit.
    assert published == {0, 7}
    assert foreign == ["data/train-00000-of-00001.parquet"]


def test_plan_split_indices_default_seed_is_unchanged_and_seed_varies_the_split() -> None:
    assert plan_split_indices(100, 0.9) == plan_split_indices(100, 0.9, seed=42)
    assert plan_split_indices(100, 0.9, seed=1)["test"] != plan_split_indices(100, 0.9, seed=2)["test"]


def test_is_retryable_distinguishes_transient_and_permanent_errors() -> None:
    class HttpError(Exception):
        def __init__(self, status):
            self.response = type("Response", (), {"status_code": status})()

    assert sp.is_retryable(ConnectionError("reset"))
    assert sp.is_retryable(HttpError(503))
    assert sp.is_retryable(HttpError(429))
    assert not sp.is_retryable(HttpError(403))
    assert not sp.is_retryable(HttpError(404))
    assert not sp.is_retryable(ValueError("bad repo id"))


def test_with_retries_gives_up_after_the_last_attempt() -> None:
    calls = []

    def flaky():
        calls.append(1)
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        sp.with_retries(flaky, description="test", attempts=3)
    assert len(calls) == 3


def test_with_retries_does_not_retry_permanent_errors() -> None:
    calls = []

    def forbidden():
        calls.append(1)
        raise ValueError("bad")

    with pytest.raises(ValueError):
        sp.with_retries(forbidden, description="test", attempts=5)
    assert len(calls) == 1


def test_stage_shard_writes_loadable_parquet_with_embedded_images(tmp_path: Path) -> None:
    shard_dir, summary = _stage(tmp_path)

    files = {item["split"]: shard_dir / item["path"] for item in summary["files"]}
    assert summary["rows"] == 25
    assert summary["status"] == "parquet_ready"
    assert {item["repo_path"] for item in summary["files"]} == {
        "data/train-000003.parquet",
        "data/test-000003.parquet",
    }

    train = Dataset.from_parquet(str(files["train"]))
    test = Dataset.from_parquet(str(files["test"]))
    assert (len(train), len(test)) == (22, 3)
    assert isinstance(train[0]["image"], PILImage.Image)
    assert train[0]["GT_markdown"].startswith("# document 3")
    assert json.loads(train[0]["GT_json"])[0]["type"] == "heading"

    # The image is embedded with its bare file name, never the local absolute path.
    table = pq.read_table(files["train"])
    embedded = table.column("image")[0].as_py()
    assert embedded["path"].startswith("markdown_") and "/" not in embedded["path"]
    assert embedded["bytes"][:8] == b"\x89PNG\r\n\x1a\n"
    text_cells = json.dumps({k: v for k, v in table.to_pydict().items() if k != "image"}, default=str)
    assert str(tmp_path) not in text_cells


def test_stage_shard_split_assignment_is_deterministic(tmp_path: Path) -> None:
    first_dir, first = _stage(tmp_path / "a")
    second_dir, second = _stage(tmp_path / "b")

    def test_docs(shard_dir, summary):
        item = next(f for f in summary["files"] if f["split"] == "test")
        return Dataset.from_parquet(str(shard_dir / item["path"]))["GT_markdown"]

    assert test_docs(first_dir, first) == test_docs(second_dir, second)


def test_first_shard_defines_the_schema_and_later_extra_keys_are_dropped(tmp_path: Path) -> None:
    _, first = _stage(tmp_path / "a", shard_index=0, features_path=tmp_path / "features.json")
    second_dir, second = _stage(
        tmp_path / "b", shard_index=1, features_path=tmp_path / "features.json", late_flag="x"
    )

    assert first["dropped_keys"] == {}
    assert second["dropped_keys"] == {"late_flag": 25}
    columns = pq.read_schema(second_dir / second["files"][0]["path"]).names
    assert "late_flag" not in columns
    assert columns == pq.read_schema(tmp_path / "a" / "shard-000000" / first["files"][0]["path"]).names


def test_stage_shard_rejects_a_row_count_mismatch(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shard-000000"
    write_fake_shard(shard_dir, 0, 5)

    with pytest.raises(ValueError, match="expected 6"):
        sp.stage_shard(
            shard_dir=shard_dir,
            features_path=tmp_path / sp.FEATURES_NAME,
            shard_index=0,
            train_ratio=0.9,
            expected_rows=6,
        )


def test_upload_shard_commits_both_splits_together_and_verifies_sizes(tmp_path: Path) -> None:
    shard_dir, summary = _stage(tmp_path)
    api = FakeHubApi()

    commit = sp.upload_shard(
        api,
        "demo/repo",
        sp.staged_files_from_summary(shard_dir, summary),
        shard_index=3,
        rows=summary["rows"],
    )

    assert commit == "commit1"
    assert len(api.commits) == 1
    assert sorted(api.commits[0][1]) == ["data/test-000003.parquet", "data/train-000003.parquet"]
    assert "000003" in api.commits[0][0]


def test_upload_shard_retries_transient_commit_failures(tmp_path: Path) -> None:
    shard_dir, summary = _stage(tmp_path)
    api = FakeHubApi(fail_commits=2)

    sp.upload_shard(api, "demo/repo", sp.staged_files_from_summary(shard_dir, summary), shard_index=3, rows=25)

    assert len(api.commits) == 1


def test_upload_shard_raises_when_the_hub_copy_has_the_wrong_size(tmp_path: Path) -> None:
    shard_dir, summary = _stage(tmp_path)
    api = FakeHubApi()
    api.size_delta = -1

    with pytest.raises(RuntimeError, match="does not match"):
        sp.upload_shard(api, "demo/repo", sp.staged_files_from_summary(shard_dir, summary), shard_index=3, rows=25)


def test_has_intact_stage_requires_matching_fingerprint_and_files(tmp_path: Path) -> None:
    shard_dir, summary = _stage(tmp_path)
    summary["fingerprint"] = "abc"

    assert sp.has_intact_stage(shard_dir, summary, "abc")
    assert not sp.has_intact_stage(shard_dir, summary, "different")
    assert not sp.has_intact_stage(shard_dir, None, "abc")

    (shard_dir / summary["files"][0]["path"]).write_bytes(b"truncated")
    assert not sp.has_intact_stage(shard_dir, summary, "abc")


def test_cleanup_shard_dir_keeps_only_the_small_artifacts(tmp_path: Path) -> None:
    shard_dir, summary = _stage(tmp_path)
    sp.write_summary(shard_dir, summary)
    (shard_dir / "_SUCCESS").write_text("25")
    (shard_dir / "realism_stats.json").write_text("{}")

    sp.cleanup_shard_dir(shard_dir)

    assert sorted(path.name for path in shard_dir.iterdir()) == ["_SUCCESS", "realism_stats.json", sp.SUMMARY_NAME]


def test_aggregate_summaries_sums_rows_splits_and_distribution(tmp_path: Path) -> None:
    for index in range(2):
        _, summary = _stage(tmp_path / "shards", shard_index=index, features_path=tmp_path / "features.json")
        sp.write_summary(tmp_path / "shards" / f"shard-{index:06d}", summary)

    rows, splits, distribution = sp.aggregate_summaries(tmp_path, ["shard-000000", "shard-000001"])

    assert rows == 50
    assert splits == {"train": 44, "test": 6}
    assert sum(distribution["document_family"].values()) == 50
