import argparse
import time
from pathlib import Path

import pytest
from shard_fakes import FakeHubApi, install_fake_generator

from src.cli.generate import add_arguments, build_context_from_args, build_stream_upload_options
from src.generation import shard_publish as sp
from src.generation.options import GenerationOptions, GenerationTaskContext, PublishOptions
from src.generation.shard_runner import (
    ShardTask,
    StreamUploadOptions,
    _wait_for_free_space,
    publish_shards_streaming,
    run_shard_task,
)
from src.generation.sharding import RunManifest, plan_shards


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


@pytest.fixture
def uploaded_cards(monkeypatch):
    cards = []
    monkeypatch.setattr(
        "src.generation.shard_runner.upload_dataset_readme_to_hub",
        lambda **kwargs: cards.append(kwargs),
    )
    return cards


def _context(size: int = 30, repo_id: str | None = "demo/repo") -> GenerationTaskContext:
    return GenerationTaskContext(
        lang="ko",
        size=size,
        generation=GenerationOptions(seed=7),
        publish=PublishOptions(repo_id=repo_id, train_ratio=0.9, test_ratio=0.1),
    )


def _run(tmp_path: Path, api, *, size=30, shard_size=10, max_shards=None, resume=False, **option_kwargs):
    context = _context(size)
    shards = plan_shards(size, shard_size, max_shards=max_shards)
    manifest = RunManifest.create(
        path=tmp_path / "run_manifest.json",
        generator_name="markdown",
        size=size,
        shard_size=shard_size,
        lang="ko",
        seed=7,
        repo_id="demo/repo",
    )
    manifest.initialize_shards(shards)
    failed = publish_shards_streaming(
        manifest=manifest,
        shard_specs=shards,
        total_shards=len(plan_shards(size, shard_size)),
        context=context,
        task_dir=tmp_path,
        font_dir="fonts/ko",
        options=StreamUploadOptions(**option_kwargs),
        resume=resume,
        api=api,
    )
    return failed, manifest


def test_every_shard_is_uploaded_then_removed_locally(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch)
    api = FakeHubApi(exists=False)

    failed, manifest = _run(tmp_path, api)

    assert failed == []
    assert api.created == [("demo/repo", True)]
    assert sorted(api.repo_files) == sorted(
        f"data/{split}-{index:06d}.parquet" for index in range(3) for split in ("test", "train")
    )
    assert len(api.commits) == 3
    assert manifest.data["completed_shards"] == ["shard-000000", "shard-000001", "shard-000002"]
    for index in range(3):
        shard_dir = tmp_path / "shards" / f"shard-{index:06d}"
        assert sorted(path.name for path in shard_dir.iterdir()) == ["_SUCCESS", sp.SUMMARY_NAME]
    assert len(uploaded_cards) == 1
    assert "`train`: 27 samples" in uploaded_cards[0]["readme_content"]
    assert "`test`: 3 samples" in uploaded_cards[0]["readme_content"]


def test_resume_skips_shards_already_on_the_hub(tmp_path, monkeypatch, uploaded_cards) -> None:
    generated = install_fake_generator(monkeypatch)
    api = FakeHubApi(files={"data/train-000000.parquet": 10, "data/test-000000.parquet": 10})

    failed, manifest = _run(tmp_path, api, resume=True)

    assert failed == []
    assert generated == [10, 20]
    assert manifest.data["shards"]["shard-000000"]["status"] == "completed"
    assert len(api.commits) == 2


def test_existing_shards_without_resume_are_refused(tmp_path, monkeypatch) -> None:
    install_fake_generator(monkeypatch)
    api = FakeHubApi(files={"data/train-000000.parquet": 10})

    with pytest.raises(ValueError, match="--resume"):
        _run(tmp_path, api)
    assert api.commits == []


def test_parquet_files_from_another_layout_are_refused(tmp_path, monkeypatch) -> None:
    install_fake_generator(monkeypatch)
    api = FakeHubApi(files={"data/train-00000-of-00001.parquet": 10})

    with pytest.raises(ValueError, match="do not follow the shard naming"):
        _run(tmp_path, api, resume=True)


def test_the_repo_is_created_public_only_when_asked(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch)
    api = FakeHubApi(exists=False)

    _run(tmp_path, api, private=False)

    assert api.created == [("demo/repo", False)]


def test_an_existing_repo_is_left_alone(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch)
    api = FakeHubApi(exists=True)

    _run(tmp_path, api, private=False)

    assert api.created == []


def test_consecutive_failures_stop_new_shards_from_starting(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch, fail_starts=[0, 10, 20, 30, 40, 50])
    api = FakeHubApi()

    failed, manifest = _run(tmp_path, api, size=60)

    assert failed == ["shard-000000", "shard-000001", "shard-000002"]
    statuses = [manifest.data["shards"][f"shard-{i:06d}"]["status"] for i in range(6)]
    assert statuses == ["failed"] * 3 + ["pending"] * 3
    assert uploaded_cards == []


def test_a_success_resets_the_failure_streak(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch, fail_starts=[0, 10, 30, 40])

    failed, manifest = _run(tmp_path, FakeHubApi(), size=60)

    assert failed == ["shard-000000", "shard-000001", "shard-000003", "shard-000004"]
    assert manifest.data["completed_shards"] == ["shard-000002", "shard-000005"]


def test_a_zero_limit_disables_the_failure_streak_stop(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch, fail_starts=[0, 10, 20, 30])

    failed, _ = _run(tmp_path, FakeHubApi(), size=40, max_consecutive_failures=0)

    assert len(failed) == 4


def test_a_failed_shard_is_recorded_and_the_rest_still_upload(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch, fail_starts=[10])
    api = FakeHubApi()

    failed, manifest = _run(tmp_path, api)

    assert failed == ["shard-000001"]
    assert manifest.data["failed_shards"] == ["shard-000001"]
    assert manifest.data["completed_shards"] == ["shard-000000", "shard-000002"]
    assert len(api.commits) == 2
    assert uploaded_cards == []  # the card waits until every shard is on the Hub


def test_partial_runs_do_not_publish_the_dataset_card(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch)
    api = FakeHubApi()

    failed, _ = _run(tmp_path, api, max_shards=2)

    assert failed == []
    assert len(api.commits) == 2
    assert uploaded_cards == []


def test_skipped_samples_lower_the_row_count_and_are_recorded(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch, skip_indices=[13])
    api = FakeHubApi()

    failed, manifest = _run(tmp_path, api)

    assert failed == []
    assert manifest.data["shards"]["shard-000001"]["generated_count"] == 9
    assert manifest.data["shards"]["shard-000001"]["skipped_samples"] == [13]
    assert "`train`: 26 samples" in uploaded_cards[0]["readme_content"]


def test_dry_run_stages_parquet_without_touching_the_hub(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch)
    api = FakeHubApi()

    failed, manifest = _run(tmp_path, api=None, dry_run=True)

    assert failed == []
    assert api.calls == 0
    assert uploaded_cards == []
    for index in range(3):
        staged = tmp_path / "shards" / f"shard-{index:06d}" / "upload"
        assert sorted(path.name for path in staged.iterdir()) == [f"test-{index:06d}.parquet", f"train-{index:06d}.parquet"]
    assert (tmp_path / "DATASET_CARD.md").exists()


def test_keep_local_leaves_the_shard_files_in_place(tmp_path, monkeypatch, uploaded_cards) -> None:
    install_fake_generator(monkeypatch)
    api = FakeHubApi()

    _run(tmp_path, api, keep_local=True)

    shard_dir = tmp_path / "shards" / "shard-000000"
    assert (shard_dir / "metadata.jsonl").exists()
    assert (shard_dir / "upload" / "train-000000.parquet").exists()


def _task(tmp_path: Path, **kwargs) -> ShardTask:
    return ShardTask(
        task_dir=tmp_path,
        font_dir="fonts/ko",
        lang="ko",
        generation=GenerationOptions(seed=7),
        shard=plan_shards(30, 10)[1],
        repo_id="demo/repo",
        train_ratio=0.9,
        **kwargs,
    )


def test_local_files_survive_a_failed_verification_and_the_next_attempt_only_reuploads(tmp_path, monkeypatch) -> None:
    generated = install_fake_generator(monkeypatch)
    api = FakeHubApi()
    api.size_delta = -1

    with pytest.raises(RuntimeError, match="does not match"):
        run_shard_task(_task(tmp_path), api=api)

    shard_dir = tmp_path / "shards" / "shard-000001"
    assert (shard_dir / "metadata.jsonl").exists()
    assert (shard_dir / "upload" / "train-000001.parquet").exists()

    api.size_delta = 0
    summary = run_shard_task(_task(tmp_path), api=api)

    assert generated == [10]  # generated once; the retry reused the staged parquet
    assert summary["status"] == "uploaded"
    assert not (shard_dir / "upload").exists()


def test_a_changed_configuration_invalidates_staged_leftovers(tmp_path, monkeypatch) -> None:
    generated = install_fake_generator(monkeypatch)
    run_shard_task(_task(tmp_path, dry_run=True))

    changed = ShardTask(**{**_task(tmp_path, dry_run=True).__dict__, "generation": GenerationOptions(seed=8)})
    run_shard_task(changed)

    assert generated == [10, 10]


def test_transient_commit_failures_are_retried(tmp_path, monkeypatch) -> None:
    install_fake_generator(monkeypatch)
    api = FakeHubApi(fail_commits=2)

    summary = run_shard_task(_task(tmp_path), api=api)

    assert summary["commit"] == "commit1"
    assert len(api.commits) == 1


def test_wait_for_free_space_raises_after_the_maximum_wait(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shutil.disk_usage", lambda path: type("Usage", (), {"free": 1_000_000_000})())

    _wait_for_free_space(tmp_path, 0.5)  # 1 GB free >= 0.5 GB needed
    with pytest.raises(OSError, match="GB free"):
        _wait_for_free_space(tmp_path, 5, poll_seconds=1, max_wait_seconds=3)


def test_stream_upload_options_require_the_stream_flag() -> None:
    def args(**overrides):
        base = dict(upload_each_shard=False, workers=1, keep_uploaded_shards=False, upload_dry_run=False)
        return argparse.Namespace(**{**base, **overrides})

    assert build_stream_upload_options(args()) is None
    assert build_stream_upload_options(args(upload_each_shard=True, workers=4)) == StreamUploadOptions(workers=4)
    assert build_stream_upload_options(args(upload_each_shard=True)).private is True
    assert build_stream_upload_options(args(upload_each_shard=True, public=True)).private is False
    with pytest.raises(ValueError, match="require --upload-each-shard"):
        build_stream_upload_options(args(workers=4))
    with pytest.raises(ValueError, match="require --upload-each-shard"):
        build_stream_upload_options(args(public=True))
    with pytest.raises(ValueError, match="at least 1"):
        build_stream_upload_options(args(upload_each_shard=True, workers=0))
    # Namespaces built before these flags existed must keep working.
    assert build_stream_upload_options(argparse.Namespace()) is None


def test_pipeline_streams_shards_and_rejects_combining_with_upload(tmp_path, monkeypatch, uploaded_cards) -> None:
    from src.pipeline import pipeline

    install_fake_generator(monkeypatch)
    api = FakeHubApi()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: api)

    pipeline(
        _context(25),
        output_dir=str(tmp_path),
        shard_size=10,
        stream_upload=StreamUploadOptions(),
    )

    assert len(api.commits) == 3
    manifest = RunManifest.load(tmp_path / "ko" / "images_markdown" / "run_manifest.json")
    assert manifest.data["status"] == "completed"
    assert manifest.data["shards"]["shard-000002"]["generated_count"] == 5

    with pytest.raises(ValueError, match="mutually exclusive"):
        pipeline(_context(25), output_dir=str(tmp_path / "other"), upload=True, stream_upload=StreamUploadOptions())


def test_pipeline_raises_after_the_run_when_a_shard_failed(tmp_path, monkeypatch, uploaded_cards) -> None:
    from src.pipeline import pipeline

    install_fake_generator(monkeypatch, fail_starts=[10])
    api = FakeHubApi()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: api)

    with pytest.raises(RuntimeError, match="shard-000001.*--resume"):
        pipeline(_context(25), output_dir=str(tmp_path), shard_size=10, stream_upload=StreamUploadOptions())

    assert len(api.commits) == 2


def test_license_and_text_source_reach_the_dataset_card_options() -> None:
    parser = add_arguments(argparse.ArgumentParser())

    default = build_context_from_args(parser.parse_args([]))
    assert (default.publish.license, default.publish.text_source) == ("unknown", None)

    chosen = build_context_from_args(
        parser.parse_args(["--license", "cc-by-sa-3.0", "--text-source", "Korean WikiText, CC BY-SA 3.0"])
    )
    assert chosen.publish.license == "cc-by-sa-3.0"
    assert chosen.publish.text_source == "Korean WikiText, CC BY-SA 3.0"
