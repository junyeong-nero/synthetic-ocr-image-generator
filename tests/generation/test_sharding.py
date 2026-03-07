import json
from pathlib import Path

from generation.sharding import (
    RunManifest,
    ensure_resume_state,
    plan_shards,
    rebuild_aggregate_outputs,
    shard_success_marker_exists,
    write_shard_success_marker,
)


def test_plan_shards_splits_size_and_respects_max_shards() -> None:
    shards = plan_shards(total_size=11, shard_size=4, max_shards=2)

    assert len(shards) == 2
    assert shards[0].index == 0
    assert shards[0].start_index == 0
    assert shards[0].num_images == 4
    assert shards[1].index == 1
    assert shards[1].start_index == 4
    assert shards[1].num_images == 4



def test_run_manifest_tracks_started_completed_and_failed_shards(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest = RunManifest.create(
        path=manifest_path,
        generator_name="markdown",
        size=20,
        shard_size=5,
        mixed=False,
        lang="ko",
        seed=7,
        repo_id="demo/repo",
    )
    shards = plan_shards(total_size=20, shard_size=5)
    manifest.initialize_shards(shards)

    manifest.mark_started(shards[0])
    manifest.mark_completed(shards[0], output_dir="/tmp/out/shard-000000", generated_count=5)
    manifest.mark_failed(shards[1], "boom")

    loaded = RunManifest.load(manifest_path)
    assert loaded.is_completed(shards[0]) is True
    assert loaded.data["shards"][shards[1].name]["status"] == "failed"
    assert shards[0].name in loaded.data["completed_shards"]
    assert shards[1].name in loaded.data["failed_shards"]



def test_ensure_resume_state_requires_manifest_for_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    manifest_path = output_dir / "run_manifest.json"

    try:
        ensure_resume_state(output_dir, manifest_path, resume=True)
    except ValueError as exc:
        assert "Cannot resume without run manifest" in str(exc)
    else:
        raise AssertionError("resume without manifest should fail")



def test_rebuild_aggregate_outputs_concatenates_completed_shards(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    shards_root = output_dir / "shards"
    shards = plan_shards(total_size=4, shard_size=2)

    for shard in shards:
        shard_dir = shards_root / shard.name
        shard_dir.mkdir(parents=True)
        metadata_path = shard_dir / "metadata.jsonl"
        rows = [
            {
                "file_name": str(shard_dir / f"sample_{shard.start_index + offset}.png"),
                "GT_markdown": f"sample-{shard.start_index + offset}",
                "GT_json": {"idx": shard.start_index + offset},
                "image_width": 100 + offset,
                "image_height": 200 + offset,
            }
            for offset in range(shard.num_images)
        ]
        with open(metadata_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_shard_success_marker(shard_dir, shard.num_images)
        assert shard_success_marker_exists(shard_dir) is True

    total_rows = rebuild_aggregate_outputs(output_dir=output_dir, shards=shards)

    assert total_rows == 4
    metadata_rows = [
        json.loads(line)
        for line in (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metadata_rows) == 4
    assert metadata_rows[0]["GT_markdown"] == "sample-0"
    stats = json.loads((output_dir / "realism_stats.json").read_text(encoding="utf-8"))
    assert stats["total_samples"] == 4
    assert stats["field_presence"]["GT_markdown"] == 4
