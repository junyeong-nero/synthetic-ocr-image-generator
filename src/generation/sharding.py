import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from generator.realism_stats import RealismStatsAccumulator, write_realism_stats


@dataclass(frozen=True)
class ShardSpec:
    index: int
    start_index: int
    num_images: int

    @property
    def name(self) -> str:
        return f"shard-{self.index:06d}"


def plan_shards(
    total_size: int,
    shard_size: Optional[int],
    max_shards: Optional[int] = None,
) -> list[ShardSpec]:
    if total_size <= 0:
        return []

    normalized_shard_size = total_size if shard_size is None or shard_size <= 0 else shard_size
    shards: list[ShardSpec] = []
    start_index = 0
    shard_index = 0

    while start_index < total_size:
        if max_shards is not None and shard_index >= max_shards:
            break
        count = min(normalized_shard_size, total_size - start_index)
        shards.append(ShardSpec(index=shard_index, start_index=start_index, num_images=count))
        start_index += count
        shard_index += 1

    return shards


class RunManifest:
    def __init__(self, path: Path, data: Dict[str, Any]):
        self.path = path
        self.data = data

    @classmethod
    def create(
        cls,
        *,
        path: Path,
        generator_name: str,
        size: int,
        shard_size: int,
        mixed: bool,
        lang: str,
        seed: Optional[int],
        repo_id: Optional[str],
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> "RunManifest":
        manifest = cls(
            path,
            {
                "version": 1,
                "status": "initialized",
                "generator": generator_name,
                "size": size,
                "shard_size": shard_size,
                "mixed": mixed,
                "lang": lang,
                "seed": seed,
                "repo_id": repo_id,
                "generation_config": generation_config or {},
                "completed_shards": [],
                "failed_shards": [],
                "shards": {},
            },
        )
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        with open(path, encoding="utf-8") as handle:
            return cls(path, json.load(handle))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
        temp_path.replace(self.path)

    def validate_or_raise(
        self,
        *,
        size: int,
        shard_size: int,
        mixed: bool,
        lang: str,
        seed: Optional[int],
        repo_id: Optional[str],
    ) -> None:
        expected = {
            "size": size,
            "shard_size": shard_size,
            "mixed": mixed,
            "lang": lang,
            "seed": seed,
            "repo_id": repo_id,
        }
        for key, value in expected.items():
            if key == "repo_id" and value is None:
                continue
            if self.data.get(key) != value:
                raise ValueError(
                    f"Resume parameters do not match existing manifest for '{key}': "
                    f"expected {self.data.get(key)!r}, got {value!r}"
                )

    def initialize_shards(self, shards: Iterable[ShardSpec]) -> None:
        shard_entries = self.data.setdefault("shards", {})
        for shard in shards:
            shard_entries.setdefault(
                shard.name,
                {
                    "index": shard.index,
                    "start_index": shard.start_index,
                    "num_images": shard.num_images,
                    "status": "pending",
                },
            )
        self.save()

    def is_completed(self, shard: ShardSpec) -> bool:
        return shard.name in set(self.data.get("completed_shards", []))

    def mark_started(self, shard: ShardSpec) -> None:
        entry = self.data.setdefault("shards", {}).setdefault(shard.name, {})
        entry.update(
            {
                "index": shard.index,
                "start_index": shard.start_index,
                "num_images": shard.num_images,
                "status": "running",
            }
        )
        self.data["status"] = "running"
        self.save()

    def mark_completed(self, shard: ShardSpec, output_dir: str, generated_count: int) -> None:
        entry = self.data.setdefault("shards", {}).setdefault(shard.name, {})
        entry.update(
            {
                "index": shard.index,
                "start_index": shard.start_index,
                "num_images": shard.num_images,
                "generated_count": generated_count,
                "output_dir": output_dir,
                "status": "completed",
            }
        )
        completed = set(self.data.setdefault("completed_shards", []))
        completed.add(shard.name)
        self.data["completed_shards"] = sorted(completed)
        self.data["failed_shards"] = [name for name in self.data.get("failed_shards", []) if name != shard.name]
        self.save()

    def mark_failed(self, shard: ShardSpec, error_message: str) -> None:
        entry = self.data.setdefault("shards", {}).setdefault(shard.name, {})
        entry.update(
            {
                "index": shard.index,
                "start_index": shard.start_index,
                "num_images": shard.num_images,
                "status": "failed",
                "error": error_message,
            }
        )
        failed = set(self.data.setdefault("failed_shards", []))
        failed.add(shard.name)
        self.data["failed_shards"] = sorted(failed)
        self.data["status"] = "failed"
        self.save()

    def mark_finished(self) -> None:
        self.data["status"] = "completed"
        self.save()


def ensure_resume_state(output_dir: Path, manifest_path: Path, resume: bool) -> Optional[RunManifest]:
    manifest_exists = manifest_path.exists()
    has_existing_artifacts = output_dir.exists() and any(output_dir.iterdir())

    if resume:
        if not manifest_exists:
            raise ValueError(f"Cannot resume without run manifest: {manifest_path}")
        return RunManifest.load(manifest_path)

    if manifest_exists or has_existing_artifacts:
        raise ValueError(
            f"Output directory '{output_dir}' already contains generation artifacts. "
            "Use --resume or choose a new output directory."
        )
    return None


def write_shard_success_marker(shard_dir: Path, generated_count: int) -> None:
    success_path = shard_dir / "_SUCCESS"
    with open(success_path, "w", encoding="utf-8") as handle:
        handle.write(str(generated_count))


def shard_success_marker_exists(shard_dir: Path) -> bool:
    return (shard_dir / "_SUCCESS").exists()


def rebuild_aggregate_outputs(
    *,
    output_dir: Path,
    shards: Iterable[ShardSpec],
    format_name: Optional[str] = None,
) -> int:
    metadata_path = output_dir / "metadata.jsonl"
    accumulator = RealismStatsAccumulator(format_name=format_name)
    total_rows = 0

    with open(metadata_path, "w", encoding="utf-8") as destination:
        for shard in shards:
            shard_metadata_path = output_dir / "shards" / shard.name / "metadata.jsonl"
            if not shard_metadata_path.exists():
                continue
            with open(shard_metadata_path, encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    destination.write(line)
                    accumulator.update(json.loads(line))
                    total_rows += 1

    write_realism_stats(output_dir, accumulator)
    return total_rows
