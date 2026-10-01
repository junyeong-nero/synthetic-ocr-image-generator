"""Publish one generated shard to the Hub as parquet, then free its disk space.

A shard becomes a single commit holding ``data/train-XXXXXX.parquet`` and
``data/test-XXXXXX.parquet`` (images embedded as bytes), so it is either fully on
the Hub or absent. The column schema is fixed once per run (``load_or_create_features``)
so every parquet file has identical columns.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from src.generation.distribution_summary import summarize_metadata_distribution

# Shared with upload_subset_to_hub so both publish paths emit the same columns.
from src.generation.hub_dataset import (
    _build_features,
    _build_upload_record,
    _infer_feature_dict,
    _iter_normalized_metadata_rows,
    _read_first_normalized_metadata_row,
)
from src.generation.hub_upload import plan_split_indices

logger = logging.getLogger(__name__)

SPLITS = ("train", "test")
SUMMARY_NAME = "shard_summary.json"
FEATURES_NAME = "features.json"
STAGING_DIR = "upload"
DEFAULT_ROW_GROUP_SIZE = 100

_DATA_FILE_RE = re.compile(r"^data/(train|test)-(\d{6})\.parquet$")


@dataclass(frozen=True)
class StagedFile:
    split: str
    path: Path
    repo_path: str
    rows: int
    size: int


def shard_repo_paths(shard_index: int) -> dict[str, str]:
    return {split: f"data/{split}-{shard_index:06d}.parquet" for split in SPLITS}


def classify_repo_data_files(repo_files: Iterable[str]) -> tuple[set[int], list[str]]:
    """Return (indices of published shards, parquet files that don't follow the shard naming).

    Both parquet files of a shard land in one commit, so the train file alone
    proves the shard is complete.
    """
    published: set[int] = set()
    foreign: list[str] = []
    for path in repo_files:
        if not (path.startswith("data/") and path.endswith(".parquet")):
            continue
        match = _DATA_FILE_RE.match(path)
        if match is None:
            foreign.append(path)
        elif match.group(1) == "train":
            published.add(int(match.group(2)))
    return published, sorted(foreign)


_NEVER_RETRIED = (ValueError, TypeError, KeyError, AttributeError, FileNotFoundError, PermissionError)


def is_retryable(exc: BaseException) -> bool:
    """Network errors and 408/429/5xx are worth retrying; other 4xx (auth, missing repo) and local bugs are not."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status in (408, 429) or status >= 500
    return not isinstance(exc, _NEVER_RETRIED)


def with_retries(
    action: Callable[[], Any],
    *,
    description: str,
    attempts: int = 5,
    base_delay: float = 15.0,
    sleep: Optional[Callable[[float], None]] = None,
) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:
            if attempt == attempts or not is_retryable(exc):
                raise
            delay = min(600.0, base_delay * 2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.0fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            (sleep or time.sleep)(delay)


def load_or_create_features(features_path: Path, metadata_path: Path):
    """Return the run-wide column schema; the first shard to finish defines it."""
    from datasets import Features

    if not features_path.exists():
        sample = _read_first_normalized_metadata_row(metadata_path, selected_indices=None)
        if sample is None:
            raise ValueError(f"No metadata rows in '{metadata_path}' to infer the schema from")
        features = _build_features(_infer_feature_dict(sample))
        candidate = features_path.with_name(f"{features_path.name}.{os.getpid()}.tmp")
        candidate.write_text(json.dumps(features.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            # link() fails if the file exists, so concurrent workers agree on one schema.
            os.link(candidate, features_path)
        except FileExistsError:
            pass
        finally:
            candidate.unlink(missing_ok=True)
    return Features.from_dict(json.loads(features_path.read_text(encoding="utf-8")))


def count_rows(metadata_path: Path) -> int:
    with open(metadata_path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_shard_parquet(
    *,
    shard_dir: Path,
    metadata_path: Path,
    features,
    split_indices: dict[str, set[int]],
    shard_index: int,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
) -> tuple[list[StagedFile], Counter]:
    """Write the shard's rows to per-split parquet files under ``shard_dir/upload``.

    Returns the staged files and a count of metadata keys that are not columns
    of ``features`` (dropped, since the schema is fixed for the run).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    staging = shard_dir / STAGING_DIR
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    schema = features.arrow_schema
    columns = set(features) | {"file_name"}
    repo_paths = shard_repo_paths(shard_index)
    split_of_row = {row: split for split, rows in split_indices.items() for row in rows}
    temp_paths = {split: staging / f"{Path(repo_paths[split]).name}.tmp" for split in SPLITS}
    writers: dict[str, Any] = {}
    buffers: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    row_counts: Counter = Counter()
    dropped_keys: Counter = Counter()

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        if split not in writers:
            writers[split] = pq.ParquetWriter(temp_paths[split], schema, compression="zstd")
        writers[split].write_table(
            pa.Table.from_pylist(buffers[split], schema=schema),
            row_group_size=row_group_size,
        )
        buffers[split].clear()

    try:
        for position, (_, data) in enumerate(_iter_normalized_metadata_rows(metadata_path)):
            dropped_keys.update(key for key in data if key not in columns)
            record = _build_upload_record(data, features, row_index=position)
            image_path = Path(record["image"])
            record["image"] = {"bytes": image_path.read_bytes(), "path": image_path.name}
            split = split_of_row[position]
            buffers[split].append(record)
            row_counts[split] += 1
            if len(buffers[split]) >= row_group_size:
                flush(split)
        for split in SPLITS:
            flush(split)
    finally:
        for writer in writers.values():
            writer.close()

    staged: list[StagedFile] = []
    for split in SPLITS:
        if split not in writers:
            continue
        final_path = staging / Path(repo_paths[split]).name
        temp_paths[split].replace(final_path)
        staged.append(
            StagedFile(
                split=split,
                path=final_path,
                repo_path=repo_paths[split],
                rows=row_counts[split],
                size=final_path.stat().st_size,
            )
        )
    return staged, dropped_keys


def stage_shard(
    *,
    shard_dir: Path,
    features_path: Path,
    shard_index: int,
    train_ratio: float,
    expected_rows: int,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
) -> dict[str, Any]:
    """Turn a generated shard into upload-ready parquet files and record them in the shard summary."""
    metadata_path = shard_dir / "metadata.jsonl"
    rows = count_rows(metadata_path)
    if rows != expected_rows:
        raise ValueError(f"{shard_dir.name}: metadata has {rows} rows, expected {expected_rows}")

    features = load_or_create_features(features_path, metadata_path)
    split_indices = plan_split_indices(rows, train_ratio, seed=shard_index)
    staged, dropped_keys = write_shard_parquet(
        shard_dir=shard_dir,
        metadata_path=metadata_path,
        features=features,
        split_indices=split_indices,
        shard_index=shard_index,
        row_group_size=row_group_size,
    )
    if dropped_keys:
        logger.warning("%s: metadata keys missing from the run schema were dropped: %s", shard_dir.name, dict(dropped_keys))

    return {
        "version": 1,
        "shard": shard_dir.name,
        "index": shard_index,
        "status": "parquet_ready",
        "rows": rows,
        "files": [
            {
                "split": item.split,
                "repo_path": item.repo_path,
                "path": str(item.path.relative_to(shard_dir)),
                "rows": item.rows,
                "size": item.size,
            }
            for item in staged
        ],
        "distribution": summarize_metadata_distribution(metadata_path),
        "dropped_keys": dict(dropped_keys),
    }


def read_summary(shard_dir: Path) -> Optional[dict[str, Any]]:
    path = shard_dir / SUMMARY_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_summary(shard_dir: Path, summary: dict[str, Any]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    temp_path = shard_dir / f"{SUMMARY_NAME}.tmp"
    temp_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(shard_dir / SUMMARY_NAME)


def staged_files_from_summary(shard_dir: Path, summary: dict[str, Any]) -> list[StagedFile]:
    return [
        StagedFile(
            split=item["split"],
            path=shard_dir / item["path"],
            repo_path=item["repo_path"],
            rows=item["rows"],
            size=item["size"],
        )
        for item in summary.get("files", [])
    ]


def has_intact_stage(shard_dir: Path, summary: Optional[dict[str, Any]], fingerprint: str) -> bool:
    """True when a previous attempt left matching, complete parquet files that only need uploading."""
    if not summary or summary.get("status") != "parquet_ready" or summary.get("fingerprint") != fingerprint:
        return False
    staged = staged_files_from_summary(shard_dir, summary)
    return bool(staged) and all(item.path.exists() and item.path.stat().st_size == item.size for item in staged)


def upload_shard(
    api: Any,
    repo_id: str,
    staged: list[StagedFile],
    *,
    shard_index: int,
    rows: int,
    sleep: Optional[Callable[[float], None]] = None,
) -> str:
    """Commit the shard's parquet files in one commit, verify them on the Hub, return the commit id."""
    from huggingface_hub import CommitOperationAdd

    def commit():
        operations = [
            CommitOperationAdd(path_in_repo=item.repo_path, path_or_fileobj=str(item.path)) for item in staged
        ]
        return api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add shard {shard_index:06d} ({rows:,} rows)",
        )

    info = with_retries(commit, description=f"Uploading shard {shard_index:06d}", sleep=sleep)
    with_retries(
        lambda: verify_remote_files(api, repo_id, staged),
        description=f"Verifying shard {shard_index:06d}",
        attempts=3,
        base_delay=5.0,
        sleep=sleep,
    )
    return str(getattr(info, "oid", "") or "")


def verify_remote_files(api: Any, repo_id: str, staged: list[StagedFile]) -> None:
    infos = api.get_paths_info(repo_id, paths=[item.repo_path for item in staged], repo_type="dataset")
    remote_sizes = {info.path: getattr(info, "size", None) for info in infos}
    for item in staged:
        if remote_sizes.get(item.repo_path) != item.size:
            raise RuntimeError(
                f"Hub copy of {item.repo_path} does not match the local file "
                f"(remote size {remote_sizes.get(item.repo_path)!r}, local {item.size})"
            )


def cleanup_shard_dir(shard_dir: Path) -> None:
    """Delete the bulky shard artifacts; the summary, realism stats and _SUCCESS marker stay."""
    shutil.rmtree(shard_dir / "markdown", ignore_errors=True)
    shutil.rmtree(shard_dir / STAGING_DIR, ignore_errors=True)
    (shard_dir / "metadata.jsonl").unlink(missing_ok=True)


def aggregate_summaries(
    task_dir: Path,
    shard_names: Iterable[str],
) -> tuple[int, dict[str, int], dict[str, dict[str, int]]]:
    """Sum row counts, split counts and distribution tables over the shard summaries on disk."""
    total_rows = 0
    split_counts: Counter = Counter()
    distribution: dict[str, Counter] = {}
    for name in shard_names:
        summary = read_summary(task_dir / "shards" / name)
        if summary is None:
            logger.warning("No shard summary for %s; it is missing from the dataset card totals", name)
            continue
        total_rows += int(summary.get("rows", 0))
        for item in summary.get("files", []):
            split_counts[item["split"]] += int(item["rows"])
        for field, counts in (summary.get("distribution") or {}).items():
            distribution.setdefault(field, Counter()).update(counts)
    merged = {field: dict(counter.most_common()) for field, counter in distribution.items()}
    return total_rows, dict(split_counts), merged
