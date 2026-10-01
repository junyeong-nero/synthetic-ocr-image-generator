"""Generate shards (optionally in parallel) and publish each one as soon as it is ready.

Every shard goes generate -> parquet -> one Hub commit -> remote size check -> local
delete, so disk use stays at a few shards regardless of the dataset size. The Hub
listing is the source of truth for what is done, which makes reruns with
``--resume`` safe after any crash.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import time
from collections import deque
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.generation import shard_publish as sp
from src.generation.hub_dataset import upload_dataset_readme_to_hub
from src.generation.markdown_dataset import MarkdownDatasetGenerator
from src.generation.options import GenerationOptions, GenerationTaskContext
from src.generation.readme_builder import build_dataset_readme
from src.generation.sharding import RunManifest, ShardSpec, write_shard_success_marker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamUploadOptions:
    workers: int = 1
    keep_local: bool = False
    dry_run: bool = False
    row_group_size: int = sp.DEFAULT_ROW_GROUP_SIZE
    min_free_gb: float = 10.0
    heartbeat_seconds: float = 600.0
    # Visibility of a repo this run creates; an existing repo keeps its own.
    private: bool = True
    # Stop starting shards after this many failures in a row (0 disables).
    max_consecutive_failures: int = 3


@dataclass
class _FailureStreak:
    limit: int
    count: int = 0

    def record(self, ok: bool) -> None:
        self.count = 0 if ok else self.count + 1

    @property
    def tripped(self) -> bool:
        return self.limit > 0 and self.count >= self.limit


@dataclass(frozen=True)
class ShardTask:
    task_dir: Path
    font_dir: str
    lang: str
    generation: GenerationOptions
    shard: ShardSpec
    repo_id: Optional[str]
    train_ratio: float
    dry_run: bool = False
    keep_local: bool = False
    row_group_size: int = sp.DEFAULT_ROW_GROUP_SIZE

    @property
    def shard_dir(self) -> Path:
        return self.task_dir / "shards" / self.shard.name

    def fingerprint(self) -> str:
        """Identifies the inputs that determine this shard's content, to validate staged leftovers."""
        payload = json.dumps(
            {
                "generation": self.generation.to_dict(),
                "lang": self.lang,
                "train_ratio": self.train_ratio,
                "start_index": self.shard.start_index,
                "num_images": self.shard.num_images,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_shard_task(task: ShardTask, api: Any = None) -> dict[str, Any]:
    """Produce and (unless dry-run) publish one shard; returns its summary. Runs in a worker process."""
    shard_dir = task.shard_dir
    fingerprint = task.fingerprint()
    summary = sp.read_summary(shard_dir)

    if sp.has_intact_stage(shard_dir, summary, fingerprint):
        logger.info("%s: reusing parquet staged by a previous attempt", task.shard.name)
    else:
        summary = _generate_and_stage(task, fingerprint)

    if task.dry_run:
        return summary

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    started = time.monotonic()
    commit = sp.upload_shard(
        api,
        task.repo_id,
        sp.staged_files_from_summary(shard_dir, summary),
        shard_index=task.shard.index,
        rows=summary["rows"],
    )
    summary.update(status="uploaded", commit=commit, upload_seconds=round(time.monotonic() - started, 1))
    sp.write_summary(shard_dir, summary)
    if not task.keep_local:
        sp.cleanup_shard_dir(shard_dir)
    write_shard_success_marker(shard_dir, summary["rows"])
    return summary


def _generate_and_stage(task: ShardTask, fingerprint: str) -> dict[str, Any]:
    shard, shard_dir = task.shard, task.shard_dir
    shutil.rmtree(shard_dir, ignore_errors=True)

    started = time.monotonic()
    generator = MarkdownDatasetGenerator(output_dir=str(shard_dir), font_dir=task.font_dir, lang=task.lang)
    generated = generator.run(
        num_images=shard.num_images,
        options=task.generation,
        sample_start_index=shard.start_index,
        skip_failed_samples=max(5, shard.num_images // 100),
    )
    if not generated:
        raise RuntimeError(f"Failed to generate {shard.name} (see the log above for the sample error)")
    generate_seconds = time.monotonic() - started

    summary = sp.stage_shard(
        shard_dir=shard_dir,
        features_path=task.task_dir / sp.FEATURES_NAME,
        shard_index=shard.index,
        train_ratio=task.train_ratio,
        expected_rows=shard.num_images - len(generator.skipped_samples),
        row_group_size=task.row_group_size,
    )
    summary.update(
        fingerprint=fingerprint,
        skipped_samples=list(generator.skipped_samples),
        generate_seconds=round(generate_seconds, 1),
    )
    sp.write_summary(shard_dir, summary)
    return summary


def publish_shards_streaming(
    *,
    manifest: RunManifest,
    shard_specs: list[ShardSpec],
    total_shards: int,
    context: GenerationTaskContext,
    task_dir: Path,
    font_dir: str,
    options: StreamUploadOptions,
    resume: bool,
    api: Any = None,
    task_runner: Callable[..., dict[str, Any]] = run_shard_task,
) -> list[str]:
    """Run every unpublished shard through ``task_runner``; return the names of shards that failed."""
    repo_id = context.publish.repo_id
    published: set[int] = set()
    if not options.dry_run:
        if not repo_id:
            raise ValueError("repo_id is required to upload shards")
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        published = _prepare_repo(api, repo_id, resume=resume, private=options.private)

    for shard in shard_specs:
        if shard.index in published and not manifest.is_completed(shard):
            manifest.mark_completed(shard, output_dir=f"hf://datasets/{repo_id}", generated_count=shard.num_images)
    todo = [shard for shard in shard_specs if shard.index not in published]
    logger.info(
        "%d of %d shards to run (%d already on the Hub), %d worker(s)%s",
        len(todo),
        len(shard_specs),
        len(shard_specs) - len(todo),
        options.workers,
        ", dry run: nothing is uploaded or deleted" if options.dry_run else "",
    )

    progress = _Progress(total=len(todo))
    failed: list[str] = []
    streak = _FailureStreak(limit=options.max_consecutive_failures)

    def make_task(shard: ShardSpec) -> ShardTask:
        return ShardTask(
            task_dir=task_dir,
            font_dir=font_dir,
            lang=context.lang,
            generation=context.generation,
            shard=shard,
            repo_id=repo_id,
            train_ratio=context.publish.train_ratio,
            dry_run=options.dry_run,
            keep_local=options.keep_local,
            row_group_size=options.row_group_size,
        )

    def record_success(shard: ShardSpec, summary: dict[str, Any]) -> None:
        manifest.mark_completed(
            shard,
            output_dir=str(task_dir / "shards" / shard.name),
            generated_count=summary["rows"],
            extra={
                "uploaded": not options.dry_run,
                "commit": summary.get("commit"),
                "skipped_samples": summary.get("skipped_samples", []),
            },
        )
        progress.add(summary["rows"])
        streak.record(ok=True)
        logger.info(
            "%s %s: %s rows in %.0fs generate + %.0fs upload | %s",
            shard.name,
            "staged (dry run)" if options.dry_run else "uploaded",
            f"{summary['rows']:,}",
            summary.get("generate_seconds", 0.0),
            summary.get("upload_seconds", 0.0),
            progress.describe(),
        )

    def record_failure(shard: ShardSpec, error: BaseException) -> None:
        manifest.mark_failed(shard, f"{type(error).__name__}: {error}")
        failed.append(shard.name)
        streak.record(ok=False)
        logger.error("%s failed: %s: %s", shard.name, type(error).__name__, error)

    def start(shard: ShardSpec) -> ShardTask:
        _wait_for_free_space(task_dir, options.min_free_gb)
        manifest.mark_started(shard)
        return make_task(shard)

    if options.workers <= 1:
        for shard in todo:
            if streak.tripped:
                break
            task = start(shard)
            try:
                summary = task_runner(task, api=api)
            except Exception as exc:
                record_failure(shard, exc)
            else:
                record_success(shard, summary)
    else:
        _run_in_pool(
            todo=todo,
            options=options,
            task_dir=task_dir,
            progress=progress,
            streak=streak,
            start=start,
            task_runner=task_runner,
            record_success=record_success,
            record_failure=record_failure,
        )

    if streak.tripped:
        not_started = [s.name for s in todo if manifest.data["shards"][s.name]["status"] == "pending"]
        logger.error(
            "Stopped after %d consecutive shard failures; %d shard(s) were not started. "
            "Fix the cause (network, Hub quota, credentials) and rerun with --resume.",
            streak.count,
            len(not_started),
        )
    if not failed and len(shard_specs) == total_shards:
        _publish_card(api=api, context=context, task_dir=task_dir, shard_specs=shard_specs, options=options)
    return failed


def _prepare_repo(api: Any, repo_id: str, *, resume: bool, private: bool) -> set[int]:
    if not api.repo_exists(repo_id, repo_type="dataset"):
        api.create_repo(repo_id, repo_type="dataset", private=private)
        logger.info("Created %s dataset repo %s", "private" if private else "public", repo_id)
    published, foreign = sp.classify_repo_data_files(api.list_repo_files(repo_id, repo_type="dataset"))
    if foreign:
        raise ValueError(
            f"{repo_id} already has parquet files that do not follow the shard naming "
            f"(e.g. {foreign[:3]}); use an empty dataset repo"
        )
    if published and not resume:
        raise ValueError(
            f"{repo_id} already holds {len(published)} published shards; pass --resume to continue that run"
        )
    return published


def _run_in_pool(
    *,
    todo: list[ShardSpec],
    options: StreamUploadOptions,
    task_dir: Path,
    progress: "_Progress",
    streak: _FailureStreak,
    start: Callable[[ShardSpec], ShardTask],
    task_runner: Callable[..., dict[str, Any]],
    record_success: Callable[[ShardSpec, dict[str, Any]], None],
    record_failure: Callable[[ShardSpec, BaseException], None],
) -> None:
    queue = deque(todo)
    in_flight: dict[concurrent.futures.Future, ShardSpec] = {}
    spawn = multiprocessing.get_context("spawn")

    with _quiet_worker_env(), concurrent.futures.ProcessPoolExecutor(
        max_workers=options.workers,
        mp_context=spawn,
        initializer=_init_worker,
    ) as pool:
        try:
            broken = False
            while (queue or in_flight) and not broken:
                while queue and len(in_flight) < options.workers and not streak.tripped:
                    shard = queue.popleft()
                    in_flight[pool.submit(task_runner, start(shard))] = shard
                if not in_flight:
                    break  # the failure streak stopped new shards and everything running has finished

                done, _ = concurrent.futures.wait(
                    in_flight,
                    timeout=options.heartbeat_seconds,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    _log_heartbeat(list(in_flight.values()), task_dir, progress)
                    continue
                for future in done:
                    shard = in_flight.pop(future)
                    error = future.exception()
                    if error is None:
                        record_success(shard, future.result())
                        continue
                    record_failure(shard, error)
                    broken = broken or isinstance(error, BrokenProcessPool)

            if broken:
                for future, shard in in_flight.items():
                    record_failure(shard, BrokenProcessPool("worker pool broke"))
                logger.error(
                    "A worker process died (out of memory?). Not starting the %d shards still queued; "
                    "rerun with --resume.",
                    len(queue),
                )
        except KeyboardInterrupt:
            for process in list(getattr(pool, "_processes", {}).values()):
                process.terminate()
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def _publish_card(
    *,
    api: Any,
    context: GenerationTaskContext,
    task_dir: Path,
    shard_specs: list[ShardSpec],
    options: StreamUploadOptions,
) -> None:
    generated, split_counts, distribution = sp.aggregate_summaries(task_dir, [s.name for s in shard_specs])
    repo_id = context.publish.repo_id or "<your-username>/<dataset-name>"
    content = build_dataset_readme(
        repo_id=repo_id,
        context=context,
        generated_count=generated,
        split_counts=split_counts,
        distribution_summary=distribution,
    )
    card_path = task_dir / "DATASET_CARD.md"
    card_path.write_text(content, encoding="utf-8")
    logger.info("Dataset card written to %s (%s rows, splits %s)", card_path, f"{generated:,}", split_counts)
    if not options.dry_run:
        upload_dataset_readme_to_hub(
            repo_id=repo_id,
            readme_content=content,
            commit_message="docs: add generation metadata dataset card",
        )


def _init_worker() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s - %(message)s",
    )


@contextlib.contextmanager
def _quiet_worker_env():
    """Silence tqdm and Hub progress bars in worker processes, which inherit the parent's environment."""
    names = ("TQDM_DISABLE", "HF_HUB_DISABLE_PROGRESS_BARS")
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _wait_for_free_space(
    path: Path,
    min_free_gb: float,
    *,
    poll_seconds: float = 60.0,
    max_wait_seconds: float = 3600.0,
    sleep: Optional[Callable[[float], None]] = None,
) -> None:
    waited = 0.0
    while True:
        free_gb = shutil.disk_usage(path).free / 1e9
        if free_gb >= min_free_gb:
            return
        if waited >= max_wait_seconds:
            raise OSError(f"Only {free_gb:.1f} GB free under {path} after waiting {waited / 60:.0f} min")
        logger.warning("Only %.1f GB free under %s (need %.1f); waiting to start the next shard", free_gb, path, min_free_gb)
        (sleep or time.sleep)(poll_seconds)
        waited += poll_seconds


def _count_generated_images(shard_dir: Path) -> int:
    try:
        return sum(1 for entry in os.scandir(shard_dir / "markdown") if entry.name.endswith(".png"))
    except FileNotFoundError:
        return 0


def _log_heartbeat(running: list[ShardSpec], task_dir: Path, progress: "_Progress") -> None:
    parts = [
        f"{shard.name} {_count_generated_images(task_dir / 'shards' / shard.name):,}/{shard.num_images:,}"
        for shard in running
    ]
    logger.info("Heartbeat, running: %s | %s", ", ".join(parts), progress.describe())


def _format_duration(seconds: float) -> str:
    if seconds != seconds:  # NaN before the first shard finishes
        return "n/a"
    hours, remainder = divmod(int(seconds), 3600)
    return f"{hours}h{remainder // 60:02d}m"


@dataclass
class _Progress:
    total: int
    started: float = field(default_factory=time.monotonic)
    done: int = 0
    rows: int = 0

    def add(self, rows: int) -> None:
        self.done += 1
        self.rows += rows

    def describe(self) -> str:
        elapsed = time.monotonic() - self.started
        eta = elapsed / self.done * (self.total - self.done) if self.done else float("nan")
        return (
            f"{self.done}/{self.total} shards this session, {self.rows:,} rows, "
            f"elapsed {_format_duration(elapsed)}, ETA {_format_duration(eta)}"
        )
