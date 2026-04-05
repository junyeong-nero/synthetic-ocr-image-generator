import logging
from pathlib import Path
from typing import Optional

from src.env_utils import set_global_seed
from src.generation.hub_upload import upload_generated_dataset
from src.generation.markdown_dataset import MarkdownDatasetGenerator
from src.generation.options import GenerationTaskContext
from src.generation.sharding import (
    RunManifest,
    ensure_resume_state,
    plan_shards,
    rebuild_aggregate_outputs,
    shard_success_marker_exists,
    write_shard_success_marker,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _resolve_shard_size(size: int, shard_size: Optional[int]) -> int:
    if shard_size is None or shard_size <= 0:
        return size
    return shard_size


def _prepare_run_manifest(
    *,
    task_output_dir: Path,
    resume: bool,
    generator_name: str,
    size: int,
    shard_size: int,
    context: GenerationTaskContext,
) -> RunManifest:
    manifest_path = task_output_dir / "run_manifest.json"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = ensure_resume_state(task_output_dir, manifest_path, resume=resume)
    if existing_manifest is not None:
        existing_manifest.validate_or_raise(
            size=size,
            shard_size=shard_size,
            lang=context.lang,
            seed=context.generation.seed,
            repo_id=context.publish.repo_id,
        )
        return existing_manifest

    return RunManifest.create(
        path=manifest_path,
        generator_name=generator_name,
        size=size,
        shard_size=shard_size,
        lang=context.lang,
        seed=context.generation.seed,
        repo_id=context.publish.repo_id,
        task_context=context.to_dict(),
    )


def publish_pipeline(
    *,
    generated_path: str,
    repo_id: Optional[str] = None,
    train_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
) -> dict[str, int]:
    generated_dir = Path(generated_path)
    manifest_path = generated_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found at '{manifest_path}'")

    manifest = RunManifest.load(manifest_path)
    context = GenerationTaskContext.from_manifest_data(manifest.data)
    context = context.with_publish_overrides(
        repo_id=repo_id,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
    )
    resolved_repo_id = context.publish.repo_id
    if not resolved_repo_id:
        raise ValueError("repo_id is required to publish the generated dataset")

    return upload_generated_dataset(
        generated_path=generated_dir,
        context=context,
    )


def pipeline(
    context: GenerationTaskContext,
    output_dir: str,
    shard_size: Optional[int] = None,
    max_shards: Optional[int] = None,
    resume: bool = False,
    upload: bool = False,
) -> None:
    logger.info("=" * 80)
    set_global_seed(context.generation.seed)
    logger.info(" Synthetic OCR Dataset Generator (Train/Test Split) ".center(80))
    logger.info("=" * 80)

    if context.size <= 0:
        logger.warning("Requested number of images is 0, terminating.")
        return

    if not (
        0.0 <= context.publish.train_ratio <= 1.0
        and 0.0 <= context.publish.test_ratio <= 1.0
    ):
        raise ValueError("train_ratio and test_ratio must be between 0.0 and 1.0")

    if abs((context.publish.train_ratio + context.publish.test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio and test_ratio must sum to 1.0")

    base_dir = Path(output_dir) / context.lang
    font_dir = Path(f"fonts/{context.lang}")
    resolved_shard_size = _resolve_shard_size(context.size, shard_size)
    shard_specs = plan_shards(context.size, resolved_shard_size, max_shards=max_shards)

    if not shard_specs:
        logger.warning("No shard work was planned, terminating.")
        return

    task_output_dir = base_dir / "images_markdown"
    manifest = _prepare_run_manifest(
        task_output_dir=task_output_dir,
        resume=resume,
        generator_name="markdown",
        size=context.size,
        shard_size=resolved_shard_size,
        context=context,
    )
    manifest.initialize_shards(shard_specs)

    for shard in shard_specs:
        shard_dir = task_output_dir / "shards" / shard.name
        if resume and manifest.is_completed(shard) and shard_success_marker_exists(shard_dir):
            logger.info("Skipping completed shard %s", shard.name)
            continue

        markdown_dataset_generator = MarkdownDatasetGenerator(
            output_dir=str(shard_dir),
            font_dir=str(font_dir),
            lang=context.lang,
        )
        manifest.mark_started(shard)
        generated_dir = markdown_dataset_generator.run(
            num_images=shard.num_images,
            options=context.generation,
            sample_start_index=shard.start_index,
        )
        if not generated_dir:
            manifest.mark_failed(shard, "split-aware shard generation failed")
            raise RuntimeError(f"Failed to generate shard {shard.name}")

        write_shard_success_marker(shard_dir, shard.num_images)
        manifest.mark_completed(shard, generated_dir, shard.num_images)

    rebuild_aggregate_outputs(output_dir=task_output_dir, shards=shard_specs, format_name="markdown")
    manifest.mark_finished()
    generated_dir = str(task_output_dir)

    if generated_dir and upload:
        if not context.publish.repo_id:
            raise ValueError("repo_id is required when upload is enabled")
        logger.info(f"\n--- Uploading to Hugging Face Hub: {context.publish.repo_id} ---")
        upload_generated_dataset(
            generated_path=Path(generated_dir),
            context=context,
        )
    elif not generated_dir:
        logger.warning("No dataset was generated, skipping upload.")

    logger.info("\n" + " Pipeline completed! ".center(80, "="))
    if upload and context.publish.repo_id:
        logger.info(f"Dataset: https://huggingface.co/datasets/{context.publish.repo_id}")
    else:
        logger.info(f"Generated dataset path: {generated_dir}")
