import logging
from pathlib import Path
from typing import Any, Optional

from env_utils import set_global_seed
from generation.hub_upload import upload_generated_dataset
from generation.mixed import MixedGenerator, build_generation_kwargs
from generation.sharding import (
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


def _build_generation_kwargs(
    *,
    template: Optional[str],
    template_family: Optional[str],
    min_template_complexity: Optional[int],
    max_template_complexity: Optional[int],
    template_config_dir: Optional[str],
    markdown_renderer: str,
    style_profile: str,
    coverage_targets: Any,
    novelty_window: int,
    novelty_threshold: float,
    novelty_max_attempts: int,
    similar_char_ratio: float,
    similarity_db_path: Optional[str],
    formula_source_mode: str,
    formula_dataset_path: Optional[str],
    formula_dataset_weight: float,
    formula_random_weight: float,
    formula_synthetic_weight: float,
    seed: Optional[int],
    add_noise: Optional[bool],
    add_blur: Optional[bool],
    sample_start_index: int = 0,
) -> dict[str, Any]:
    return build_generation_kwargs(
        template=template,
        template_family=template_family,
        min_template_complexity=min_template_complexity,
        max_template_complexity=max_template_complexity,
        template_config_dir=template_config_dir,
        markdown_renderer=markdown_renderer,
        style_profile=style_profile,
        coverage_targets=coverage_targets,
        novelty_window=novelty_window,
        novelty_threshold=novelty_threshold,
        novelty_max_attempts=novelty_max_attempts,
        similar_char_ratio=similar_char_ratio,
        similarity_db_path=similarity_db_path,
        formula_source_mode=formula_source_mode,
        formula_dataset_path=formula_dataset_path,
        formula_dataset_weight=formula_dataset_weight,
        formula_random_weight=formula_random_weight,
        formula_synthetic_weight=formula_synthetic_weight,
        seed=seed,
        add_noise=add_noise,
        add_blur=add_blur,
        sample_start_index=sample_start_index,
    )


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
    mixed: bool,
    lang: str,
    seed: Optional[int],
    repo_id: Optional[str],
    generation_config: dict[str, Any],
) -> RunManifest:
    manifest_path = task_output_dir / "run_manifest.json"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = ensure_resume_state(task_output_dir, manifest_path, resume=resume)
    if existing_manifest is not None:
        existing_manifest.validate_or_raise(
            size=size,
            shard_size=shard_size,
            mixed=mixed,
            lang=lang,
            seed=seed,
            repo_id=repo_id,
        )
        return existing_manifest

    return RunManifest.create(
        path=manifest_path,
        generator_name=generator_name,
        size=size,
        shard_size=shard_size,
        mixed=mixed,
        lang=lang,
        seed=seed,
        repo_id=repo_id,
        generation_config=generation_config,
    )


def _build_publish_context(
    *,
    lang: str,
    size: int,
    template: Optional[str],
    template_family: Optional[str],
    min_template_complexity: Optional[int],
    max_template_complexity: Optional[int],
    template_config_dir: Optional[str],
    markdown_renderer: str,
    style_profile: str,
    coverage_targets: Any,
    novelty_window: int,
    novelty_threshold: float,
    novelty_max_attempts: int,
    similar_char_ratio: float,
    similarity_db_path: Optional[str],
    formula_source_mode: str,
    formula_dataset_path: Optional[str],
    formula_dataset_weight: float,
    formula_random_weight: float,
    formula_synthetic_weight: float,
    add_noise: Optional[bool],
    add_blur: Optional[bool],
    mixed: bool,
    train_ratio: float,
    test_ratio: float,
    seed: Optional[int],
) -> dict[str, Any]:
    return {
        "lang": lang,
        "size": size,
        "template": template,
        "template_family": template_family,
        "min_template_complexity": min_template_complexity,
        "max_template_complexity": max_template_complexity,
        "template_config_dir": template_config_dir,
        "markdown_renderer": markdown_renderer,
        "style_profile": style_profile,
        "coverage_targets": coverage_targets,
        "novelty_window": novelty_window,
        "novelty_threshold": novelty_threshold,
        "novelty_max_attempts": novelty_max_attempts,
        "similar_char_ratio": similar_char_ratio,
        "similarity_db_path": similarity_db_path,
        "formula_source_mode": formula_source_mode,
        "formula_dataset_path": formula_dataset_path,
        "formula_dataset_weight": formula_dataset_weight,
        "formula_random_weight": formula_random_weight,
        "formula_synthetic_weight": formula_synthetic_weight,
        "add_noise": add_noise,
        "add_blur": add_blur,
        "mixed": mixed,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    }


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
    generation_config = dict(manifest.data.get("generation_config") or {})
    resolved_repo_id = repo_id or manifest.data.get("repo_id")
    if not resolved_repo_id:
        raise ValueError("repo_id is required to publish the generated dataset")

    resolved_train_ratio = train_ratio if train_ratio is not None else generation_config.get("train_ratio", 0.9)
    resolved_test_ratio = test_ratio if test_ratio is not None else generation_config.get("test_ratio", 0.1)

    return upload_generated_dataset(
        repo_id=resolved_repo_id,
        generated_path=generated_dir,
        mixed=bool(manifest.data.get("mixed", generation_config.get("mixed", False))),
        train_ratio=float(resolved_train_ratio),
        test_ratio=float(resolved_test_ratio),
        lang=str(manifest.data.get("lang", generation_config.get("lang", "ko"))),
        size=int(manifest.data.get("size", generation_config.get("size", 0))),
        template=generation_config.get("template"),
        template_family=generation_config.get("template_family"),
        min_template_complexity=generation_config.get("min_template_complexity"),
        max_template_complexity=generation_config.get("max_template_complexity"),
        template_config_dir=generation_config.get("template_config_dir"),
        markdown_renderer=str(generation_config.get("markdown_renderer", "pil")),
        style_profile=str(generation_config.get("style_profile", "balanced")),
        coverage_targets=generation_config.get("coverage_targets"),
        novelty_window=int(generation_config.get("novelty_window", 80)),
        novelty_threshold=float(generation_config.get("novelty_threshold", 0.95)),
        novelty_max_attempts=int(generation_config.get("novelty_max_attempts", 4)),
        similar_char_ratio=float(generation_config.get("similar_char_ratio", 0.08)),
        similarity_db_path=generation_config.get("similarity_db_path"),
        formula_source_mode=str(generation_config.get("formula_source_mode", "mixed")),
        formula_dataset_path=generation_config.get("formula_dataset_path"),
        formula_dataset_weight=float(generation_config.get("formula_dataset_weight", 0.45)),
        formula_random_weight=float(generation_config.get("formula_random_weight", 0.30)),
        formula_synthetic_weight=float(generation_config.get("formula_synthetic_weight", 0.25)),
        add_noise=generation_config.get("add_noise"),
        add_blur=generation_config.get("add_blur"),
        seed=manifest.data.get("seed", generation_config.get("seed")),
    )


def pipeline(
    repo_id: Optional[str],
    size: int,
    output_dir: str,
    lang: str,
    template: Optional[str] = None,
    template_family: Optional[str] = None,
    min_template_complexity: Optional[int] = None,
    max_template_complexity: Optional[int] = None,
    template_config_dir: Optional[str] = None,
    markdown_renderer: str = "pil",
    style_profile: str = "balanced",
    coverage_targets: Any = None,
    novelty_window: int = 80,
    novelty_threshold: float = 0.95,
    novelty_max_attempts: int = 4,
    similar_char_ratio: float = 0.08,
    similarity_db_path: Optional[str] = None,
    formula_source_mode: str = "mixed",
    formula_dataset_path: Optional[str] = None,
    formula_dataset_weight: float = 0.45,
    formula_random_weight: float = 0.30,
    formula_synthetic_weight: float = 0.25,
    add_noise: Optional[bool] = None,
    add_blur: Optional[bool] = None,
    mixed: bool = False,
    train_ratio: float = 0.9,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
    shard_size: Optional[int] = None,
    max_shards: Optional[int] = None,
    resume: bool = False,
    upload: bool = False,
) -> None:
    logger.info("=" * 80)
    set_global_seed(seed)
    if mixed:
        logger.info(" Synthetic OCR Dataset Generator (Train/Test Split) ".center(80))
    else:
        logger.info(" Synthetic OCR Dataset Generator ".center(80))
        logger.info(" Format: markdown ".center(80))
    logger.info("=" * 80)

    if size <= 0:
        logger.warning("Requested number of images is 0, terminating.")
        return

    if not (0.0 <= train_ratio <= 1.0 and 0.0 <= test_ratio <= 1.0):
        raise ValueError("train_ratio and test_ratio must be between 0.0 and 1.0")

    if abs((train_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio and test_ratio must sum to 1.0")

    base_dir = Path(output_dir) / lang
    font_dir = Path(f"fonts/{lang}")
    resolved_shard_size = _resolve_shard_size(size, shard_size)
    shard_specs = plan_shards(size, resolved_shard_size, max_shards=max_shards)
    publish_context = _build_publish_context(
        lang=lang,
        size=size,
        template=template,
        template_family=template_family,
        min_template_complexity=min_template_complexity,
        max_template_complexity=max_template_complexity,
        template_config_dir=template_config_dir,
        markdown_renderer=markdown_renderer,
        style_profile=style_profile,
        coverage_targets=coverage_targets,
        novelty_window=novelty_window,
        novelty_threshold=novelty_threshold,
        novelty_max_attempts=novelty_max_attempts,
        similar_char_ratio=similar_char_ratio,
        similarity_db_path=similarity_db_path,
        formula_source_mode=formula_source_mode,
        formula_dataset_path=formula_dataset_path,
        formula_dataset_weight=formula_dataset_weight,
        formula_random_weight=formula_random_weight,
        formula_synthetic_weight=formula_synthetic_weight,
        add_noise=add_noise,
        add_blur=add_blur,
        mixed=mixed,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    if not shard_specs:
        logger.warning("No shard work was planned, terminating.")
        return

    if mixed:
        task_output_dir = base_dir / "images_mixed"
        manifest = _prepare_run_manifest(
            task_output_dir=task_output_dir,
            resume=resume,
            generator_name="mixed",
            size=size,
            shard_size=resolved_shard_size,
            mixed=True,
            lang=lang,
            seed=seed,
            repo_id=repo_id,
            generation_config=publish_context,
        )
        manifest.initialize_shards(shard_specs)

        for shard in shard_specs:
            shard_dir = task_output_dir / "shards" / shard.name
            if resume and manifest.is_completed(shard) and shard_success_marker_exists(shard_dir):
                logger.info("Skipping completed shard %s", shard.name)
                continue

            mixed_gen = MixedGenerator(
                output_dir=str(shard_dir),
                font_dir=str(font_dir),
                lang=lang,
            )
            manifest.mark_started(shard)
            generated_dir = mixed_gen.run(
                num_images=shard.num_images,
                template=template,
                template_family=template_family,
                min_template_complexity=min_template_complexity,
                max_template_complexity=max_template_complexity,
                template_config_dir=template_config_dir,
                markdown_renderer=markdown_renderer,
                style_profile=style_profile,
                coverage_targets=coverage_targets,
                novelty_window=novelty_window,
                novelty_threshold=novelty_threshold,
                novelty_max_attempts=novelty_max_attempts,
                similar_char_ratio=similar_char_ratio,
                similarity_db_path=similarity_db_path,
                formula_source_mode=formula_source_mode,
                formula_dataset_path=formula_dataset_path,
                formula_dataset_weight=formula_dataset_weight,
                formula_random_weight=formula_random_weight,
                formula_synthetic_weight=formula_synthetic_weight,
                add_noise=add_noise,
                add_blur=add_blur,
                seed=seed,
                sample_start_index=shard.start_index,
            )
            if not generated_dir:
                manifest.mark_failed(shard, "mixed shard generation failed")
                raise RuntimeError(f"Failed to generate shard {shard.name}")

            write_shard_success_marker(shard_dir, shard.num_images)
            manifest.mark_completed(shard, generated_dir, shard.num_images)

        rebuild_aggregate_outputs(output_dir=task_output_dir, shards=shard_specs, format_name="markdown")
        manifest.mark_finished()
        generated_dir = str(task_output_dir)
    else:
        from generator import Generator

        task_output_dir = base_dir / "images_markdown"
        manifest = _prepare_run_manifest(
            task_output_dir=task_output_dir,
            resume=resume,
            generator_name="markdown",
            size=size,
            shard_size=resolved_shard_size,
            mixed=False,
            lang=lang,
            seed=seed,
            repo_id=repo_id,
            generation_config=publish_context,
        )
        manifest.initialize_shards(shard_specs)

        for shard in shard_specs:
            shard_dir = task_output_dir / "shards" / shard.name
            if resume and manifest.is_completed(shard) and shard_success_marker_exists(shard_dir):
                logger.info("Skipping completed shard %s", shard.name)
                continue

            generator = Generator(
                output_dir=str(shard_dir),
                font_dir=str(font_dir),
                lang=lang,
            )
            generation_kwargs = _build_generation_kwargs(
                template=template,
                template_family=template_family,
                min_template_complexity=min_template_complexity,
                max_template_complexity=max_template_complexity,
                template_config_dir=template_config_dir,
                markdown_renderer=markdown_renderer,
                style_profile=style_profile,
                coverage_targets=coverage_targets,
                novelty_window=novelty_window,
                novelty_threshold=novelty_threshold,
                novelty_max_attempts=novelty_max_attempts,
                similar_char_ratio=similar_char_ratio,
                similarity_db_path=similarity_db_path,
                formula_source_mode=formula_source_mode,
                formula_dataset_path=formula_dataset_path,
                formula_dataset_weight=formula_dataset_weight,
                formula_random_weight=formula_random_weight,
                formula_synthetic_weight=formula_synthetic_weight,
                seed=seed,
                add_noise=add_noise,
                add_blur=add_blur,
                sample_start_index=shard.start_index,
            )
            generation_kwargs["num_images"] = shard.num_images
            manifest.mark_started(shard)
            generated_dir = generator.run(**generation_kwargs)
            if not generated_dir:
                manifest.mark_failed(shard, "markdown shard generation failed")
                raise RuntimeError(f"Failed to generate shard {shard.name}")

            write_shard_success_marker(shard_dir, shard.num_images)
            manifest.mark_completed(shard, generated_dir, shard.num_images)

        rebuild_aggregate_outputs(output_dir=task_output_dir, shards=shard_specs)
        manifest.mark_finished()
        generated_dir = str(task_output_dir)

    if generated_dir and upload:
        if not repo_id:
            raise ValueError("repo_id is required when upload is enabled")
        logger.info(f"\n--- Uploading to Hugging Face Hub: {repo_id} ---")
        upload_generated_dataset(
            repo_id=repo_id,
            generated_path=Path(generated_dir),
            mixed=mixed,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            lang=lang,
            size=size,
            template=template,
            template_family=template_family,
            min_template_complexity=min_template_complexity,
            max_template_complexity=max_template_complexity,
            template_config_dir=template_config_dir,
            markdown_renderer=markdown_renderer,
            style_profile=style_profile,
            coverage_targets=coverage_targets,
            novelty_window=novelty_window,
            novelty_threshold=novelty_threshold,
            novelty_max_attempts=novelty_max_attempts,
            similar_char_ratio=similar_char_ratio,
            similarity_db_path=similarity_db_path,
            formula_source_mode=formula_source_mode,
            formula_dataset_path=formula_dataset_path,
            formula_dataset_weight=formula_dataset_weight,
            formula_random_weight=formula_random_weight,
            formula_synthetic_weight=formula_synthetic_weight,
            add_noise=add_noise,
            add_blur=add_blur,
            seed=seed,
        )
    elif not generated_dir:
        logger.warning("No dataset was generated, skipping upload.")

    logger.info("\n" + " Pipeline completed! ".center(80, "="))
    if upload and repo_id:
        logger.info(f"Dataset: https://huggingface.co/datasets/{repo_id}")
    else:
        logger.info(f"Generated dataset path: {generated_dir}")
