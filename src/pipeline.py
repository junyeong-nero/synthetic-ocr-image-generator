import logging
from pathlib import Path
from typing import Any, Optional

from env_utils import set_global_seed
from generation.hub_upload import upload_generated_dataset
from generation.mixed import MixedGenerator, build_generation_kwargs

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
    )


def pipeline(
    repo_id: str,
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

    if mixed:
        task_output_dir = base_dir / "images_mixed"
        mixed_gen = MixedGenerator(
            output_dir=str(task_output_dir),
            font_dir=str(font_dir),
            lang=lang,
        )

        generated_dir = mixed_gen.run(
            num_images=size,
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
    else:
        from generator import Generator

        task_output_dir = base_dir / "images_markdown"
        generator = Generator(
            output_dir=str(task_output_dir),
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
        )
        generation_kwargs["num_images"] = size
        generated_dir = generator.run(**generation_kwargs)

    if generated_dir:
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
    else:
        logger.warning("No dataset was generated, skipping upload.")

    logger.info("\n" + " Pipeline completed! ".center(80, "="))
    logger.info(f"Dataset: https://huggingface.co/datasets/{repo_id}")
