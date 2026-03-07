import json
import logging
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from generation.ground_truth import attach_unified_ground_truth
from generator.realism_stats import RealismStatsAccumulator, write_realism_stats

logger = logging.getLogger(__name__)


def build_generation_kwargs(
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
    generation_kwargs: dict[str, Any] = {
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
        "seed": seed,
        "sample_start_index": sample_start_index,
    }
    if add_noise is not None:
        generation_kwargs["add_noise"] = add_noise
    if add_blur is not None:
        generation_kwargs["add_blur"] = add_blur
    return generation_kwargs


class MarkdownDatasetGenerator:
    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_dir = Path(font_dir)
        self.lang = lang
        self._markdown_generator = None

    @property
    def markdown_generator(self):
        if self._markdown_generator is None:
            from generator import Generator

            self._markdown_generator = Generator(
                output_dir=str(self.output_dir / "markdown"),
                font_dir=str(self.font_dir),
                lang=self.lang,
            )
        return self._markdown_generator

    def run(
        self,
        num_images: int,
        template: Optional[str] = None,
        template_family: Optional[str] = None,
        min_template_complexity: Optional[int] = None,
        max_template_complexity: Optional[int] = None,
        template_config_dir: Optional[str] = None,
        markdown_renderer: str = "playwright",
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
        seed: Optional[int] = None,
        sample_start_index: int = 0,
    ) -> Optional[str]:
        try:
            logger.info(f"Starting markdown generation: {num_images:,} images")

            generation_kwargs = build_generation_kwargs(
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

            self.markdown_generator._configure_generation(**generation_kwargs)

            metadata_path = self.output_dir / "metadata.jsonl"
            stats_accumulator = RealismStatsAccumulator(format_name="markdown")
            generated_count = 0
            sample_start_index = int(generation_kwargs.pop("sample_start_index", 0))

            with open(metadata_path, "w", encoding="utf-8") as metadata_handle:
                for idx in tqdm(range(num_images), desc="Generating markdown images"):
                    sample_index = sample_start_index + idx
                    image, meta = self.markdown_generator.generate_single(sample_index=sample_index)
                    filename = f"markdown_{sample_index:05d}.png"
                    self.markdown_generator.save_image(image, filename)
                    meta["file_name"] = str(self.markdown_generator.output_dir / filename)
                    meta = attach_unified_ground_truth("markdown", meta)
                    metadata_handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    stats_accumulator.update(meta)
                    generated_count += 1

            logger.info(f"Saved metadata to '{metadata_path}'")
            write_realism_stats(self.output_dir, stats_accumulator)
            logger.info(f"Successfully generated {generated_count:,} markdown images")
            return str(self.output_dir)

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return None
