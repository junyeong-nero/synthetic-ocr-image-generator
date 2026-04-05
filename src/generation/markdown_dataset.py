import json
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.generation.ground_truth import attach_unified_ground_truth
from src.generation.options import GenerationOptions
from src.generator.realism_stats import RealismStatsAccumulator, write_realism_stats

logger = logging.getLogger(__name__)
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
            from src.generator import Generator

            self._markdown_generator = Generator(
                output_dir=str(self.output_dir / "markdown"),
                font_dir=str(self.font_dir),
                lang=self.lang,
            )
        return self._markdown_generator

    def run(
        self,
        num_images: int,
        options: GenerationOptions,
        sample_start_index: int = 0,
    ) -> Optional[str]:
        try:
            logger.info(f"Starting markdown generation: {num_images:,} images")

            generation_kwargs = options.to_generator_kwargs(
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
