import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.generation.ground_truth import attach_unified_ground_truth
from src.generation.options import GenerationOptions
from src.generator.realism_stats import RealismStatsAccumulator, write_realism_stats

logger = logging.getLogger(__name__)

# Rendering can fail transiently (browser crash, timeout), and over a long run it
# will. Retrying reuses the same sample index (hence the same seed), so a
# recovered sample is identical.
SAMPLE_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


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
        self.skipped_samples: list[int] = []

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

    @staticmethod
    def _renderer_session(options: GenerationOptions):
        """Keep one headless Chromium open for the whole run instead of one per sample."""
        if options.markdown_renderer != "playwright":
            return contextlib.nullcontext()
        from src.generator.markdown_renderers import playwright_session

        return playwright_session()

    def _generate_with_retry(self, sample_index: int):
        for attempt in range(1, SAMPLE_ATTEMPTS + 1):
            try:
                return self.markdown_generator.generate_single(sample_index=sample_index)
            except Exception as exc:
                if attempt == SAMPLE_ATTEMPTS:
                    raise
                logger.warning(
                    "Sample %d failed (attempt %d/%d): %s; retrying",
                    sample_index,
                    attempt,
                    SAMPLE_ATTEMPTS,
                    exc,
                )
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    def run(
        self,
        num_images: int,
        options: GenerationOptions,
        sample_start_index: int = 0,
        skip_failed_samples: int = 0,
    ) -> Optional[str]:
        """Generate ``num_images`` samples into ``output_dir``.

        A sample that still fails after retries aborts the run, unless fewer than
        ``skip_failed_samples`` samples were skipped so far; skipped indices are
        collected in ``self.skipped_samples`` and leave a gap in the metadata.
        """
        try:
            logger.info(f"Starting markdown generation: {num_images:,} images")

            generation_kwargs = options.to_generator_kwargs(
                sample_start_index=sample_start_index,
            )

            self.markdown_generator._configure_generation(**generation_kwargs)

            metadata_path = self.output_dir / "metadata.jsonl"
            stats_accumulator = RealismStatsAccumulator(format_name="markdown")
            generated_count = 0
            self.skipped_samples = []
            sample_start_index = int(generation_kwargs.pop("sample_start_index", 0))

            with self._renderer_session(options), open(metadata_path, "w", encoding="utf-8") as metadata_handle:
                for idx in tqdm(range(num_images), desc="Generating markdown images"):
                    sample_index = sample_start_index + idx
                    try:
                        image, meta = self._generate_with_retry(sample_index)
                    except Exception as exc:
                        if len(self.skipped_samples) >= skip_failed_samples:
                            raise
                        self.skipped_samples.append(sample_index)
                        logger.error(
                            "Skipping sample %d after %d failed attempts: %s",
                            sample_index,
                            SAMPLE_ATTEMPTS,
                            exc,
                        )
                        continue
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
