"""Base generator class for synthetic OCR image generation."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

from PIL import Image

from generator.realism_stats import RealismStatsAccumulator, write_realism_stats

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract base class for all image generators."""

    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str = "ko",
    ):
        """
        Initialize the generator.

        Args:
            output_dir: Directory to save generated images and metadata.
            font_dir: Directory containing font files.
            lang: Language code for text generation.
        """
        self.output_dir = Path(output_dir)
        self.font_dir = Path(font_dir)
        self.lang = lang

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.font_paths = self._load_fonts()
        if not self.font_paths:
            raise ValueError(f"No .ttf font files found in '{font_dir}'")

    def _load_fonts(self) -> List[str]:
        """Load all TTF font paths from the font directory."""
        return [str(p) for p in self.font_dir.glob("*.ttf")]

    @abstractmethod
    def generate(self, num_images: int, **kwargs) -> int:
        raise NotImplementedError

    @abstractmethod
    def generate_single(self, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generate a single synthetic image.

        Args:
            **kwargs: Additional generator-specific parameters.

        Returns:
            Tuple of (generated image, metadata dictionary).
        """
        raise NotImplementedError

    def _metadata_path(self) -> Path:
        return self.output_dir / "metadata.jsonl"

    def _create_realism_stats_accumulator(self) -> RealismStatsAccumulator:
        return RealismStatsAccumulator()

    def _open_metadata_writer(self) -> tuple[Path, TextIO, RealismStatsAccumulator]:
        metadata_path = self._metadata_path()
        handle = open(metadata_path, "w", encoding="utf-8", buffering=1)
        accumulator = self._create_realism_stats_accumulator()
        return metadata_path, handle, accumulator

    def append_metadata(
        self,
        handle: TextIO,
        accumulator: RealismStatsAccumulator,
        item: Dict[str, Any],
    ) -> None:
        import json

        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        accumulator.update(item)

    def finalize_metadata(
        self,
        metadata_path: Path,
        handle: TextIO,
        accumulator: RealismStatsAccumulator,
    ) -> Path:
        handle.flush()
        handle.close()
        logger.info("Saved metadata to '%s'", metadata_path)
        write_realism_stats(self.output_dir, accumulator)
        return metadata_path

    def save_image(self, image: Image.Image, filename: str) -> Path:
        """
        Save a single image.

        Args:
            image: PIL Image to save.
            filename: Filename for the image.

        Returns:
            Path to the saved image.
        """
        filepath = self.output_dir / filename
        image.save(filepath)
        return filepath

    def run(self, num_images: int, **kwargs) -> Optional[str]:
        """
        Run the full generation pipeline.

        Args:
            num_images: Number of images to generate.
            **kwargs: Additional generator-specific parameters.

        Returns:
            Path to output directory, or None if generation fails.
        """
        try:
            logger.info(
                "Starting %s: generating %s images",
                self.__class__.__name__,
                f"{num_images:,}",
            )
            metadata_path, metadata_handle, stats_accumulator = self._open_metadata_writer()
            try:
                generated_count = self.generate(
                    num_images,
                    metadata_handle=metadata_handle,
                    stats_accumulator=stats_accumulator,
                    **kwargs,
                )
            finally:
                self.finalize_metadata(metadata_path, metadata_handle, stats_accumulator)

            logger.info("Successfully generated %s images", f"{generated_count:,}")
            return str(self.output_dir)
        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            return None
