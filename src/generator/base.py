"""Base generator class for synthetic OCR image generation."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

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
    def generate(self, num_images: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Generate synthetic images.

        Args:
            num_images: Number of images to generate.
            **kwargs: Additional generator-specific parameters.

        Returns:
            List of metadata dictionaries for each generated image.
        """
        pass

    @abstractmethod
    def generate_single(self, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generate a single synthetic image.

        Args:
            **kwargs: Additional generator-specific parameters.

        Returns:
            Tuple of (generated image, metadata dictionary).
        """
        pass

    def save_metadata(self, metadata: List[Dict[str, Any]]) -> Path:
        """
        Save metadata to JSONL file.

        Args:
            metadata: List of metadata dictionaries.

        Returns:
            Path to the saved metadata file.
        """
        metadata_path = self.output_dir / "metadata.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for item in metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Saved metadata to '{metadata_path}'")
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
                f"Starting {self.__class__.__name__}: generating {num_images:,} images"
            )
            metadata = self.generate(num_images, **kwargs)
            self.save_metadata(metadata)
            logger.info(f"Successfully generated {len(metadata):,} images")
            return str(self.output_dir)
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return None
