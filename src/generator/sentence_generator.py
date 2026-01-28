import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image
from tqdm import tqdm

from generator.base import BaseGenerator
from generator.effects import render_text_with_effects
from character_similarity import generate_sentence_typos
from utils import read_txt, read_json

logger = logging.getLogger(__name__)


class SentenceGenerator(BaseGenerator):

    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        corpus_path: str,
        similarity_db_path: str,
        lang: str = "ko",
    ):
        super().__init__(output_dir, font_dir, lang)
        self.corpus_path = Path(corpus_path)
        self.similarity_db_path = Path(similarity_db_path)

        self._sentences = self._load_corpus()
        self._similarity_db = read_json(str(similarity_db_path))
        if self._similarity_db is None:
             logger.warning(f"Failed to load similarity DB from {similarity_db_path}")
             self._similarity_db = {}

    def _load_corpus(self) -> List[str]:
        content = read_txt(str(self.corpus_path))
        if content is None:
            return []
        lines = content.splitlines()
        return [" ".join(line.split()[:5]).strip() for line in lines if line.strip()]

    def generate(
        self,
        num_images: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        typo_ratio = kwargs.get("typo_ratio", 0.15)
        resolution_range = kwargs.get("resolution_range", (24, 48))
        
        self.typo_ratio = typo_ratio
        self.resolution_range = resolution_range
        
        if self._similarity_db is None:
             raise ValueError("Similarity DB not loaded")
             
        original_typo_pairs = generate_sentence_typos(
            self._sentences, self._similarity_db, typo_ratio=typo_ratio
        )
        self._original_typo_pairs = original_typo_pairs

        metadata = []
        for idx in tqdm(range(num_images), desc="Generating sentence images"):
            image, meta = self.generate_single()

            filename = f"image_{idx:05d}.png"
            self.save_image(image, filename)
            meta["file_name"] = str(self.output_dir / filename)

            metadata.append(meta)

        return metadata

    def generate_single(self, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        if not hasattr(self, "_original_typo_pairs"):
             self._original_typo_pairs = generate_sentence_typos(
                self._sentences, self._similarity_db, typo_ratio=kwargs.get("typo_ratio", 0.15)
            )

        original_text, typo_text = random.choice(self._original_typo_pairs)
        resolution_range = kwargs.get("resolution_range", (24, 48))
        if hasattr(self, "resolution_range"):
            resolution_range = self.resolution_range

        params = self._random_params(resolution_range)
        image = render_text_with_effects(text=typo_text, **params)

        metadata = {
            "typo_text": typo_text,
            "original_text": original_text,
            **params,
        }
        return image, metadata

    def _random_params(self, resolution_range: Tuple[int, int]) -> Dict[str, Any]:
        return {
            "font_path": random.choice(self.font_paths),
            "background_color": (
                random.randint(200, 255),
                random.randint(200, 255),
                random.randint(200, 255),
            ),
            "font_size": random.randint(*resolution_range),
            "bold": random.choice([True, False]),
            "tilt": random.randint(-30, 30),
            "shadow": random.choice([True, False]),
            "distortion": random.choice([True, False]),
            "blur": random.choice([True, False]),
            "contrast": random.choice([True, False]),
        }
