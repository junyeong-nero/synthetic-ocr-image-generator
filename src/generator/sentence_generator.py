import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

    def _load_corpus(self) -> List[str]:
        lines = read_txt(str(self.corpus_path)).splitlines()
        return [" ".join(line.split()[:5]).strip() for line in lines if line.strip()]

    def generate(
        self,
        num_images: int,
        typo_ratio: float = 0.15,
        resolution_range: Tuple[int, int] = (24, 48),
    ) -> List[Dict[str, Any]]:
        original_typo_pairs = generate_sentence_typos(
            self._sentences, self._similarity_db, typo_ratio=typo_ratio
        )

        metadata = []
        for idx in tqdm(range(num_images), desc="Generating sentence images"):
            original_text, typo_text = random.choice(original_typo_pairs)

            params = self._random_params(resolution_range)
            image = render_text_with_effects(text=typo_text, **params)

            filename = f"image_{idx:05d}.png"
            self.save_image(image, filename)

            metadata.append({
                "file_name": str(self.output_dir / filename),
                "typo_text": typo_text,
                "original_text": original_text,
                **params,
            })

        return metadata

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
