import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

from tqdm import tqdm
from utils import read_txt, read_json

from character_similarity import find_similar_chars
from generator.basic_generator import _generate_text_image

logger = logging.getLogger(__name__)


def needle_in_a_haystack_generator(
    corpus_path, db_path, top_n=3, num_size=(5, 10), needle_ratio=0.2
):
    corpus = read_txt(corpus_path)
    db = read_json(db_path)

    result = []
    chars = set(list(corpus))
    for char in chars:
        size_x = random.randint(num_size[0], num_size[1])
        size_y = random.randint(num_size[0], num_size[1])
        num_needle_to_inject = int(size_x * size_y * needle_ratio)

        sim_chars = find_similar_chars(char, db, top_n=top_n)
        if not sim_chars:
            continue

        temp_grid = [[char for _ in range(size_y)] for _ in range(size_x)]

        for _ in range(num_needle_to_inject):
            x = random.randint(0, size_x - 1)
            y = random.randint(0, size_y - 1)
            temp_grid[x][y] = sim_chars[0][0]

        generated_text = "\n".join(["".join(row) for row in temp_grid])

        actual_needle_count = generated_text.count(sim_chars[0][0])

        result.append(
            {
                "generated_text": generated_text,
                "original_char": char,
                "needle_char": sim_chars[0][0],
                "needle_count": actual_needle_count,
            }
        )

    return result


def generate_needle_in_a_haystack_images(
    corpus_path: str,
    db_path: str,
    num_images: int = 1000,
    output_dir: str = "images",
    resolution_range: Tuple[int, int] = (70, 90),
    lang: str = "ko",
) -> Optional[str]:
    """
    Reads text from a given corpus file, generates single-sentence images,
    and saves the image-text pair metadata.
    """
    logger.info(
        f"\nStarting [single sentence] image generation using '{corpus_path}'. Target number of images: {num_images:,}"
    )

    font_dir = Path(f"fonts/{lang}")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]
    if not font_paths:
        logger.error(
            f"Error: No .ttf font files found in the 'fonts/{lang}' directory. Aborting."
        )
        return None

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    generated_haystacks = needle_in_a_haystack_generator(corpus_path, db_path, top_n=3)
    if not generated_haystacks:
        logger.error("Error: No haystacks generated. Aborting.")
        return None

    background_colors = [
        (255, 255, 255),
        (240, 240, 240),
        (255, 250, 240),
        (240, 255, 240),
        (240, 248, 255),
        (255, 240, 245),
        (245, 245, 220),
        (250, 250, 210),
        (230, 230, 250),
    ]

    image_text_pairs: List[Dict[str, str]] = []

    for idx in tqdm(range(num_images), desc="Generating needle images"):
        font_path = random.choice(font_paths)

        selected_haystack_data = random.choice(generated_haystacks)
        text_to_render = selected_haystack_data["generated_text"]
        needle_char = selected_haystack_data["needle_char"]
        needle_count = selected_haystack_data["needle_count"]

        bg_color = random.choice(background_colors)
        font_size = random.randint(*resolution_range)

        bold = random.choice([True, False])
        tilt = random.randint(-15, 15)
        shadow = random.choice([True, False])
        distortion = random.choice([True, False])
        blur = random.choice([True, False])
        contrast = random.choice([True, False])

        img = _generate_text_image(
            text=text_to_render,
            font_path=font_path,
            background_color=bg_color,
            font_size=font_size,
            bold=bold,
            tilt=tilt,
            shadow=shadow,
            distortion=distortion,
            blur=blur,
            contrast=contrast,
        )

        image_filename = f"image_{idx:04d}.png"
        image_path = output_path / image_filename
        img.save(image_path)

        prompt_text = f"Count the number of '{needle_char}' characters in the image."
        response_text = str(needle_count)

        image_text_pairs.append(
            {
                "file_name": str(image_path),
                "text": text_to_render,
                "prompt": prompt_text,
                "response": response_text,
            }
        )

    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in image_text_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        f"Successfully generated {len(image_text_pairs):,} images and metadata."
    )
    return str(output_path)


if __name__ == "__main__":
    generate_needle_in_a_haystack_images(corpus_path="korean_char_corpus.txt")
