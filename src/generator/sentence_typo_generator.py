import json
import random
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils import read_txt, read_json
from generator.basic_generator import _generate_text_image
from character_similarity import generate_sentence_typos

logger = logging.getLogger(__name__)

BACKGROUND_COLORS: List[Tuple[int, int, int]] = [
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


def generate_sentence_typos_images(
    corpus_path: str,
    db_path: str,
    num_images: int = 1000,
    output_dir: str = "images",
    resolution_range: Tuple[int, int] = (70, 90),
    bold: Optional[bool] = None,
    tilt: Optional[int] = None,
    shadow: Optional[bool] = None,
    distortion: Optional[bool] = None,
    blur: Optional[bool] = None,
    contrast: Optional[bool] = None,
    lang: str = "ko",
) -> Optional[str]:
    """
    Generates a dataset of images from a text corpus.

    Args:
        corpus_path: Path to the input text file.
        num_images: The total number of images to generate.
        output_dir: The directory to save the images and metadata.
        resolution_range: A tuple (min, max) for random font sizes.
        bold: Force bold effect. If None, it's randomly applied.
        tilt: Force a specific tilt angle. If None, it's randomly set.
        shadow: Force shadow effect. If None, it's randomly applied.
        distortion: Force distortion. If None, it's randomly applied.
        blur: Force blur. If None, it's randomly applied.
        contrast: Force contrast adjustment. If None, it's randomly applied.

    Returns:
        The path to the output directory, or None if an error occurs.
    """
    logger.info(
        f"\nStarting image generation from '{corpus_path}'. "
        f"Target: {num_images:,} images."
    )

    font_dir = Path(f"fonts/{lang}")
    font_paths = [str(p) for p in font_dir.glob("*.ttf")]
    if not font_paths:
        logger.error(
            f"Error: No .ttf font files found in 'fonts/{lang}' directory. Aborting."
        )
        return None

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    lines = read_txt(corpus_path).splitlines()
    korean_texts = [line[:20].strip() for line in lines if line.strip()]
    if not korean_texts:
        logger.error(f"Error: No text found in '{corpus_path}'. Aborting.")
        return None

    db = read_json(db_path)
    korean_texts = generate_sentence_typos(korean_texts, db, top_n=1)

    image_text_pairs: List[Dict[str, str]] = []

    for idx in tqdm(range(num_images), desc="Generating Images"):
        # --- Determine Parameters for this Image ---
        font_path = random.choice(font_paths)
        text = random.choice(korean_texts)
        bg_color = random.choice(BACKGROUND_COLORS)
        font_size = random.randint(*resolution_range)

        # Apply effects randomly if not specified by the user
        apply_bold = random.choice([True, False]) if bold is None else bold
        apply_tilt = random.randint(-15, 15) if tilt is None else tilt
        apply_shadow = random.choice([True, False]) if shadow is None else shadow
        apply_dist = random.choice([True, False]) if distortion is None else distortion
        apply_blur = random.choice([True, False]) if blur is None else blur
        apply_contrast = random.choice([True, False]) if contrast is None else contrast

        # --- Generate and Save Image ---
        img = _generate_text_image(
            text=text,
            font_path=font_path,
            background_color=bg_color,
            font_size=font_size,
            bold=apply_bold,
            tilt=apply_tilt,
            shadow=apply_shadow,
            distortion=apply_dist,
            blur=apply_blur,
            contrast=apply_contrast,
        )

        image_filename = f"image_{idx:05d}.png"
        image_filepath = output_path / image_filename
        img.save(image_filepath)

        image_text_pairs.append(
            {
                "file_name": str(image_filepath),
                "response": text,
                "background_color": bg_color,
                "font_size": font_size,
                "bold": apply_bold,
                "tilt": apply_tilt,
                "shadow": apply_shadow,
                "distortion": apply_dist,
                "blur": apply_blur,
                "contrast": apply_contrast,
            }
        )

    # --- Save Metadata ---
    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in image_text_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Successfully generated {len(image_text_pairs):,} images.")
    return str(output_path)
