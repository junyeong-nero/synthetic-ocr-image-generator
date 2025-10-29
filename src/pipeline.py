import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, Any

from corpus_generator import create_corpus_from_wiki
from character_similarity import generate_similar_chars_db

from generator.document_generator import generate_document_images
from generator.sentence_generator import generate_sentence_images
from generator.table_generator import generate_table_images
from generator.table_numeric_generator import generate_table_numeric_images
from generator.needle_in_a_haystack_generator import (
    generate_needle_in_a_haystack_images,
)

from utils import upload_subset_to_hub

logger = logging.getLogger(__name__)


def pipeline(
    repo_id: str,
    num_sentence_images: int,
    num_sentence_noise_images: int,
    num_document_images: int,
    num_table_images: int,
    num_needle_images: int,
    output_dir: str,
    lang: str,
    **kwargs: Any,
) -> None:
    """
    Executes the full data generation and upload pipeline.

    This function orchestrates the process of:
    1. Creating a text corpus from Wikipedia if it doesn't exist.
    2. Generating synthetic images for sentences, documents, tables, and "needle-in-a-haystack."
    3. Uploading the generated datasets to the Hugging Face Hub.

    Args:
        corpus_path: Path to the text corpus file.
        repo_id: The Hugging Face Hub repository ID (e.g., "username/dataset-name").
        num_sentence_images: The number of sentence images to generate.
        num_document_images: The number of document images to generate.
        num_table_images: The number of table images to generate.
        num_needle_images: The number of "needle in a haystack" images to generate.
        output_dir: The root directory to save all generated content.
        lang: The language code for corpus generation (e.g., "ko", "en").
        **kwargs: Additional keyword arguments (currently unused).
    """
    logger.info("=" * 80)
    logger.info(" VDG: Visual Document Generation Pipeline ".center(80))
    logger.info("=" * 80)

    # --- 1. Path Initialization and Directory Setup ---
    logger.info(f"\n[SETUP] Initializing paths and directories in '{output_dir}'...")
    base_output_path = Path(output_dir)
    db_path = base_output_path / f"char_similarity_db_{lang}.json"
    corpus_file_path = base_output_path / f"corpus_{lang}.txt"

    # Define specific output paths for each data type
    paths = {
        "sentence": base_output_path / "images_sentence",
        "sentence_noise": base_output_path / "images_sentence_noise",
        "table": base_output_path / "images_table",
        "table_numeric": base_output_path / "images_table_numeric",
        "document": base_output_path / "images_document",
        "needle": base_output_path / "images_needle_in_a_haystack",
    }

    # Create all necessary directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    if not corpus_file_path.parent.exists():
        corpus_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[SETUP] All directories are ready.")

    # --- 2. Corpus Generation ---
    if not corpus_file_path.exists():
        logger.info(
            f"\n[CORPUS] Corpus not found at '{corpus_file_path}'. Creating from Wikipedia..."
        )
        create_corpus_from_wiki(
            output_path=str(corpus_file_path), lang=lang, num_sentences=5000
        )
        logger.info(f"[CORPUS] Successfully created corpus.")
    else:
        logger.info(f"\n[CORPUS] Using existing corpus at '{corpus_file_path}'.")

    if not db_path.exists():
        logger.info(
            f"\n[DB] Character similarity database not found at '{db_path}'. Creating..."
        )
        generate_similar_chars_db(
            corpus_path=str(corpus_file_path), db_path=str(db_path)
        )
        logger.info(f"[DB] Successfully created character similarity database.")
    else:
        logger.info(
            f"\n[DB] Using existing character similarity database at '{db_path}'."
        )

    # --- 3. Image Generation Tasks ---
    GENERATION_TASKS = [
        {
            "name": "Sentence",
            "func": generate_sentence_images,
            "args": {
                "num_images": num_sentence_images,
                "output_dir": str(paths["sentence"]),
                "lang": lang,
                "bold": False,
                "tilt": 0,
                "shadow": False,
                "distortion": False,
                "blur": False,
                "contrast": False,
            },
            "config_name": "sentence",
        },
        {
            "name": "Sentence Noise",
            "func": generate_sentence_images,
            "args": {
                "num_images": num_sentence_noise_images,
                "output_dir": str(paths["sentence_noise"]),
                "lang": lang,
            },
            "config_name": "sentence_noise",
        },
        {
            "name": "Document",
            "func": generate_document_images,
            "args": {
                "num_images": num_document_images,
                "output_dir": str(paths["document"]),
                "lang": lang,
            },
            "config_name": "document",
        },
        {
            "name": "Table",
            "func": generate_table_images,
            "args": {
                "num_images": num_table_images,
                "output_dir": str(paths["table"]),
                "lang": lang,
            },
            "config_name": "table",
        },
        {
            "name": "Table Numeric",
            "func": generate_table_numeric_images,
            "args": {
                "num_images": num_table_images,
                "output_dir": str(paths["table_numeric"]),
                "lang": lang,
            },
            "config_name": "table",
        },
        {
            "name": "Needle in a Haystack",
            "func": generate_needle_in_a_haystack_images,
            "args": {
                "db_path": str(db_path),
                "num_images": num_needle_images,
                "output_dir": str(paths["needle"]),
                "lang": lang,
            },
            "config_name": "needle_in_a_haystack",
        },
    ]

    generated_dirs: Dict[str, Path] = {}
    for task in GENERATION_TASKS:
        name = task["name"]
        num_images = task["args"]["num_images"]

        logger.info(f"\n--- Generating {name} Images ---")
        if num_images > 0:
            logger.info(f"Requesting {num_images} images.")
            generated_dir = task["func"](
                corpus_path=str(corpus_file_path), **task["args"]
            )
            if generated_dir is None:
                logger.error(f"Error: Failed to generate {name} images. Aborting.")
                return
            generated_dirs[task["config_name"]] = Path(generated_dir)
            logger.info(f"Successfully generated {name} images in '{generated_dir}'")
        else:
            logger.info(f"Skipping {name} generation (0 images requested).")

    # --- 4. Upload to Hugging Face Hub ---
    logger.info(f"\n--- Uploading to Hugging Face Hub: {repo_id} ---")
    if not generated_dirs:
        logger.info("No datasets were generated, so nothing to upload.")
    else:
        for config_name, dir_path in generated_dirs.items():
            logger.info(f"Uploading '{config_name}' subset from '{dir_path}'...")
            upload_subset_to_hub(str(dir_path), repo_id, config_name=config_name)
            logger.info(f"Successfully uploaded '{config_name}'.")

    logger.info("\n" + " Pipeline Completed Successfully! ".center(80, "="))
    logger.info(
        f"Check your dataset on the Hub: https://huggingface.co/datasets/{repo_id}"
    )
