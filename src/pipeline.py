import argparse
import logging
from pathlib import Path
from typing import Any

# Assuming each module actually exists.
from corpus_generator import create_corpus_from_wiki
from character_similarity import generate_similar_chars_db
from generator.sentence_generator import generate_sentence_typos_images
from utils import upload_subset_to_hub

# Logger setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_paths_and_prerequisites(
    base_dir: Path, font_path: str, lang: str, num_sentences: int
) -> None:
    """
    Prepares the necessary base paths, corpus, and DB for the pipeline execution.
    """
    corpus_path = base_dir / f"corpus_{lang}.txt"
    db_path = base_dir / f"char_similarity_db_{lang}.json"

    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[SETUP] Base directory ready: '{base_dir}'")

    if not corpus_path.exists():
        logger.info(f"[CORPUS] Corpus not found, generating from Wikipedia...")
        create_corpus_from_wiki(
            output_path=corpus_path, lang=lang, num_sentences=num_sentences
        )
        logger.info(f"[CORPUS] Corpus created: '{corpus_path}'")
    else:
        logger.info(f"[CORPUS] Using existing corpus: '{corpus_path}'")

    if not db_path.exists():
        logger.info(f"[DB] Character similarity DB not found, generating...")
        generate_similar_chars_db(
            corpus_path=corpus_path, db_path=db_path, font_path=font_path
        )
        logger.info(f"[DB] Character similarity DB created: '{db_path}'")
    else:
        logger.info(f"[DB] Using existing character similarity DB: '{db_path}'")


def pipeline(
    repo_id: str,
    font_path: str,
    size: int,
    corpus_size: int,
    output_dir: str,
    lang: str,
    typo_ratio: float = 0.15,
    **kwargs: Any,  # kwargs is kept to accept other arguments from argparse
) -> None:
    """
    Runs the pipeline to generate and upload the 'Sentence Typos' dataset.
    """
    logger.info("=" * 80)
    logger.info(" Synthetic OCR Datasets - Sentence Typos ".center(80))
    logger.info("=" * 80)

    # --- 1. Setup Paths and Prepare Corpus/DB ---
    base_dir = Path(output_dir) / lang
    setup_paths_and_prerequisites(base_dir, font_path, lang, corpus_size)

    corpus_path = base_dir / f"corpus_{lang}.txt"
    db_path = base_dir / f"char_similarity_db_{lang}.json"

    # --- 2. Generate 'Sentence Typos' Images ---
    task_name = "Sentence Typos"
    logger.info(f"\n--- [TASK] Starting {task_name} image generation ---")

    if size <= 0:
        logger.warning("Requested number of images is 0, terminating the pipeline.")
        return

    logger.info(f"Requested number of images: {size}")

    task_output_dir = base_dir / "images_sentence_typos"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = None
    try:
        # Call the generation function directly.
        generated_dir_path = generate_sentence_typos_images(
            corpus_path=corpus_path,
            db_path=db_path,
            lang=lang,
            num_images=size,
            output_dir=task_output_dir,
            typo_ratio=typo_ratio,
        )

        if generated_dir_path is None or not Path(generated_dir_path).exists():
            raise RuntimeError(
                "The generation function did not return a valid directory path."
            )

        generated_dir = Path(generated_dir_path)
        logger.info(f"✓ Success: Images generated in '{generated_dir}'.")

    except Exception as e:
        logger.error(
            f"✗ Failure: An error occurred during {task_name} image generation: {e}",
            exc_info=True,
        )
        return  # Stop the pipeline on error

    # --- 3. Upload to Hugging Face Hub ---
    if generated_dir:
        logger.info(f"\n--- [UPLOAD] Starting upload to Hugging Face Hub: {repo_id} ---")
        config_name = "default"  # Specify as 'default' or 'main' for a single subset

        try:
            logger.info(
                f"Uploading '{config_name}' subset from '{generated_dir}'..."
            )
            upload_subset_to_hub(
                repo_id=repo_id, subset_dir=generated_dir, config_name="default"
            )
            logger.info(f"✓ Success: Uploaded '{config_name}' subset.")
        except Exception as e:
            logger.error(
                f"✗ Failure: An error occurred during subset upload: {e}",
                exc_info=True,
            )
    else:
        logger.warning("No dataset was generated, skipping upload.")

    logger.info("\n" + " Pipeline completed successfully! ".center(80, "="))
    logger.info(
        f"Check your dataset on the Hub: https://huggingface.co/datasets/{repo_id}"
    )
