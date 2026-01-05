import logging
from pathlib import Path
from typing import Any

from corpus_generator import create_corpus_from_wiki
from character_similarity import generate_similar_chars_db
from generator import SentenceGenerator
from utils import upload_subset_to_hub

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _ensure_corpus_and_db(
    base_dir: Path, font_path: str, lang: str, num_sentences: int
) -> tuple[Path, Path]:
    corpus_path = base_dir / f"corpus_{lang}.txt"
    db_path = base_dir / f"char_similarity_db_{lang}.json"

    base_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        logger.info("[CORPUS] Generating from Wikipedia...")
        create_corpus_from_wiki(
            output_path=corpus_path, lang=lang, num_sentences=num_sentences
        )
    else:
        logger.info(f"[CORPUS] Using existing: {corpus_path}")

    if not db_path.exists():
        logger.info("[DB] Generating character similarity DB...")
        generate_similar_chars_db(
            corpus_path=corpus_path, db_path=db_path, font_path=font_path
        )
    else:
        logger.info(f"[DB] Using existing: {db_path}")

    return corpus_path, db_path


def pipeline(
    repo_id: str,
    font_path: str,
    size: int,
    corpus_size: int,
    output_dir: str,
    lang: str,
    typo_ratio: float = 0.15,
    **kwargs: Any,
) -> None:
    logger.info("=" * 80)
    logger.info(" Synthetic OCR Dataset Generator ".center(80))
    logger.info("=" * 80)

    if size <= 0:
        logger.warning("Requested number of images is 0, terminating.")
        return

    base_dir = Path(output_dir) / lang
    corpus_path, db_path = _ensure_corpus_and_db(base_dir, font_path, lang, corpus_size)

    task_output_dir = base_dir / "images_sentence_typos"
    font_dir = Path(f"fonts/{lang}")

    generator = SentenceGenerator(
        output_dir=str(task_output_dir),
        font_dir=str(font_dir),
        corpus_path=str(corpus_path),
        similarity_db_path=str(db_path),
        lang=lang,
    )

    generated_dir = generator.run(num_images=size, typo_ratio=typo_ratio)

    if generated_dir:
        logger.info(f"\n--- Uploading to Hugging Face Hub: {repo_id} ---")
        try:
            upload_subset_to_hub(
                repo_id=repo_id,
                subset_dir=Path(generated_dir),
                config_name="default",
            )
        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)
    else:
        logger.warning("No dataset was generated, skipping upload.")

    logger.info("\n" + " Pipeline completed! ".center(80, "="))
    logger.info(f"Dataset: https://huggingface.co/datasets/{repo_id}")
