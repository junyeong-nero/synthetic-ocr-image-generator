import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from corpus_generator import create_corpus_from_wiki
from character_similarity import generate_similar_chars_db
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


def _parse_table_size(table_size: str) -> Tuple[int, int]:
    """Parse table size string like '3-8' into (min, max) tuple."""
    try:
        parts = table_size.split("-")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return 3, 8
    except (ValueError, AttributeError):
        return 3, 8


class MixedGenerator:
    """Generator for mixed format datasets (sentence, table, document)."""

    FORMATS = ["sentence", "table", "document"]

    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str,
        corpus_path: Path,
        db_path: Path,
        font_path: str,
        corpus_size: int,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_dir = Path(font_dir)
        self.lang = lang
        self.corpus_path = corpus_path
        self.db_path = db_path
        self.font_path = font_path
        self.corpus_size = corpus_size

        self._sentence_generator = None
        self._table_generator = None
        self._document_generator = None

    @property
    def sentence_generator(self):
        if self._sentence_generator is None:
            from generator import SentenceGenerator
            self._sentence_generator = SentenceGenerator(
                output_dir=str(self.output_dir / "sentence"),
                font_dir=str(self.font_dir),
                corpus_path=str(self.corpus_path),
                similarity_db_path=str(self.db_path),
                lang=self.lang,
            )
        return self._sentence_generator

    @property
    def table_generator(self):
        if self._table_generator is None:
            from generator import TableGenerator
            self._table_generator = TableGenerator(
                output_dir=str(self.output_dir / "table"),
                font_dir=str(self.font_dir),
                lang=self.lang,
            )
        return self._table_generator

    @property
    def document_generator(self):
        if self._document_generator is None:
            from generator import DocumentGenerator
            self._document_generator = DocumentGenerator(
                output_dir=str(self.output_dir / "document"),
                font_dir=str(self.font_dir),
                lang=self.lang,
            )
        return self._document_generator

    def run(
        self,
        num_images: int,
        typo_ratio: float = 0.15,
        table_size: str = "3-8",
        format_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Generate mixed format dataset."""
        if format_weights is None:
            format_weights = {"sentence": 0.4, "table": 0.3, "document": 0.3}

        min_rows, max_rows = _parse_table_size(table_size)
        row_range = (min_rows, max_rows)
        col_range = (min_rows, max_rows)

        all_metadata: List[Dict[str, Any]] = []

        try:
            logger.info(
                f"Starting MixedGenerator: generating {num_images:,} images "
                f"(weights: {format_weights})"
            )

            for idx in range(num_images):
                fmt = random.choices(
                    list(format_weights.keys()),
                    weights=list(format_weights.values()),
                )[0]

                if fmt == "sentence":
                    original_text, typo_text = random.choice(
                        self.sentence_generator._sentences
                    ), random.choice(self.sentence_generator._sentences)
                    params = self.sentence_generator._random_params((24, 48))
                    from generator.effects import render_text_with_effects
                    image = render_text_with_effects(text=typo_text, **params)
                    filename = f"sentence_{idx:05d}.png"
                    self.sentence_generator.save_image(image, filename)
                    all_metadata.append({
                        "file_name": str(self.sentence_generator.output_dir / filename),
                        "format": "sentence",
                        "typo_text": typo_text,
                        "original_text": original_text,
                        **params,
                    })

                elif fmt == "table":
                    template = random.choice(list(self.table_generator.data_generator.TEMPLATES.keys()))
                    num_rows = random.randint(*row_range)
                    num_cols = random.randint(*col_range)
                    font_size = random.randint(12, 18)

                    table = self.table_generator.data_generator.generate_table(
                        template=template,
                        num_rows=num_rows,
                        num_cols=num_cols,
                    )

                    font_path = random.choice(self.table_generator.font_paths)
                    from generator.table_generator import TableRenderer
                    renderer = TableRenderer(font_path, font_size)
                    image = renderer.render(table)

                    filename = f"table_{idx:05d}.png"
                    self.table_generator.save_image(image, filename)

                    all_metadata.append({
                        "file_name": str(self.table_generator.output_dir / filename),
                        "format": "table",
                        "html": table.to_html(),
                        "json": table.to_json(),
                        "template": template,
                        "num_rows": table.num_rows,
                        "num_cols": table.num_cols,
                        "font_size": font_size,
                    })

                elif fmt == "document":
                    from generator.document_generator import DocumentTemplate
                    template = random.choice(list(DocumentTemplate))
                    style = self.document_generator.data_generator._random_style()

                    document = self.document_generator.data_generator.generate_document(
                        template=template,
                        style=style,
                    )

                    font_path = random.choice(self.document_generator.font_paths)
                    from generator.document_generator import DocumentRenderer
                    renderer = DocumentRenderer(font_path, random.randint(10, 14))
                    image, _ = renderer.render(document)

                    filename = f"document_{idx:05d}.png"
                    self.document_generator.save_image(image, filename)

                    gt = document.to_ground_truth()
                    all_metadata.append({
                        "file_name": str(self.document_generator.output_dir / filename),
                        "format": "document",
                        "template": template.value,
                        "ground_truth": gt,
                        "elements_count": len(gt["elements"]),
                    })

            metadata_path = self.output_dir / "metadata.jsonl"
            with open(metadata_path, "w", encoding="utf-8") as f:
                for item in all_metadata:
                    f.write(f"{item}\n".replace("'", '"'))

            logger.info(f"Saved metadata to '{metadata_path}'")
            logger.info(f"Successfully generated {len(all_metadata):,} mixed format images")
            return str(self.output_dir)

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return None


def pipeline(
    repo_id: str,
    font_path: str,
    size: int,
    corpus_size: int,
    output_dir: str,
    lang: str,
    typo_ratio: float = 0.15,
    format: str = "sentence",
    template: str = None,
    table_size: str = "3-8",
    mixed: bool = False,
    **kwargs: Any,
) -> None:
    logger.info("=" * 80)
    if mixed:
        logger.info(" Synthetic OCR Dataset Generator (Mixed Format) ".center(80))
    else:
        logger.info(" Synthetic OCR Dataset Generator ".center(80))
        logger.info(f" Format: {format} ".center(80))
    logger.info("=" * 80)

    if size <= 0:
        logger.warning("Requested number of images is 0, terminating.")
        return

    base_dir = Path(output_dir) / lang
    font_dir = Path(f"fonts/{lang}")

    if mixed:
        corpus_path, db_path = _ensure_corpus_and_db(base_dir, font_path, lang, corpus_size)
        task_output_dir = base_dir / "images_mixed"

        mixed_gen = MixedGenerator(
            output_dir=str(task_output_dir),
            font_dir=str(font_dir),
            lang=lang,
            corpus_path=corpus_path,
            db_path=db_path,
            font_path=font_path,
            corpus_size=corpus_size,
        )

        generated_dir = mixed_gen.run(
            num_images=size,
            typo_ratio=typo_ratio,
            table_size=table_size,
        )

    elif format == "sentence":
        from generator import SentenceGenerator

        corpus_path, db_path = _ensure_corpus_and_db(base_dir, font_path, lang, corpus_size)
        task_output_dir = base_dir / "images_sentence_typos"

        generator = SentenceGenerator(
            output_dir=str(task_output_dir),
            font_dir=str(font_dir),
            corpus_path=str(corpus_path),
            similarity_db_path=str(db_path),
            lang=lang,
        )

        generated_dir = generator.run(num_images=size, typo_ratio=typo_ratio)

    elif format == "table":
        from generator import TableGenerator

        task_output_dir = base_dir / "images_tables"
        min_rows, max_rows = _parse_table_size(table_size)
        row_range = (min_rows, max_rows)
        col_range = (min_rows, max_rows)

        generator = TableGenerator(
            output_dir=str(task_output_dir),
            font_dir=str(font_dir),
            lang=lang,
        )

        generated_dir = generator.run(
            num_images=size,
            template=template,
            row_range=row_range,
            col_range=col_range,
        )

    elif format == "document":
        from generator import DocumentGenerator

        task_output_dir = base_dir / "images_documents"

        generator = DocumentGenerator(
            output_dir=str(task_output_dir),
            font_dir=str(font_dir),
            lang=lang,
        )

        generated_dir = generator.run(num_images=size, template=template)

    else:
        logger.error(f"Unknown format: {format}")
        return

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
