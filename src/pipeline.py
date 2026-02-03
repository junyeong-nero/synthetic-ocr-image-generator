import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from corpus_generator import create_corpus_from_wiki
from character_similarity import generate_similar_chars_db
from env_utils import set_global_seed
from utils import upload_subset_to_hub
from generator.realism_stats import write_realism_stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _ensure_corpus_and_db(
    base_dir: Path,
    font_path: str,
    lang: str,
    num_sentences: int,
    similarity_threshold: float,
    similarity_top_k: int,
) -> tuple[Path, Path]:
    corpus_path = base_dir / f"corpus_{lang}.txt"
    db_path = base_dir / f"char_similarity_db_{lang}.json"

    base_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        logger.info("[CORPUS] Generating from Wikipedia...")
        create_corpus_from_wiki(
            output_path=str(corpus_path), lang=lang, num_sentences=num_sentences
        )
    else:
        logger.info(f"[CORPUS] Using existing: {corpus_path}")

    if not db_path.exists():
        logger.info("[DB] Generating character similarity DB...")
        generate_similar_chars_db(
            corpus_path=str(corpus_path),
            db_path=str(db_path),
            font_path=font_path,
            threshold=similarity_threshold,
            top_k=similarity_top_k,
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
    """Generator for mixed format datasets (sentence, table, document, markdown, kie)."""

    FORMATS = ["sentence", "table", "document", "markdown", "kie"]

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
        self._markdown_generator = None
        self._kie_generator = None

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

    @property
    def markdown_generator(self):
        if self._markdown_generator is None:
            from generator import MarkdownGenerator
            self._markdown_generator = MarkdownGenerator(
                output_dir=str(self.output_dir / "markdown"),
                font_dir=str(self.font_dir),
                lang=self.lang,
            )
        return self._markdown_generator

    @property
    def kie_generator(self):
        if self._kie_generator is None:
            from generator import KIEGenerator
            self._kie_generator = KIEGenerator(
                output_dir=str(self.output_dir / "kie"),
                font_dir=str(self.font_dir),
                lang=self.lang,
            )
        return self._kie_generator

    def run(
        self,
        num_images: int,
        typo_ratio: float = 0.15,
        table_size: str = "3-8",
        format_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Generate mixed format dataset."""
        if format_weights is None:
            format_weights = {"sentence": 0.25, "table": 0.2, "document": 0.2, "markdown": 0.15, "kie": 0.2}

        min_rows, max_rows = _parse_table_size(table_size)
        row_range = (min_rows, max_rows)
        col_range = (min_rows, max_rows)

        all_metadata: List[Dict[str, Any]] = []

        try:
            logger.info(
                f"Starting MixedGenerator: generating {num_images:,} images "
                f"(weights: {format_weights})"
            )

            for idx in tqdm(range(num_images), desc="Generating mixed images"):
                fmt = random.choices(
                    list(format_weights.keys()),
                    weights=list(format_weights.values()),
                )[0]

                if fmt == "sentence":
                    image, meta = self.sentence_generator.generate_single(
                        typo_ratio=typo_ratio
                    )
                    filename = f"sentence_{idx:05d}.png"
                    self.sentence_generator.save_image(image, filename)
                    meta["file_name"] = str(self.sentence_generator.output_dir / filename)
                    meta["format"] = "sentence"
                    all_metadata.append(meta)

                elif fmt == "table":
                    image, meta = self.table_generator.generate_single(
                        row_range=row_range,
                        col_range=col_range,
                    )
                    filename = f"table_{idx:05d}.png"
                    self.table_generator.save_image(image, filename)
                    meta["file_name"] = str(self.table_generator.output_dir / filename)
                    meta["format"] = "table"
                    all_metadata.append(meta)

                elif fmt == "document":
                    image, meta = self.document_generator.generate_single()
                    filename = f"document_{idx:05d}.png"
                    self.document_generator.save_image(image, filename)
                    meta["file_name"] = str(self.document_generator.output_dir / filename)
                    meta["format"] = "document"
                    all_metadata.append(meta)

                elif fmt == "markdown":
                    image, meta = self.markdown_generator.generate_single()
                    filename = f"markdown_{idx:05d}.png"
                    self.markdown_generator.save_image(image, filename)
                    meta["file_name"] = str(self.markdown_generator.output_dir / filename)
                    meta["format"] = "markdown"
                    all_metadata.append(meta)

                elif fmt == "kie":
                    image, meta = self.kie_generator.generate_single()
                    filename = f"kie_{idx:05d}.png"
                    self.kie_generator.save_image(image, filename)
                    meta["file_name"] = str(self.kie_generator.output_dir / filename)
                    meta["format"] = "kie"
                    all_metadata.append(meta)

            metadata_path = self.output_dir / "metadata.jsonl"
            with open(metadata_path, "w", encoding="utf-8") as f:
                for item in all_metadata:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logger.info(f"Saved metadata to '{metadata_path}'")
            write_realism_stats(self.output_dir, all_metadata, format_name="mixed")
            logger.info(f"Successfully generated {len(all_metadata):,} mixed format images")
            return str(self.output_dir)

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return None



def _upload_mixed_format_to_hub(repo_id: str, output_dir: Path):
    """Upload mixed format dataset, separating each format type into its own subset."""
    import shutil
    import tempfile

    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return

    # Group metadata by format type
    format_groups = {"sentence": [], "table": [], "document": [], "markdown": [], "kie": []}

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            fmt = data.get("format", "sentence")
            if fmt in format_groups:
                format_groups[fmt].append(data)

    # Upload each format as separate subset
    for fmt, items in format_groups.items():
        if not items:
            continue

        logger.info(f"  Uploading '{fmt}' subset: {len(items):,} samples")

        # Create temporary directory for this subset
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write metadata file
            subset_metadata_path = temp_path / "metadata.jsonl"
            with open(subset_metadata_path, "w", encoding="utf-8") as f:
                for item in items:
                    # Create copy without the format field for cleaner subset
                    item_copy = item.copy()
                    item_copy.pop("format", None)
                    f.write(json.dumps(item_copy, ensure_ascii=False) + "\n")

            # Copy images
            for item in items:
                src = Path(item["file_name"])
                dst = temp_path / src.name
                if src.exists():
                    shutil.copy(src, dst)

            # Upload
            try:
                upload_subset_to_hub(
                    repo_id=repo_id,
                    subset_dir=temp_path,
                    config_name=fmt,
                )
            except Exception as e:
                logger.error(f"  Failed to upload '{fmt}' subset: {e}")


def pipeline(
    repo_id: str,
    font_path: str,
    size: int,
    corpus_size: int,
    output_dir: str,
    lang: str,
    typo_ratio: float = 0.15,
    similarity_threshold: float = 0.6,
    similarity_top_k: int = 8,
    format: str = "sentence",
    template: Optional[str] = None,
    table_size: str = "3-8",
    mixed: bool = False,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> None:
    logger.info("=" * 80)
    set_global_seed(seed)
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
        corpus_path, db_path = _ensure_corpus_and_db(
            base_dir,
            font_path,
            lang,
            corpus_size,
            similarity_threshold,
            similarity_top_k,
        )
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

    else:
        from generator.registry import GeneratorRegistry
        
        try:
            generator_cls = GeneratorRegistry.get_generator_class(format)
        except ValueError:
            logger.error(f"Unknown format: {format}")
            return

        task_output_dir = base_dir / f"images_{format}"
        
        # Prepare initialization arguments
        init_kwargs = {
            "output_dir": str(task_output_dir),
            "font_dir": str(font_dir),
            "lang": lang,
        }
        
        if format == "sentence":
            corpus_path, db_path = _ensure_corpus_and_db(
                base_dir,
                font_path,
                lang,
                corpus_size,
                similarity_threshold,
                similarity_top_k,
            )
            init_kwargs["corpus_path"] = str(corpus_path)
            init_kwargs["similarity_db_path"] = str(db_path)
            
        generator = generator_cls(**init_kwargs)
        
        # Prepare run arguments
        run_kwargs: Dict[str, Any] = {"num_images": size}
        if format == "sentence":
            run_kwargs["typo_ratio"] = typo_ratio
        elif format == "table":
            min_rows, max_rows = _parse_table_size(table_size)
            run_kwargs["row_range"] = (min_rows, max_rows)
            run_kwargs["col_range"] = (min_rows, max_rows)
            run_kwargs["template"] = template
        elif format == "document":
            run_kwargs["template"] = template
        elif format == "markdown":
            run_kwargs["template"] = template
        elif format == "kie":
            run_kwargs["doc_type"] = template
            
        generated_dir = generator.run(**run_kwargs)

    if generated_dir:
        logger.info(f"\n--- Uploading to Hugging Face Hub: {repo_id} ---")

        if mixed:
            # For mixed format, upload each format type as separate subset
            _upload_mixed_format_to_hub(repo_id, Path(generated_dir))
        else:
            # For single format, use format name as subset
            try:
                upload_subset_to_hub(
                    repo_id=repo_id,
                    subset_dir=Path(generated_dir),
                    config_name=format,
                )
            except Exception as e:
                logger.error(f"Upload failed: {e}", exc_info=True)
    else:
        logger.warning("No dataset was generated, skipping upload.")

    logger.info("\n" + " Pipeline completed! ".center(80, "="))
    logger.info(f"Dataset: https://huggingface.co/datasets/{repo_id}")
