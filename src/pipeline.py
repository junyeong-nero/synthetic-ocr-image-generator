import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from env_utils import set_global_seed
from utils import markdown_to_json_ast, upload_subset_to_hub
from generator.realism_stats import write_realism_stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _json_to_markdown(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _build_unified_ground_truth(fmt: str, metadata: Dict[str, Any]) -> Tuple[str, Any]:
    if fmt == "markdown":
        markdown_text = str(metadata.get("GT_markdown", metadata.get("markdown", "")))
        return markdown_text, markdown_to_json_ast(markdown_text)

    fallback_json = {"ground_truth": metadata.get("ground_truth", ""), "format": fmt}
    return _json_to_markdown(fallback_json), fallback_json


def _attach_unified_ground_truth(fmt: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    markdown_gt, json_gt = _build_unified_ground_truth(fmt, metadata)
    updated = dict(metadata)
    updated["GT_markdown"] = markdown_gt
    updated["GT_json"] = json_gt
    updated.pop("ground_truth", None)
    updated.pop("markdown", None)
    updated.pop("json", None)
    return updated


class MixedGenerator:
    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_dir = Path(font_dir)
        self.lang = lang
        self._markdown_generator = None

    @property
    def markdown_generator(self):
        if self._markdown_generator is None:
            from generator import Generator
            self._markdown_generator = Generator(
                output_dir=str(self.output_dir / "markdown"),
                font_dir=str(self.font_dir),
                lang=self.lang,
            )
        return self._markdown_generator

    def run(
        self,
        num_images: int,
        template: Optional[str] = None,
        markdown_renderer: str = "pil",
        similar_char_ratio: float = 0.08,
        similarity_db_path: Optional[str] = None,
        add_noise: Optional[bool] = None,
        add_blur: Optional[bool] = None,
    ) -> Optional[str]:
        all_metadata: List[Dict[str, Any]] = []

        try:
            logger.info(f"Starting markdown generation: {num_images:,} images")

            for idx in tqdm(range(num_images), desc="Generating markdown images"):
                generation_kwargs: Dict[str, Any] = {
                    "template": template,
                    "markdown_renderer": markdown_renderer,
                    "similar_char_ratio": similar_char_ratio,
                    "similarity_db_path": similarity_db_path,
                }
                if add_noise is not None:
                    generation_kwargs["add_noise"] = add_noise
                if add_blur is not None:
                    generation_kwargs["add_blur"] = add_blur

                image, meta = self.markdown_generator.generate_single(**generation_kwargs)
                filename = f"markdown_{idx:05d}.png"
                self.markdown_generator.save_image(image, filename)
                meta["file_name"] = str(self.markdown_generator.output_dir / filename)
                meta = _attach_unified_ground_truth("markdown", meta)
                all_metadata.append(meta)

            metadata_path = self.output_dir / "metadata.jsonl"
            with open(metadata_path, "w", encoding="utf-8") as f:
                for item in all_metadata:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            logger.info(f"Saved metadata to '{metadata_path}'")
            write_realism_stats(self.output_dir, all_metadata, format_name="markdown")
            logger.info(f"Successfully generated {len(all_metadata):,} markdown images")
            return str(self.output_dir)

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return None



def _upload_mixed_format_to_hub(
    repo_id: str,
    output_dir: Path,
    train_ratio: float = 0.9,
    test_ratio: float = 0.1,
):
    import shutil
    import tempfile

    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return

    records: List[Dict[str, Any]] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        logger.warning("No records found, skipping upload.")
        return

    logger.info(
        "Using mixed split ratios: train=%.3f, test=%.3f",
        train_ratio,
        test_ratio,
    )

    random.Random(42).shuffle(records)
    split_index = int(len(records) * train_ratio)
    if len(records) > 1:
        split_index = min(max(split_index, 1), len(records) - 1)
    else:
        split_index = len(records)

    split_to_items = {
        "train": records[:split_index],
        "test": records[split_index:],
    }

    for split_name, items in split_to_items.items():
        if not items:
            continue

        logger.info(f"  Uploading split '{split_name}': {len(items):,} samples")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            subset_metadata_path = temp_path / "metadata.jsonl"
            with open(subset_metadata_path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            for item in items:
                src = Path(item["file_name"])
                dst = temp_path / src.name
                if src.exists():
                    shutil.copy(src, dst)

            try:
                upload_subset_to_hub(
                    repo_id=repo_id,
                    subset_dir=temp_path,
                    config_name="default",
                    split=split_name,
                    reuse_existing_schema=True,
                )
            except Exception as e:
                logger.error(f"  Failed to upload '{split_name}' split: {e}")


def pipeline(
    repo_id: str,
    size: int,
    output_dir: str,
    lang: str,
    template: Optional[str] = None,
    markdown_renderer: str = "pil",
    similar_char_ratio: float = 0.08,
    similarity_db_path: Optional[str] = None,
    add_noise: Optional[bool] = None,
    add_blur: Optional[bool] = None,
    mixed: bool = False,
    train_ratio: float = 0.9,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> None:
    logger.info("=" * 80)
    set_global_seed(seed)
    if mixed:
        logger.info(" Synthetic OCR Dataset Generator (Train/Test Split) ".center(80))
    else:
        logger.info(" Synthetic OCR Dataset Generator ".center(80))
        logger.info(" Format: markdown ".center(80))
    logger.info("=" * 80)

    if size <= 0:
        logger.warning("Requested number of images is 0, terminating.")
        return

    if not (0.0 <= train_ratio <= 1.0 and 0.0 <= test_ratio <= 1.0):
        raise ValueError("train_ratio and test_ratio must be between 0.0 and 1.0")

    if abs((train_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio and test_ratio must sum to 1.0")

    base_dir = Path(output_dir) / lang
    font_dir = Path(f"fonts/{lang}")

    if mixed:
        task_output_dir = base_dir / "images_mixed"

        mixed_gen = MixedGenerator(
            output_dir=str(task_output_dir),
            font_dir=str(font_dir),
            lang=lang,
        )

        generated_dir = mixed_gen.run(
            num_images=size,
            template=template,
            markdown_renderer=markdown_renderer,
            similar_char_ratio=similar_char_ratio,
            similarity_db_path=similarity_db_path,
            add_noise=add_noise,
            add_blur=add_blur,
        )

    else:
        from generator import Generator

        task_output_dir = base_dir / "images_markdown"
        generator = Generator(
            output_dir=str(task_output_dir),
            font_dir=str(font_dir),
            lang=lang,
        )
        generation_kwargs: Dict[str, Any] = {
            "num_images": size,
            "template": template,
            "markdown_renderer": markdown_renderer,
            "similar_char_ratio": similar_char_ratio,
            "similarity_db_path": similarity_db_path,
        }
        if add_noise is not None:
            generation_kwargs["add_noise"] = add_noise
        if add_blur is not None:
            generation_kwargs["add_blur"] = add_blur
        generated_dir = generator.run(**generation_kwargs)

    if generated_dir:
        logger.info(f"\n--- Uploading to Hugging Face Hub: {repo_id} ---")

        if mixed:
            _upload_mixed_format_to_hub(
                repo_id=repo_id,
                output_dir=Path(generated_dir),
                train_ratio=train_ratio,
                test_ratio=test_ratio,
            )
        else:
            try:
                upload_subset_to_hub(
                    repo_id=repo_id,
                    subset_dir=Path(generated_dir),
                    config_name="markdown",
                    reuse_existing_schema=True,
                )
            except Exception as e:
                logger.error(f"Upload failed: {e}", exc_info=True)
    else:
        logger.warning("No dataset was generated, skipping upload.")

    logger.info("\n" + " Pipeline completed! ".center(80, "="))
    logger.info(f"Dataset: https://huggingface.co/datasets/{repo_id}")
