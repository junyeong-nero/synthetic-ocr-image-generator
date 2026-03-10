import logging
import random
from pathlib import Path
from typing import Any, Optional

from generation.hub_dataset import upload_dataset_readme_to_hub, upload_subset_to_hub
from generation.readme_builder import build_dataset_readme

logger = logging.getLogger(__name__)


def count_metadata_rows(output_dir: Path) -> int:
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return 0

    count = 0
    with open(metadata_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def upload_split_dataset_to_hub(
    repo_id: str,
    output_dir: Path,
    train_ratio: float = 0.9,
    test_ratio: float = 0.1,
) -> dict[str, int]:
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}

    record_count = count_metadata_rows(output_dir)
    if record_count == 0:
        logger.warning("No records found, skipping upload.")
        return {}

    logger.info(
        "Using dataset split ratios: train=%.3f, test=%.3f",
        train_ratio,
        test_ratio,
    )

    shuffled_indices = list(range(record_count))
    random.Random(42).shuffle(shuffled_indices)
    split_index = int(record_count * train_ratio)
    if record_count > 1:
        split_index = min(max(split_index, 1), record_count - 1)
    else:
        split_index = record_count

    split_to_items = {
        "train": set(shuffled_indices[:split_index]),
        "test": set(shuffled_indices[split_index:]),
    }
    uploaded_split_counts: dict[str, int] = {}

    for split_name, selected_indices in split_to_items.items():
        if not selected_indices:
            continue

        logger.info(
            "  Uploading split '%s': %s samples",
            split_name,
            f"{len(selected_indices):,}",
        )

        try:
            upload_subset_to_hub(
                repo_id=repo_id,
                metadata_path=metadata_path,
                config_name="default",
                split=split_name,
                reuse_existing_schema=True,
                selected_indices=selected_indices,
            )
            uploaded_split_counts[split_name] = len(selected_indices)
        except Exception as exc:
            logger.error(f"  Failed to upload '{split_name}' split: {exc}")

    return uploaded_split_counts


def upload_generated_dataset(
    *,
    repo_id: str,
    generated_path: Path,
    train_ratio: float,
    test_ratio: float,
    lang: str,
    size: int,
    template: Optional[str],
    template_family: Optional[str],
    min_template_complexity: Optional[int],
    max_template_complexity: Optional[int],
    template_config_dir: Optional[str],
    markdown_renderer: str,
    style_profile: str,
    coverage_targets: Any,
    novelty_window: int,
    novelty_threshold: float,
    novelty_max_attempts: int,
    similar_char_ratio: float,
    similarity_db_path: Optional[str],
    formula_source_mode: str,
    formula_dataset_path: Optional[str],
    formula_dataset_weight: float,
    formula_random_weight: float,
    formula_synthetic_weight: float,
    add_noise: Optional[bool],
    add_blur: Optional[bool],
    seed: Optional[int],
) -> dict[str, int]:
    generated_count = count_metadata_rows(generated_path)
    uploaded_split_counts = upload_split_dataset_to_hub(
        repo_id=repo_id,
        output_dir=generated_path,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
    )

    if uploaded_split_counts:
        readme_content = build_dataset_readme(
            repo_id=repo_id,
            lang=lang,
            size=size,
            generated_count=generated_count,
            template=template,
            template_family=template_family,
            min_template_complexity=min_template_complexity,
            max_template_complexity=max_template_complexity,
            template_config_dir=template_config_dir,
            markdown_renderer=markdown_renderer,
            style_profile=style_profile,
            coverage_targets=coverage_targets,
            novelty_window=novelty_window,
            novelty_threshold=novelty_threshold,
            novelty_max_attempts=novelty_max_attempts,
            similar_char_ratio=similar_char_ratio,
            similarity_db_path=similarity_db_path,
            formula_source_mode=formula_source_mode,
            formula_dataset_path=formula_dataset_path,
            formula_dataset_weight=formula_dataset_weight,
            formula_random_weight=formula_random_weight,
            formula_synthetic_weight=formula_synthetic_weight,
            add_noise=add_noise,
            add_blur=add_blur,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_counts=uploaded_split_counts,
        )
        upload_dataset_readme_to_hub(
            repo_id=repo_id,
            readme_content=readme_content,
            commit_message="docs: add generation metadata dataset card",
        )

    return uploaded_split_counts
