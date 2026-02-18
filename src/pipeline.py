import json
import logging
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from env_utils import set_global_seed
from utils import markdown_to_json_ast, upload_dataset_readme_to_hub, upload_subset_to_hub
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


def _count_metadata_rows(output_dir: Path) -> int:
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return 0

    count = 0
    with open(metadata_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def _format_optional_value(value: Any) -> str:
    if value is None:
        return "auto"
    return str(value)


def _format_coverage_targets(value: Any) -> str:
    if value is None:
        return "auto"
    if isinstance(value, dict):
        if not value:
            return "auto"
        return ", ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value if str(item).strip()]
        return ", ".join(items) if items else "auto"
    text = str(value).strip()
    return text or "auto"


def _build_generation_kwargs(
    *,
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
    seed: Optional[int],
    add_noise: Optional[bool],
    add_blur: Optional[bool],
) -> Dict[str, Any]:
    generation_kwargs: Dict[str, Any] = {
        "template": template,
        "template_family": template_family,
        "min_template_complexity": min_template_complexity,
        "max_template_complexity": max_template_complexity,
        "template_config_dir": template_config_dir,
        "markdown_renderer": markdown_renderer,
        "style_profile": style_profile,
        "coverage_targets": coverage_targets,
        "novelty_window": novelty_window,
        "novelty_threshold": novelty_threshold,
        "novelty_max_attempts": novelty_max_attempts,
        "similar_char_ratio": similar_char_ratio,
        "similarity_db_path": similarity_db_path,
        "seed": seed,
    }
    if add_noise is not None:
        generation_kwargs["add_noise"] = add_noise
    if add_blur is not None:
        generation_kwargs["add_blur"] = add_blur
    return generation_kwargs


def _run_git_command(args: List[str]) -> str:
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _normalize_github_url(raw_url: str) -> str:
    url = raw_url.strip()
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split("git@github.com:", 1)[1]
    elif url.startswith("ssh://git@github.com/"):
        url = "https://github.com/" + url.split("ssh://git@github.com/", 1)[1]
    elif url.startswith("http://github.com/"):
        url = "https://github.com/" + url.split("http://github.com/", 1)[1]

    if url.endswith(".git"):
        url = url[:-4]
    return url.rstrip("/")


def _resolve_git_metadata() -> Dict[str, str]:
    remote_url = _run_git_command(["remote", "get-url", "origin"])
    return {
        "github_url": _normalize_github_url(remote_url) if remote_url else "",
        "commit": _run_git_command(["rev-parse", "HEAD"]),
        "branch": _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _build_dataset_readme(
    repo_id: str,
    lang: str,
    size: int,
    generated_count: int,
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
    add_noise: Optional[bool],
    add_blur: Optional[bool],
    mixed: bool,
    train_ratio: float,
    test_ratio: float,
    seed: Optional[int],
    split_counts: Dict[str, int],
) -> str:
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    git_meta = _resolve_git_metadata()

    split_lines = []
    for split_name in ["train", "test"]:
        if split_name in split_counts:
            split_lines.append(f"- `{split_name}`: {split_counts[split_name]:,} samples")

    if not split_lines and generated_count > 0:
        split_lines.append(f"- `train`: {generated_count:,} samples")

    split_block = "\n".join(split_lines) if split_lines else "- No split summary available"
    mode = "mixed" if mixed else "markdown"

    generation_command = [
        "uv run main.py generate",
        f'--repo-id "{repo_id}"',
        f"--lang {lang}",
        f"--size {size}",
        f"--markdown-renderer {markdown_renderer}",
        f"--style-profile {style_profile}",
        f"--similar-char-ratio {similar_char_ratio}",
        f"--novelty-window {novelty_window}",
        f"--novelty-threshold {novelty_threshold}",
        f"--novelty-max-attempts {novelty_max_attempts}",
    ]
    if template:
        generation_command.append(f"--template {template}")
    if template_family:
        generation_command.append(f"--template-family {template_family}")
    if min_template_complexity is not None:
        generation_command.append(f"--min-template-complexity {min_template_complexity}")
    if max_template_complexity is not None:
        generation_command.append(f"--max-template-complexity {max_template_complexity}")
    if template_config_dir:
        generation_command.append(f"--template-config-dir {template_config_dir}")
    if coverage_targets:
        if isinstance(coverage_targets, dict):
            for family, ratio in coverage_targets.items():
                generation_command.append(f"--coverage-target {family}={ratio}")
        elif isinstance(coverage_targets, (list, tuple, set)):
            for item in coverage_targets:
                generation_command.append(f"--coverage-target {item}")
        else:
            generation_command.append(f"--coverage-target {coverage_targets}")
    if similarity_db_path:
        generation_command.append(f"--similarity-db-path {similarity_db_path}")
    if add_noise is not None:
        generation_command.append("--add-noise" if add_noise else "--no-add-noise")
    if add_blur is not None:
        generation_command.append("--add-blur" if add_blur else "--no-add-blur")
    if mixed:
        generation_command.extend([
            "--mixed",
            f"--train-ratio {train_ratio}",
            f"--test-ratio {test_ratio}",
        ])
    if seed is not None:
        generation_command.append(f"--seed {seed}")
    command_block = " \\\n  ".join(generation_command)

    github_line = (
        f"- GitHub: [{git_meta['github_url']}]({git_meta['github_url']})"
        if git_meta["github_url"]
        else "- GitHub: unavailable"
    )

    commit_line = f"- Commit: `{git_meta['commit']}`" if git_meta["commit"] else "- Commit: unavailable"
    branch_line = f"- Branch: `{git_meta['branch']}`" if git_meta["branch"] else "- Branch: unavailable"

    return "\n".join(
        [
            "---",
            "language:",
            f"- {lang}",
            "tags:",
            "- ocr",
            "- synthetic",
            "- markdown",
            "---",
            "",
            "# Synthetic OCR Dataset",
            "",
            "This dataset is generated by the Synthetic OCR Image Generator pipeline and uploaded automatically.",
            "",
            "## Dataset Links",
            "",
            f"- Hugging Face dataset: [https://huggingface.co/datasets/{repo_id}](https://huggingface.co/datasets/{repo_id})",
            github_line,
            commit_line,
            branch_line,
            "",
            "## Generation Metadata",
            "",
            f"- Generated at (UTC): `{now_utc}`",
            f"- Mode: `{mode}`",
            f"- Language: `{lang}`",
            f"- Requested sample size: `{size}`",
            f"- Generated sample count: `{generated_count}`",
            f"- Template: `{template or 'random'}`",
            f"- Template family: `{_format_optional_value(template_family)}`",
            f"- Min template complexity: `{_format_optional_value(min_template_complexity)}`",
            f"- Max template complexity: `{_format_optional_value(max_template_complexity)}`",
            f"- Template config dir: `{_format_optional_value(template_config_dir)}`",
            f"- Markdown renderer: `{markdown_renderer}`",
            f"- Style profile: `{style_profile}`",
            f"- Coverage targets: `{_format_coverage_targets(coverage_targets)}`",
            f"- Novelty window: `{novelty_window}`",
            f"- Novelty threshold: `{novelty_threshold}`",
            f"- Novelty max attempts: `{novelty_max_attempts}`",
            f"- Similar char ratio: `{similar_char_ratio}`",
            f"- Similarity DB path: `{_format_optional_value(similarity_db_path)}`",
            f"- Add noise: `{_format_optional_value(add_noise)}`",
            f"- Add blur: `{_format_optional_value(add_blur)}`",
            f"- Seed: `{_format_optional_value(seed)}`",
            f"- Train ratio: `{train_ratio}`",
            f"- Test ratio: `{test_ratio}`",
            "",
            "## Uploaded Splits",
            "",
            split_block,
            "",
            "## Reproducible Command",
            "",
            "```bash",
            command_block,
            "```",
        ]
    )


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
        template_family: Optional[str] = None,
        min_template_complexity: Optional[int] = None,
        max_template_complexity: Optional[int] = None,
        template_config_dir: Optional[str] = None,
        markdown_renderer: str = "pil",
        style_profile: str = "balanced",
        coverage_targets: Any = None,
        novelty_window: int = 80,
        novelty_threshold: float = 0.95,
        novelty_max_attempts: int = 4,
        similar_char_ratio: float = 0.08,
        similarity_db_path: Optional[str] = None,
        add_noise: Optional[bool] = None,
        add_blur: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> Optional[str]:
        all_metadata: List[Dict[str, Any]] = []

        try:
            logger.info(f"Starting markdown generation: {num_images:,} images")

            generation_kwargs = _build_generation_kwargs(
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
                seed=seed,
                add_noise=add_noise,
                add_blur=add_blur,
            )

            self.markdown_generator._configure_generation(**generation_kwargs)

            for idx in tqdm(range(num_images), desc="Generating markdown images"):
                image, meta = self.markdown_generator.generate_single(sample_index=idx)
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
)-> Dict[str, int]:
    import shutil
    import tempfile

    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}

    records: List[Dict[str, Any]] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        logger.warning("No records found, skipping upload.")
        return {}

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
    uploaded_split_counts: Dict[str, int] = {}

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
                uploaded_split_counts[split_name] = len(items)
            except Exception as e:
                logger.error(f"  Failed to upload '{split_name}' split: {e}")

    return uploaded_split_counts


def pipeline(
    repo_id: str,
    size: int,
    output_dir: str,
    lang: str,
    template: Optional[str] = None,
    template_family: Optional[str] = None,
    min_template_complexity: Optional[int] = None,
    max_template_complexity: Optional[int] = None,
    template_config_dir: Optional[str] = None,
    markdown_renderer: str = "pil",
    style_profile: str = "balanced",
    coverage_targets: Any = None,
    novelty_window: int = 80,
    novelty_threshold: float = 0.95,
    novelty_max_attempts: int = 4,
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
            add_noise=add_noise,
            add_blur=add_blur,
            seed=seed,
        )

    else:
        from generator import Generator

        task_output_dir = base_dir / "images_markdown"
        generator = Generator(
            output_dir=str(task_output_dir),
            font_dir=str(font_dir),
            lang=lang,
        )
        generation_kwargs = _build_generation_kwargs(
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
            seed=seed,
            add_noise=add_noise,
            add_blur=add_blur,
        )
        generation_kwargs["num_images"] = size
        generated_dir = generator.run(**generation_kwargs)

    if generated_dir:
        logger.info(f"\n--- Uploading to Hugging Face Hub: {repo_id} ---")

        generated_path = Path(generated_dir)
        generated_count = _count_metadata_rows(generated_path)
        uploaded_split_counts: Dict[str, int] = {}

        if mixed:
            uploaded_split_counts = _upload_mixed_format_to_hub(
                repo_id=repo_id,
                output_dir=generated_path,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
            )
        else:
            try:
                upload_subset_to_hub(
                    repo_id=repo_id,
                    subset_dir=generated_path,
                    config_name="markdown",
                    reuse_existing_schema=True,
                )
                uploaded_split_counts["train"] = generated_count
            except Exception as e:
                logger.error(f"Upload failed: {e}", exc_info=True)

        if uploaded_split_counts:
            readme_content = _build_dataset_readme(
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
                add_noise=add_noise,
                add_blur=add_blur,
                mixed=mixed,
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
    else:
        logger.warning("No dataset was generated, skipping upload.")

    logger.info("\n" + " Pipeline completed! ".center(80, "="))
    logger.info(f"Dataset: https://huggingface.co/datasets/{repo_id}")
