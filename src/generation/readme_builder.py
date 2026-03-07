from datetime import datetime, timezone
from typing import Any, Optional

from generation.git_metadata import resolve_git_metadata


def format_optional_value(value: Any) -> str:
    if value is None:
        return "auto"
    return str(value)


def format_coverage_targets(value: Any) -> str:
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


def build_dataset_readme(
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
    formula_source_mode: str,
    formula_dataset_path: Optional[str],
    formula_dataset_weight: float,
    formula_random_weight: float,
    formula_synthetic_weight: float,
    add_noise: Optional[bool],
    add_blur: Optional[bool],
    mixed: bool,
    train_ratio: float,
    test_ratio: float,
    seed: Optional[int],
    split_counts: dict[str, int],
) -> str:
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    git_meta = resolve_git_metadata()

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
        f"--formula-source-mode {formula_source_mode}",
        f"--formula-dataset-weight {formula_dataset_weight}",
        f"--formula-random-weight {formula_random_weight}",
        f"--formula-synthetic-weight {formula_synthetic_weight}",
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
    if formula_dataset_path:
        generation_command.append(f"--formula-dataset-path {formula_dataset_path}")
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
    command_block = " \\\n+  ".join(generation_command)

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
            f"- Template family: `{format_optional_value(template_family)}`",
            f"- Min template complexity: `{format_optional_value(min_template_complexity)}`",
            f"- Max template complexity: `{format_optional_value(max_template_complexity)}`",
            f"- Template config dir: `{format_optional_value(template_config_dir)}`",
            f"- Markdown renderer: `{markdown_renderer}`",
            f"- Style profile: `{style_profile}`",
            f"- Coverage targets: `{format_coverage_targets(coverage_targets)}`",
            f"- Novelty window: `{novelty_window}`",
            f"- Novelty threshold: `{novelty_threshold}`",
            f"- Novelty max attempts: `{novelty_max_attempts}`",
            f"- Similar char ratio: `{similar_char_ratio}`",
            f"- Similarity DB path: `{format_optional_value(similarity_db_path)}`",
            f"- Formula source mode: `{formula_source_mode}`",
            f"- Formula dataset path: `{format_optional_value(formula_dataset_path)}`",
            f"- Formula source weights (dataset/random/synthetic): `{formula_dataset_weight}/{formula_random_weight}/{formula_synthetic_weight}`",
            f"- Add noise: `{format_optional_value(add_noise)}`",
            f"- Add blur: `{format_optional_value(add_blur)}`",
            f"- Seed: `{format_optional_value(seed)}`",
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
