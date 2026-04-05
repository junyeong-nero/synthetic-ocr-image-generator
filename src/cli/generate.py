from __future__ import annotations

import argparse

from src.generation.options import GenerationOptions, GenerationTaskContext, PublishOptions


def _add_optional_generation_effect_argument(
    parser: argparse.ArgumentParser,
    option_name: str,
    help_text: str,
) -> None:
    parser.add_argument(
        option_name,
        action=argparse.BooleanOptionalAction,
        default=None,
        help=help_text,
    )


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--repo-id",
        required=False,
        default=None,
        help="Hugging Face Hub repository ID used only when upload is enabled",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Base directory for all generated data",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ko",
        help="Language code (e.g., ko, en)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of samples per shard output directory",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Limit generation to the first N planned shards",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume a previously started sharded generation run",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        default=False,
        help="Upload to Hugging Face Hub after generation completes",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Template for generation",
    )
    parser.add_argument(
        "--template-family",
        type=str,
        default=None,
        help="Template family filter (e.g. sections, operations, api)",
    )
    parser.add_argument(
        "--min-template-complexity",
        type=int,
        default=None,
        help="Minimum template complexity filter (1-5)",
    )
    parser.add_argument(
        "--max-template-complexity",
        type=int,
        default=None,
        help="Maximum template complexity filter (1-5)",
    )
    parser.add_argument(
        "--template-config-dir",
        type=str,
        default=None,
        help="Directory containing template YAML configs (default: configs/generator/templates)",
    )
    parser.add_argument(
        "--markdown-renderer",
        type=str,
        default="playwright",
        choices=["pil", "html2image", "playwright"],
        help="Markdown rendering pipeline (pil, html2image, or headless Playwright)",
    )
    parser.add_argument(
        "--style-profile",
        type=str,
        default="balanced",
        choices=["legacy", "balanced", "aggressive"],
        help="Style sampling profile controlling visual variation",
    )
    parser.add_argument(
        "--coverage-target",
        action="append",
        default=None,
        help="Coverage target per family, e.g. sections=0.5 (repeatable)",
    )
    parser.add_argument(
        "--novelty-window",
        type=int,
        default=80,
        help="Recent sample window size used for novelty checks",
    )
    parser.add_argument(
        "--novelty-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for novelty guard (higher means stricter)",
    )
    parser.add_argument(
        "--novelty-max-attempts",
        type=int,
        default=4,
        help="Max attempts per sample before accepting low-novelty output",
    )
    parser.add_argument(
        "--similar-char-ratio",
        type=float,
        default=0.08,
        help="Ratio of characters to replace with similar-looking characters",
    )
    parser.add_argument(
        "--similarity-db-path",
        type=str,
        default=None,
        help="Path to character similarity DB JSON from src/character_similarity.py",
    )
    parser.add_argument(
        "--formula-source-mode",
        type=str,
        default="mixed",
        choices=["mixed", "dataset", "random", "synthetic"],
        help="Formula source strategy: mixed, dataset, random, or synthetic",
    )
    parser.add_argument(
        "--formula-dataset-path",
        type=str,
        default=None,
        help="Path to formula dataset file (.txt/.json/.jsonl/.csv/.tsv)",
    )
    parser.add_argument(
        "--formula-dataset-weight",
        type=float,
        default=0.45,
        help="Formula source weight for dataset entries when --formula-source-mode=mixed",
    )
    parser.add_argument(
        "--formula-random-weight",
        type=float,
        default=0.30,
        help="Formula source weight for random templates when --formula-source-mode=mixed",
    )
    parser.add_argument(
        "--formula-synthetic-weight",
        type=float,
        default=0.25,
        help="Formula source weight for synthetic formulas when --formula-source-mode=mixed",
    )
    _add_optional_generation_effect_argument(
        parser,
        "--add-noise",
        "Enable or disable noise effect (default: generator setting)",
    )
    _add_optional_generation_effect_argument(
        parser,
        "--add-blur",
        "Enable or disable blur effect (default: generator setting)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio for dataset publishing (default: 0.9)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio for dataset publishing (default: 0.1)",
    )
    return parser


def build_context_from_args(args: argparse.Namespace) -> GenerationTaskContext:
    generation = GenerationOptions(
        template=args.template,
        template_family=args.template_family,
        min_template_complexity=args.min_template_complexity,
        max_template_complexity=args.max_template_complexity,
        template_config_dir=args.template_config_dir,
        markdown_renderer=args.markdown_renderer,
        style_profile=args.style_profile,
        coverage_targets=args.coverage_target,
        novelty_window=args.novelty_window,
        novelty_threshold=args.novelty_threshold,
        novelty_max_attempts=args.novelty_max_attempts,
        similar_char_ratio=args.similar_char_ratio,
        similarity_db_path=args.similarity_db_path,
        formula_source_mode=args.formula_source_mode,
        formula_dataset_path=args.formula_dataset_path,
        formula_dataset_weight=args.formula_dataset_weight,
        formula_random_weight=args.formula_random_weight,
        formula_synthetic_weight=args.formula_synthetic_weight,
        add_noise=args.add_noise,
        add_blur=args.add_blur,
        seed=args.seed,
    )
    publish = PublishOptions(
        repo_id=args.repo_id,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
    )
    return GenerationTaskContext(
        lang=args.lang,
        size=args.size,
        generation=generation,
        publish=publish,
    )


def run_with_args(args: argparse.Namespace) -> None:
    from src.pipeline import pipeline

    pipeline(
        context=build_context_from_args(args),
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        max_shards=args.max_shards,
        resume=args.resume,
        upload=args.upload,
    )
