"""Main entry point for generation and evaluation pipelines."""

import sys
import asyncio
import argparse
import os
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

sys.path.insert(0, "src")
from cli.evaluation_outputs import persist_evaluation_outputs
from env_utils import load_env_file, set_global_seed

load_env_file()

if TYPE_CHECKING:
    from evaluation.model_config import ModelSpecificConfig


def get_api_key(backend: str) -> Optional[str]:
    """Get API key from environment variable."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "upstage": "UPSTAGE_API_KEY",
    }
    env_var = key_map.get(backend)
    return os.environ.get(env_var) if env_var else None


def _resolve_execution_mode(args: argparse.Namespace):
    from evaluation.config import EvaluationMode

    if args.inference_only:
        return EvaluationMode.INFERENCE_ONLY
    if args.evaluate_only:
        return EvaluationMode.EVALUATE_ONLY
    return EvaluationMode.ALL


def _resolve_evaluation_runtime(
    args: argparse.Namespace,
    model_specific_config,
) -> dict[str, Any]:
    backend_str = args.backend or model_specific_config.backend
    runtime = {
        "backend": backend_str,
        "temperature": model_specific_config.get_temperature(),
        "max_tokens": model_specific_config.get_max_tokens(),
        "batch_size": model_specific_config.get_batch_size(),
        "tensor_parallel_size": model_specific_config.tensor_parallel_size,
        "api_base": model_specific_config.api_base,
        "timeout": model_specific_config.timeout,
        "max_retries": model_specific_config.max_retries,
        "device": model_specific_config.device,
        "dtype": model_specific_config.dtype,
        "rate_limit_rpm": model_specific_config.rate_limit_rpm,
    }

    cli_overrides = {
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "batch_size": "batch_size",
        "tensor_parallel": "tensor_parallel_size",
        "api_base": "api_base",
    }
    for arg_key, runtime_key in cli_overrides.items():
        value = getattr(args, arg_key, None)
        if value is not None:
            runtime[runtime_key] = value

    return runtime


def _build_evaluation_config(
    args: argparse.Namespace,
    model_specific_config,
):
    from evaluation.config import InferenceBackend, ModelConfig, EvaluationConfig

    runtime = _resolve_evaluation_runtime(args, model_specific_config)
    backend_str = str(runtime["backend"])
    model_config = ModelConfig(
        model_id=model_specific_config.get_model_id(),
        backend=InferenceBackend(backend_str),
        api_key=get_api_key(backend_str),
        api_base=runtime["api_base"],
        tensor_parallel_size=runtime["tensor_parallel_size"],
        temperature=runtime["temperature"],
        max_tokens=runtime["max_tokens"],
        timeout=runtime["timeout"],
        max_retries=runtime["max_retries"],
        device=runtime["device"],
        dtype=runtime["dtype"],
        rate_limit_rpm=runtime["rate_limit_rpm"],
    )

    execution_mode = _resolve_execution_mode(args)
    config = EvaluationConfig(
        dataset_id=args.dataset,
        split=args.split,
        language=args.language,
        model=model_config,
        batch_size=runtime["batch_size"],
        max_samples=args.max_samples,
        output_dir=str(args.output_dir),
        seed=args.seed,
        batch_api=args.batch_api,
        batch_poll_seconds=args.batch_poll_seconds,
        batch_timeout_seconds=args.batch_timeout_seconds,
        batch_completion_window=args.batch_completion_window,
        execution_mode=execution_mode,
        model_config_path=args.model_config,
    )

    return config, model_config, runtime, execution_mode


def _save_evaluation_reports(output, output_path: Path, report_format: str) -> None:
    from evaluation.report import ReportGenerator

    generator = ReportGenerator(output)
    if report_format == "all":
        paths = generator.save_all(output_path)
        print("\nReports saved:")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")
        return

    method = getattr(generator, f"to_{report_format}")
    path = method(output_path / f"report.{report_format}")
    print(f"\nReport saved: {path}")


def print_results(metrics: dict, format_name: str) -> None:
    """Print evaluation results to console."""
    print("\n" + "=" * 60)
    print(f" {format_name.upper()} EVALUATION RESULTS ")
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("=" * 60)


def load_model_config(
    model_config_path: str,
) -> "ModelSpecificConfig":
    """Load model-specific configuration."""
    from evaluation.model_config import ModelConfigLoader

    loader = ModelConfigLoader()
    return loader.load_from_path(Path(model_config_path))


GENERATE_ARG_TO_PIPELINE_KEY: tuple[tuple[str, str], ...] = (
    ("repo_id", "repo_id"),
    ("output_dir", "output_dir"),
    ("lang", "lang"),
    ("size", "size"),
    ("template", "template"),
    ("template_family", "template_family"),
    ("min_template_complexity", "min_template_complexity"),
    ("max_template_complexity", "max_template_complexity"),
    ("template_config_dir", "template_config_dir"),
    ("markdown_renderer", "markdown_renderer"),
    ("style_profile", "style_profile"),
    ("coverage_target", "coverage_targets"),
    ("novelty_window", "novelty_window"),
    ("novelty_threshold", "novelty_threshold"),
    ("novelty_max_attempts", "novelty_max_attempts"),
    ("similar_char_ratio", "similar_char_ratio"),
    ("similarity_db_path", "similarity_db_path"),
    ("formula_source_mode", "formula_source_mode"),
    ("formula_dataset_path", "formula_dataset_path"),
    ("formula_dataset_weight", "formula_dataset_weight"),
    ("formula_random_weight", "formula_random_weight"),
    ("formula_synthetic_weight", "formula_synthetic_weight"),
    ("add_noise", "add_noise"),
    ("add_blur", "add_blur"),
    ("mixed", "mixed"),
    ("train_ratio", "train_ratio"),
    ("test_ratio", "test_ratio"),
    ("seed", "seed"),
    ("shard_size", "shard_size"),
    ("max_shards", "max_shards"),
    ("resume", "resume"),
)


def _build_generate_pipeline_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        pipeline_key: getattr(args, arg_name)
        for arg_name, pipeline_key in GENERATE_ARG_TO_PIPELINE_KEY
    }


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


def _configure_generate_parser(gen_parser: argparse.ArgumentParser) -> None:
    gen_parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face Hub repository ID for dataset upload",
    )
    gen_parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Base directory for all generated data",
    )
    gen_parser.add_argument(
        "--lang",
        type=str,
        default="ko",
        help="Language code (e.g., ko, en)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    gen_parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Number of images to generate",
    )
    gen_parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of samples per shard output directory",
    )
    gen_parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Limit generation to the first N planned shards",
    )
    gen_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume a previously started sharded generation run",
    )
    gen_parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Template for generation",
    )
    gen_parser.add_argument(
        "--template-family",
        type=str,
        default=None,
        help="Template family filter (e.g. sections, operations, api)",
    )
    gen_parser.add_argument(
        "--min-template-complexity",
        type=int,
        default=None,
        help="Minimum template complexity filter (1-5)",
    )
    gen_parser.add_argument(
        "--max-template-complexity",
        type=int,
        default=None,
        help="Maximum template complexity filter (1-5)",
    )
    gen_parser.add_argument(
        "--template-config-dir",
        type=str,
        default=None,
        help="Directory containing template YAML configs (default: configs/generator/templates)",
    )
    gen_parser.add_argument(
        "--markdown-renderer",
        type=str,
        default="pil",
        choices=["pil", "html2image"],
        help="Markdown rendering pipeline (pil or markdown->html->image via html2image)",
    )
    gen_parser.add_argument(
        "--style-profile",
        type=str,
        default="balanced",
        choices=["legacy", "balanced", "aggressive"],
        help="Style sampling profile controlling visual variation",
    )
    gen_parser.add_argument(
        "--coverage-target",
        action="append",
        default=None,
        help="Coverage target per family, e.g. sections=0.5 (repeatable)",
    )
    gen_parser.add_argument(
        "--novelty-window",
        type=int,
        default=80,
        help="Recent sample window size used for novelty checks",
    )
    gen_parser.add_argument(
        "--novelty-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for novelty guard (higher means stricter)",
    )
    gen_parser.add_argument(
        "--novelty-max-attempts",
        type=int,
        default=4,
        help="Max attempts per sample before accepting low-novelty output",
    )
    gen_parser.add_argument(
        "--similar-char-ratio",
        type=float,
        default=0.08,
        help="Ratio of characters to replace with similar-looking characters",
    )
    gen_parser.add_argument(
        "--similarity-db-path",
        type=str,
        default=None,
        help="Path to character similarity DB JSON from src/character_similarity.py",
    )
    gen_parser.add_argument(
        "--formula-source-mode",
        type=str,
        default="mixed",
        choices=["mixed", "dataset", "random", "synthetic"],
        help="Formula source strategy: mixed, dataset, random, or synthetic",
    )
    gen_parser.add_argument(
        "--formula-dataset-path",
        type=str,
        default=None,
        help="Path to formula dataset file (.txt/.json/.jsonl/.csv/.tsv)",
    )
    gen_parser.add_argument(
        "--formula-dataset-weight",
        type=float,
        default=0.45,
        help="Formula source weight for dataset entries in mixed mode",
    )
    gen_parser.add_argument(
        "--formula-random-weight",
        type=float,
        default=0.30,
        help="Formula source weight for random templates in mixed mode",
    )
    gen_parser.add_argument(
        "--formula-synthetic-weight",
        type=float,
        default=0.25,
        help="Formula source weight for synthetic formulas in mixed mode",
    )
    _add_optional_generation_effect_argument(
        gen_parser,
        "--add-noise",
        "Enable or disable noise effect (default: generator setting)",
    )
    _add_optional_generation_effect_argument(
        gen_parser,
        "--add-blur",
        "Enable or disable blur effect (default: generator setting)",
    )
    gen_parser.add_argument(
        "--mixed",
        action="store_true",
        default=False,
        help="Generate mixed format dataset",
    )
    gen_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio in mixed mode (default: 0.9)",
    )
    gen_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio in mixed mode (default: 0.1)",
    )


def cmd_generate(args: argparse.Namespace) -> None:
    """Run generation command."""
    from pipeline import pipeline

    set_global_seed(args.seed)
    pipeline_args = _build_generate_pipeline_args(args)
    pipeline(**pipeline_args)


async def cmd_corpus_generate(args: argparse.Namespace) -> int:
    """Run corpus generation command."""
    from corpus_llm.cli import run_with_args

    return await run_with_args(args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation command."""
    from evaluation.config import EvaluationMode
    from evaluation.pipeline import EvaluationPipeline

    model_specific_config = load_model_config(args.model_config)

    set_global_seed(args.seed)

    representative_metrics = {
        "markdown": "avg_markdown_overall_score",
    }

    config, model_config, runtime, execution_mode = _build_evaluation_config(
        args,
        model_specific_config,
    )
    backend_str = str(runtime["backend"])

    print("\nConfiguration:")
    print(f"  Model: {model_config.model_id}")
    print(f"  Backend: {backend_str}")
    print(f"  Dataset: {args.dataset} ({args.split})")
    print(f"  Language: {args.language}")
    print("  Format: markdown (fixed)")
    print(f"  Batch Size: {runtime['batch_size']}")
    print(f"  Temperature: {runtime['temperature']}")
    print(f"  Max Tokens: {runtime['max_tokens']}")
    print(f"  Config File: {args.model_config}")
    print(f"  Mode: {execution_mode.value}")

    pipeline = EvaluationPipeline(config)
    if execution_mode == EvaluationMode.INFERENCE_ONLY:
        print(f"\nRunning inference only for {model_config.model_id} on {args.dataset}...")
        results = pipeline.run_inference_only()
        checkpoint_path = Path(args.output_dir) / "checkpoints.json"
        print(f"Inference complete: {len(results)} samples processed")
        print(f"Checkpoint saved: {checkpoint_path}")
        return

    if execution_mode == EvaluationMode.EVALUATE_ONLY:
        print(f"\nEvaluating from checkpoint for {model_config.model_id} on {args.dataset}...")
        output = pipeline.run_evaluate_only()
    else:
        print(f"\nEvaluating {model_config.model_id} on {args.dataset}...")
        output = pipeline.run()

    output_path = Path(args.output_dir)
    _save_evaluation_reports(output, output_path, args.report_format)

    resolved_format = str(output.config.get("format") or "markdown")
    metric_key = representative_metrics.get(resolved_format)
    metric_value = output.metrics.get(metric_key) if metric_key else None

    protocol_path, summary_path = persist_evaluation_outputs(
        output_dir=output_path,
        output=output,
        report_format=args.report_format,
        model_id=model_config.model_id,
        backend=backend_str,
        dataset=args.dataset,
        split=args.split,
        language=args.language,
        seed=args.seed,
        resolved_format=resolved_format,
        metric_key=metric_key,
        metric_value=metric_value,
    )
    print(f"Protocol snapshot saved: {protocol_path}")

    print_results(output.metrics, resolved_format)
    print(f"\nSamples: {output.summary['successful']}/{output.summary['total_samples']}")
    print(f"Avg Latency: {output.summary['avg_latency_ms']:.2f}ms")

    if metric_key and metric_value is None:
        print(f"Warning: Missing representative metric '{metric_key}' for {resolved_format}")

    print(f"\nModel summary saved: {summary_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Run comparison command."""
    from evaluation.comparator import ModelComparator

    paths = [Path(p) for p in args.report_files]

    for path in paths:
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    comparator = ModelComparator.from_json_files(paths)

    comparator.save_comparison(Path(args.output))

    comparator.print_summary()

    print(f"\nComparison saved to: {args.output}.json, {args.output}.md")


def cmd_list_backends(args: argparse.Namespace) -> None:
    """List available backends."""
    from models.registry import list_backends

    print("\nAvailable inference backends:")
    print("-" * 40)
    for name, description in list_backends().items():
        print(f"  {name:12} - {description}")
    print()


def cmd_list_configs(args: argparse.Namespace) -> None:
    """List available model configurations."""
    from evaluation.model_config import ModelConfigLoader

    loader = ModelConfigLoader()
    configs = loader.list_available_configs()

    print("\nAvailable model configurations:")
    print("-" * 70)
    print("  {:<30} {:<20} {}".format("Config Name", "Dependency Group", "Backend"))
    print("-" * 70)
    if configs:
        for config_name in configs:
            if config_name.startswith("_"):
                continue
            try:
                config = loader.load_from_path(
                    Path("configs/models") / f"{config_name}.yaml"
                )
                dep_group = config.dependency_group or "-"
                backend = config.backend
            except Exception:
                dep_group = "?"
                backend = "?"
            print(f"  {config_name:<30} {dep_group:<20} {backend}")
    else:
        print("  No model configs found in configs/models/")
    print("-" * 70)
    print("\nConfig search paths:")
    for path in loader.config_dirs:
        exists = "+" if path.exists() else "-"
        print(f"  {exists} {path}")
    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synthetic OCR Image Generator & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    _configure_generate_parser(gen_parser)

    # Corpus command
    corpus_parser = subparsers.add_parser("corpus", help="Corpus generation commands")
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_command")
    corpus_subparsers.required = True

    corpus_generate_parser = corpus_subparsers.add_parser(
        "generate",
        help="Generate corpus data using LLM",
    )
    from corpus_llm.cli import add_arguments as add_corpus_arguments

    add_corpus_arguments(corpus_generate_parser)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    eval_parser.add_argument(
        "--model-config",
        required=True,
        help="Path to model-specific config YAML",
    )
    eval_parser.add_argument(
        "-d", "--dataset", required=True, help="Hugging Face dataset ID or local path"
    )
    eval_parser.add_argument(
        "-b",
        "--backend",
        choices=[
                "openai",
                "anthropic",
                "google",
                "upstage",
                "transformers",
                "paddleocr",
                "surya",
        ],
        help="Inference backend (optional if model config exists)",
    )
    eval_parser.add_argument("--split", default="train", help="Dataset split")
    eval_parser.add_argument(
        "--language",
        "-l",
        default="ko",
        help="Dataset language code",
    )
    eval_parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to evaluate"
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible evaluation",
    )
    eval_parser.add_argument(
        "--batch-api",
        action="store_true",
        default=False,
        help="Use OpenAI Batch API for evaluation",
    )
    eval_parser.add_argument(
        "--batch-poll-seconds",
        type=int,
        default=60,
        help="Polling interval for batch status",
    )
    eval_parser.add_argument(
        "--batch-timeout-seconds",
        type=int,
        default=86400,
        help="Max wait time for batch completion",
    )
    eval_parser.add_argument(
        "--batch-completion-window",
        default="24h",
        help="Batch completion window",
    )
    eval_parser.add_argument(
        "--output-dir", default="./evaluation_result", help="Output directory"
    )
    eval_parser.add_argument(
        "--report-format",
        default="all",
        choices=["json", "markdown", "html", "all"],
        help="Report output format",
    )
    mode_group = eval_parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--inference-only",
        action="store_true",
        default=False,
        help="Run model inference only and save checkpoints.json without report generation",
    )
    mode_group.add_argument(
        "--evaluate-only",
        action="store_true",
        default=False,
        help="Skip model inference and compute reports from checkpoints.json",
    )

    # Evaluation CLI overrides
    override_group = eval_parser.add_argument_group(
        "config overrides", "Override values from model config"
    )
    override_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides model config)",
    )
    override_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature",
    )
    override_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens",
    )
    override_group.add_argument(
        "--api-base", default=None, help="Custom API base URL"
    )
    override_group.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
        help="Tensor parallel size",
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare evaluation results")
    compare_parser.add_argument(
        "report_files", nargs="+", help="JSON report files to compare"
    )
    compare_parser.add_argument(
        "-o", "--output", default="comparison", help="Output file prefix"
    )

    # List backends command
    subparsers.add_parser("list-backends", help="List available backends")

    # List configs command
    subparsers.add_parser("list-configs", help="List available model configurations")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "corpus" and args.corpus_command == "generate":
        sys.exit(asyncio.run(cmd_corpus_generate(args)))
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "list-backends":
        cmd_list_backends(args)
    elif args.command == "list-configs":
        cmd_list_configs(args)
    else:
        parser.print_help()
        sys.exit(1)



if __name__ == "__main__":
    main()
