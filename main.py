"""Main entry point for generation and evaluation pipelines."""

import sys
import argparse
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, "src")
from pipeline import pipeline
from evaluation.config import (
    EvaluationConfig,
    FormatType,
    InferenceBackend,
    ModelConfig,
)
from evaluation.model_config import ModelConfigLoader, ModelSpecificConfig
from evaluation.pipeline import EvaluationPipeline
from evaluation.report import ReportGenerator
from evaluation.comparator import ModelComparator


def get_api_key(backend: str) -> Optional[str]:
    """Get API key from environment variable."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = key_map.get(backend)
    return os.environ.get(env_var) if env_var else None


def print_results(metrics: dict, format_type: str) -> None:
    """Print evaluation results to console."""
    print("\n" + "=" * 60)
    print(f" {format_type.upper()} EVALUATION RESULTS ")
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("=" * 60)


def load_model_config(
    model_config_path: str,
) -> ModelSpecificConfig:
    """Load model-specific configuration."""
    loader = ModelConfigLoader()
    return loader.load_from_path(Path(model_config_path))


def cmd_generate(args: argparse.Namespace) -> None:
    """Run generation command."""
    pipeline_args = {
        "repo_id": args.repo_id,
        "font_path": args.font_path,
        "output_dir": args.output_dir,
        "lang": args.lang,
        "corpus_size": args.corpus_size,
        "size": args.size,
        "typo_ratio": args.typo_ratio,
        "format": args.format,
        "template": args.template,
        "table_size": args.table_size,
        "mixed": args.mixed,
    }
    pipeline(**pipeline_args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation command."""
    model_specific_config = load_model_config(args.model_config)

    backend_str = args.backend
    if not backend_str:
        backend_str = model_specific_config.backend

    subset = args.subset
    
    # Try to map subset to FormatType, defaulting to SENTENCE if not found
    try:
        format_type = FormatType(subset)
    except ValueError:
        # If subset name doesn't match a format type (e.g. "korean_sentence"),
        # we might need a heuristic or just default to sentence.
        # For now, let's assume subset name IS the format type or contains it.
        if "table" in subset:
            format_type = FormatType.TABLE
        elif "document" in subset:
            format_type = FormatType.DOCUMENT
        elif "markdown" in subset:
            format_type = FormatType.MARKDOWN
        elif "kie" in subset:
            format_type = FormatType.KIE
        else:
            format_type = FormatType.SENTENCE

    temperature = model_specific_config.get_temperature(subset)
    max_tokens = model_specific_config.get_max_tokens(subset)
    batch_size = model_specific_config.get_batch_size(subset)
    tensor_parallel = model_specific_config.tensor_parallel_size
    api_base = model_specific_config.api_base
    timeout = model_specific_config.timeout
    max_retries = model_specific_config.max_retries
    device = model_specific_config.device
    dtype = model_specific_config.dtype
    rate_limit_rpm = model_specific_config.rate_limit_rpm

    if hasattr(args, "temperature") and args.temperature is not None:
        temperature = args.temperature
    if hasattr(args, "max_tokens") and args.max_tokens is not None:
        max_tokens = args.max_tokens
    if hasattr(args, "batch_size") and args.batch_size is not None:
        batch_size = args.batch_size
    if hasattr(args, "tensor_parallel") and args.tensor_parallel is not None:
        tensor_parallel = args.tensor_parallel
    if hasattr(args, "api_base") and args.api_base is not None:
        api_base = args.api_base

    model_config = ModelConfig(
        model_id=model_specific_config.get_model_id(subset),
        backend=InferenceBackend(backend_str),
        api_key=get_api_key(backend_str),
        api_base=api_base,
        tensor_parallel_size=tensor_parallel,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        device=device,
        dtype=dtype,
        rate_limit_rpm=rate_limit_rpm,
    )

    config = EvaluationConfig(
        dataset_id=args.dataset,
        subset=subset,
        split=args.split,
        format_type=format_type,
        model=model_config,
        batch_size=batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        model_config_path=args.model_config,
    )

    print(f"\nConfiguration:")
    print(f"  Model: {model_config.model_id}")
    print(f"  Backend: {backend_str}")
    print(f"  Dataset: {args.dataset} ({subset}/{args.split})")
    print(f"  Format: {format_type.value}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Temperature: {temperature}")
    print(f"  Max Tokens: {max_tokens}")
    print(f"  Config File: {args.model_config}")

    print(f"\nEvaluating {model_config.model_id} on {args.dataset}...")
    pipeline = EvaluationPipeline(config)
    output = pipeline.run()

    generator = ReportGenerator(output)
    output_path = Path(args.output_dir)

    if args.report_format == "all":
        paths = generator.save_all(output_path)
        print("\nReports saved:")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")
    else:
        method = getattr(generator, f"to_{args.report_format}")
        path = method(output_path / f"report.{args.report_format}")
        print(f"\nReport saved: {path}")

    print_results(output.metrics, format_type.value)

    print(
        f"\nSamples: {output.summary['successful']}/{output.summary['total_samples']}"
    )
    print(f"Avg Latency: {output.summary['avg_latency_ms']:.2f}ms")


def cmd_compare(args: argparse.Namespace) -> None:
    """Run comparison command."""
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
    loader = ModelConfigLoader()
    configs = loader.list_available_configs()

    print("\nAvailable model configurations:")
    print("-" * 70)
    print(f"  {'Config Name':<30} {'Dependency Group':<20} {'Backend'}")
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

    # Generate command (legacy main arguments)
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen_parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face Hub repository ID for dataset upload",
    )
    gen_parser.add_argument(
        "--font-path",
        type=str,
        required=True,
        help="Font file path for character similarity DB generation",
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
        "--corpus-size",
        type=int,
        default=10000,
        help="Number of sentences to extract from Wikipedia",
    )
    gen_parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Number of images to generate",
    )
    gen_parser.add_argument(
        "--typo-ratio",
        type=float,
        default=0.15,
        help="Ratio of words to introduce typos",
    )
    gen_parser.add_argument(
        "--format",
        type=str,
        default="sentence",
        choices=["sentence", "table", "document", "markdown", "kie"],
        help="Format of images to generate",
    )
    gen_parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Template for generation",
    )
    gen_parser.add_argument(
        "--table-size",
        type=str,
        default="3-8",
        help="Table size range as 'min_rows-max_cols'",
    )
    gen_parser.add_argument(
        "--mixed",
        action="store_true",
        default=False,
        help="Generate mixed format dataset",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    eval_parser.add_argument(
        "--model-config",
        required=True,
        help="Path to model-specific config YAML",
    )
    eval_parser.add_argument(
        "-b",
        "--backend",
        choices=[
            "openai",
            "anthropic",
            "google",
            "transformers",
            "paddleocr",
        ],
        help="Inference backend (optional if model config exists)",
    )
    eval_parser.add_argument(
        "-s", "--subset", default="default", help="Dataset subset (and format type)"
    )
    eval_parser.add_argument("--split", default="test", help="Dataset split")
    eval_parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to evaluate"
    )
    eval_parser.add_argument(
        "--output-dir", default="./evaluation_results", help="Output directory"
    )
    eval_parser.add_argument(
        "--report-format",
        default="all",
        choices=["json", "markdown", "html", "all"],
        help="Report output format",
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
