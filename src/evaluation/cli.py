"""Command-line interface for evaluation pipeline."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

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
    model_id: str,
    model_config_path: Optional[str] = None,
) -> Optional[ModelSpecificConfig]:
    """Load model-specific configuration."""
    loader = ModelConfigLoader()

    if model_config_path:
        return loader.load_from_path(Path(model_config_path))

    return loader.load(model_id)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation command."""
    # Load model-specific config if available
    model_specific_config = load_model_config(
        args.model,
        model_config_path=getattr(args, "model_config", None),
    )

    # Determine backend
    backend_str = args.backend
    if not backend_str and model_specific_config:
        backend_str = model_specific_config.backend
    if not backend_str:
        print(
            "Error: Backend must be specified via --backend or model config",
            file=sys.stderr,
        )
        sys.exit(1)

    # Merge configs: CLI args override model config, which overrides defaults
    subset = args.subset

    # Get values from model config with subset overrides
    if model_specific_config:
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
    else:
        temperature = 0.0
        max_tokens = 4096
        batch_size = 1
        tensor_parallel = 1
        api_base = None
        timeout = 120
        max_retries = 3
        device = "cuda"
        dtype = "bfloat16"
        rate_limit_rpm = None

    # CLI overrides (only if explicitly provided)
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

    # Build model config
    model_config = ModelConfig(
        model_id=args.model,
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

    # Build evaluation config
    config = EvaluationConfig(
        dataset_id=args.dataset,
        subset=subset,
        split=args.split,
        format_type=FormatType(args.format),
        model=model_config,
        batch_size=batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        model_config_path=getattr(args, "model_config", None),
    )

    # Print configuration summary
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Backend: {backend_str}")
    print(f"  Dataset: {args.dataset} ({subset}/{args.split})")
    print(f"  Format: {args.format}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Temperature: {temperature}")
    print(f"  Max Tokens: {max_tokens}")
    if model_specific_config:
        print(f"  Config File: {args.model_config or 'auto-detected'}")

    # Run pipeline
    print(f"\nEvaluating {args.model} on {args.dataset}...")
    pipeline = EvaluationPipeline(config)
    output = pipeline.run()

    # Generate reports
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

    # Print results
    print_results(output.metrics, args.format)

    # Print summary
    print(
        f"\nSamples: {output.summary['successful']}/{output.summary['total_samples']}"
    )
    print(f"Avg Latency: {output.summary['avg_latency_ms']:.2f}ms")


def cmd_compare(args: argparse.Namespace) -> None:
    """Run comparison command."""
    paths = [Path(p) for p in args.report_files]

    # Check files exist
    for path in paths:
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Create comparator
    comparator = ModelComparator.from_json_files(paths)

    # Save comparison
    comparator.save_comparison(Path(args.output))

    # Print summary
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


def cmd_get_run_command(args: argparse.Namespace) -> None:
    """Generate the uv run command for a model."""
    loader = ModelConfigLoader()
    config_path = Path("configs/models") / f"{args.config}.yaml"

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = loader.load_from_path(config_path)

    if not config.dependency_group:
        print(f"# No dependency_group defined for {args.config}")
        print(f"uv run python -m evaluation.cli evaluate \\")
    else:
        print(f"uv run --group {config.dependency_group} python -m evaluation.cli evaluate \\")

    print(f'  -m "{config.model_id}" \\')
    print(f'  --model-config "{config_path}" \\')
    print(f'  -d "<DATASET>" \\')
    print(f'  -f sentence \\')
    print(f'  --max-samples 10')


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VLM/OCR Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with model config file (recommended)
  uv run python -m evaluation.cli evaluate -m gpt-4o -d junyeong-nero/synthetic-ocr-images-korean -f sentence

  # Evaluate with explicit config path
  uv run python -m evaluation.cli evaluate -m gpt-4o --model-config configs/models/gpt-4o.yaml -d dataset -f sentence

  # Override config values via CLI
  uv run python -m evaluation.cli evaluate -m gpt-4o -d dataset -f sentence --temperature 0.5 --batch-size 4

  # Evaluate without model config (all params via CLI)
  uv run python -m evaluation.cli evaluate -m gpt-4o -b openai -d dataset -f sentence --temperature 0.0 --max-tokens 4096

  # Compare multiple model results
  uv run python -m evaluation.cli compare results/gpt4o/report.json results/qwen/report.json -o comparison

  # List available model configs
  uv run python -m evaluation.cli list-configs

  # List available backends
  uv run python -m evaluation.cli list-backends
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    eval_parser.add_argument("-m", "--model", required=True, help="Model ID")
    eval_parser.add_argument(
        "-b",
        "--backend",
        choices=[
            "openai",
            "anthropic",
            "google",
            "transformers",
        ],
        help="Inference backend (optional if model config exists)",
    )
    eval_parser.add_argument(
        "-d", "--dataset", required=True, help="HuggingFace dataset ID"
    )
    eval_parser.add_argument("--subset", default="default", help="Dataset subset")
    eval_parser.add_argument("--split", default="test", help="Dataset split")
    eval_parser.add_argument(
        "-f",
        "--format",
        default="sentence",
        choices=["sentence", "table", "document", "markdown", "kie"],
        help="Evaluation format type",
    )
    eval_parser.add_argument(
        "--model-config",
        default=None,
        help="Path to model-specific config YAML (auto-detected if not provided)",
    )
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

    # CLI overrides (optional, override model config values)
    override_group = eval_parser.add_argument_group(
        "config overrides", "Override values from model config (optional)"
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
        help="Generation temperature (overrides model config)",
    )
    override_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens (overrides model config)",
    )
    override_group.add_argument(
        "--api-base", default=None, help="Custom API base URL (overrides model config)"
    )
    override_group.add_argument(
        "--tensor-parallel",
        type=int,
        default=None,
        help="Tensor parallel size (overrides model config)",
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

    # Get run command
    run_cmd_parser = subparsers.add_parser(
        "get-run-command", help="Generate uv run command for a model"
    )
    run_cmd_parser.add_argument("config", help="Config name (without .yaml)")

    args = parser.parse_args()

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "list-backends":
        cmd_list_backends(args)
    elif args.command == "list-configs":
        cmd_list_configs(args)
    elif args.command == "get-run-command":
        cmd_get_run_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
