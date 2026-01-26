"""Command-line interface for evaluation pipeline."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from evaluation.config import EvaluationConfig, FormatType, InferenceBackend, ModelConfig
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


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation command."""
    # Build model config
    model_config = ModelConfig(
        model_id=args.model,
        backend=InferenceBackend(args.backend),
        api_key=get_api_key(args.backend),
        api_base=args.api_base,
        tensor_parallel_size=args.tensor_parallel,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Build evaluation config
    config = EvaluationConfig(
        dataset_id=args.dataset,
        subset=args.subset,
        split=args.split,
        format_type=FormatType(args.format),
        model=model_config,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )

    # Run pipeline
    print(f"Evaluating {args.model} on {args.dataset}...")
    pipeline = EvaluationPipeline(config)
    output = pipeline.run()

    # Generate reports
    generator = ReportGenerator(output)
    output_path = Path(args.output_dir)

    if args.report_format == "all":
        paths = generator.save_all(output_path)
        print(f"\nReports saved:")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")
    else:
        method = getattr(generator, f"to_{args.report_format}")
        path = method(output_path / f"report.{args.report_format}")
        print(f"\nReport saved: {path}")

    # Print results
    print_results(output.metrics, args.format)

    # Print summary
    print(f"\nSamples: {output.summary['successful']}/{output.summary['total_samples']}")
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


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VLM/OCR Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with OpenAI GPT-4o
  evaluate evaluate -m gpt-4o -b openai -d nero-nlp/synthetic-ocr-korean

  # Evaluate with vLLM
  evaluate evaluate -m Qwen/Qwen2-VL-7B -b vllm -d nero-nlp/synthetic-ocr-korean

  # Evaluate with Ollama
  evaluate evaluate -m llava -b ollama -d nero-nlp/synthetic-ocr-korean

  # Compare multiple models
  evaluate compare results/gpt4o/report.json results/qwen/report.json -o comparison

  # List available backends
  evaluate list-backends
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    eval_parser.add_argument(
        "-m", "--model", required=True, help="Model ID"
    )
    eval_parser.add_argument(
        "-b", "--backend", required=True,
        choices=["openai", "anthropic", "google", "transformers", "vllm", "sglang", "ollama"],
        help="Inference backend",
    )
    eval_parser.add_argument(
        "-d", "--dataset", required=True, help="HuggingFace dataset ID"
    )
    eval_parser.add_argument(
        "--subset", default="default", help="Dataset subset"
    )
    eval_parser.add_argument(
        "--split", default="test", help="Dataset split"
    )
    eval_parser.add_argument(
        "-f", "--format", default="sentence",
        choices=["sentence", "table", "document", "markdown", "kie"],
        help="Evaluation format type",
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size"
    )
    eval_parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to evaluate"
    )
    eval_parser.add_argument(
        "--output-dir", default="./evaluation_results", help="Output directory"
    )
    eval_parser.add_argument(
        "--api-base", default=None, help="Custom API base URL"
    )
    eval_parser.add_argument(
        "--tensor-parallel", type=int, default=1, help="Tensor parallel size (vLLM/SGLang)"
    )
    eval_parser.add_argument(
        "--temperature", type=float, default=0.0, help="Generation temperature"
    )
    eval_parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max output tokens"
    )
    eval_parser.add_argument(
        "--report-format", default="all",
        choices=["json", "markdown", "html", "all"],
        help="Report output format",
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
    list_parser = subparsers.add_parser("list-backends", help="List available backends")

    args = parser.parse_args()

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "list-backends":
        cmd_list_backends(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
