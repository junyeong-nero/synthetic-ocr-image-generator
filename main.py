"""Main entry point for generation and evaluation pipelines."""

import sys
import argparse
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

sys.path.insert(0, "src")
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
    }
    env_var = key_map.get(backend)
    return os.environ.get(env_var) if env_var else None


PROTOCOL_VERSION = "1.0"


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_metric(metric_key: Optional[str], metric_value: Optional[float]) -> Optional[float]:
    if metric_key is None or metric_value is None:
        return None
    lower_better = {"avg_cer", "avg_wer"}
    normalized = 1.0 - metric_value if metric_key in lower_better else metric_value
    return max(0.0, min(1.0, normalized))


def _safe_numeric(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _write_protocol_snapshot(
    output_dir: Path,
    output,
    report_format: str,
) -> Path:
    snapshot = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": _iso_timestamp(),
        "command": "evaluate",
        "report_format": report_format,
        "config": output.config,
        "summary": output.summary,
        "prompt": output.config.get("prompt"),
        "prompt_source": output.config.get("prompt_source"),
        "system_prompt": output.config.get("system_prompt"),
    }
    path = output_dir / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return path


def _write_leaderboard(output_dir: Path, summary_entries: list[dict]) -> None:
    leaderboard_entries = []
    for entry in summary_entries:
        metric_key = entry.get("metric_key")
        metric_value = entry.get("metric_value")
        normalized_average = _normalize_metric(metric_key, metric_value)

        entry_metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
        leaderboard_entries.append(
            {
                "timestamp": entry.get("timestamp"),
                "protocol_version": entry.get("protocol_version"),
                "model_id": entry.get("model_id"),
                "backend": entry.get("backend"),
                "dataset": entry.get("dataset"),
                "split": entry.get("split"),
                "format": entry.get("format"),
                "average_score": entry.get("average_score", metric_value),
                "normalized_average_score": normalized_average,
                "average_empty_rate": entry.get("average_empty_rate", entry.get("empty_rate")),
                "average_parse_fail_rate": entry.get(
                    "average_parse_fail_rate", entry.get("parse_fail_rate")
                ),
                "markdown_text_score": _safe_numeric(entry_metrics.get("avg_markdown_text_score")),
                "markdown_table_teds": _safe_numeric(entry_metrics.get("avg_markdown_table_teds")),
                "markdown_formula_score": _safe_numeric(entry_metrics.get("avg_markdown_formula_score")),
                "markdown_order_score": _safe_numeric(entry_metrics.get("avg_markdown_order_score")),
            }
        )

    leaderboard_entries.sort(
        key=lambda item: (
            item.get("normalized_average_score") is not None,
            item.get("normalized_average_score") or 0,
        ),
        reverse=True,
    )

    leaderboard_path = output_dir / "leaderboard.json"
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard_entries, f, ensure_ascii=False, indent=2)

    lines = [
        "# OCR Benchmark Leaderboard",
        "",
        "| Rank | Model | Backend | Dataset | Split | Format | Normalized | Raw | Text | Table | Formula | Order | Empty Rate | Parse Fail Rate |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, entry in enumerate(leaderboard_entries, start=1):
        normalized_score = entry.get("normalized_average_score")
        raw_score = entry.get("average_score")
        empty_rate = entry.get("average_empty_rate")
        parse_fail_rate = entry.get("average_parse_fail_rate")
        text_score = entry.get("markdown_text_score")
        table_score = entry.get("markdown_table_teds")
        formula_score = entry.get("markdown_formula_score")
        order_score = entry.get("markdown_order_score")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                idx,
                entry.get("model_id") or "-",
                entry.get("backend") or "-",
                entry.get("dataset") or "-",
                entry.get("split") or "-",
                entry.get("format") or "-",
                f"{normalized_score:.4f}" if isinstance(normalized_score, (int, float)) else "-",
                f"{raw_score:.4f}" if isinstance(raw_score, (int, float)) else "-",
                f"{text_score:.4f}" if isinstance(text_score, (int, float)) else "-",
                f"{table_score:.4f}" if isinstance(table_score, (int, float)) else "-",
                f"{formula_score:.4f}" if isinstance(formula_score, (int, float)) else "-",
                f"{order_score:.4f}" if isinstance(order_score, (int, float)) else "-",
                f"{empty_rate:.4f}" if isinstance(empty_rate, (int, float)) else "-",
                f"{parse_fail_rate:.4f}" if isinstance(parse_fail_rate, (int, float)) else "-",
            )
        )

    leaderboard_md = output_dir / "leaderboard.md"
    with open(leaderboard_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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


def cmd_generate(args: argparse.Namespace) -> None:
    """Run generation command."""
    from pipeline import pipeline

    set_global_seed(args.seed)
    pipeline_args = {
        "repo_id": args.repo_id,
        "output_dir": args.output_dir,
        "lang": args.lang,
        "size": args.size,
        "template": args.template,
        "markdown_renderer": args.markdown_renderer,
        "similar_char_ratio": args.similar_char_ratio,
        "similarity_db_path": args.similarity_db_path,
        "add_noise": args.add_noise,
        "add_blur": args.add_blur,
        "mixed": args.mixed,
        "train_ratio": args.train_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    pipeline(**pipeline_args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation command."""
    from evaluation.config import (
        InferenceBackend,
        ModelConfig,
        EvaluationConfig,
        EvaluationMode,
    )
    from evaluation.pipeline import EvaluationPipeline
    from evaluation.report import ReportGenerator

    model_specific_config = load_model_config(args.model_config)

    set_global_seed(args.seed)

    representative_metrics = {
        "markdown": "avg_markdown_overall_score",
    }

    backend_str = args.backend
    if not backend_str:
        backend_str = model_specific_config.backend

    temperature = model_specific_config.get_temperature()
    max_tokens = model_specific_config.get_max_tokens()
    batch_size = model_specific_config.get_batch_size()
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
        model_id=model_specific_config.get_model_id(),
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

    execution_mode = EvaluationMode.ALL
    if args.inference_only:
        execution_mode = EvaluationMode.INFERENCE_ONLY
    elif args.evaluate_only:
        execution_mode = EvaluationMode.EVALUATE_ONLY

    config = EvaluationConfig(
        dataset_id=args.dataset,
        split=args.split,
        model=model_config,
        batch_size=batch_size,
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

    print("\nConfiguration:")
    print(f"  Model: {model_config.model_id}")
    print(f"  Backend: {backend_str}")
    print(f"  Dataset: {args.dataset} ({args.split})")
    print("  Format: auto-detect")
    print(f"  Batch Size: {batch_size}")
    print(f"  Temperature: {temperature}")
    print(f"  Max Tokens: {max_tokens}")
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
    generator = ReportGenerator(output)
    if args.report_format == "all":
        paths = generator.save_all(output_path)
        print("\nReports saved:")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")
    else:
        method = getattr(generator, f"to_{args.report_format}")
        path = method(output_path / f"report.{args.report_format}")
        print(f"\nReport saved: {path}")

    protocol_path = _write_protocol_snapshot(output_path, output, args.report_format)
    print(f"Protocol snapshot saved: {protocol_path}")

    resolved_format = str(output.config.get("format") or "markdown")
    print_results(output.metrics, resolved_format)
    print(f"\nSamples: {output.summary['successful']}/{output.summary['total_samples']}")
    print(f"Avg Latency: {output.summary['avg_latency_ms']:.2f}ms")

    metric_key = representative_metrics.get(resolved_format)
    metric_value = output.metrics.get(metric_key) if metric_key else None
    if metric_key and metric_value is None:
        print(f"Warning: Missing representative metric '{metric_key}' for {resolved_format}")

    summary_entry = {
        "timestamp": _iso_timestamp(),
        "protocol_version": PROTOCOL_VERSION,
        "model_id": model_config.model_id,
        "backend": backend_str,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "format": resolved_format,
        "metric_key": metric_key,
        "metric_value": metric_value,
        "average_score": metric_value,
        "empty_rate": output.summary.get("empty_rate"),
        "parse_fail_rate": output.summary.get("parse_fail_rate"),
        "total_samples": output.summary.get("total_samples"),
        "prompt_source": output.config.get("prompt_source"),
        "metrics": {
            key: float(value)
            for key, value in output.metrics.items()
            if isinstance(value, (int, float))
        },
    }

    summary_output_dir = Path(args.output_dir)

    summary_path = summary_output_dir / "model_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing = loaded
            elif isinstance(loaded, dict):
                existing = [loaded]
        except (json.JSONDecodeError, OSError) as exc:
            corrupt_path = summary_path.with_suffix(".corrupt.json")
            try:
                if summary_path.exists():
                    summary_path.replace(corrupt_path)
            finally:
                raise RuntimeError(
                    f"Corrupt summary file moved to {corrupt_path}"
                ) from exc

    existing.append(summary_entry)

    tmp_path = summary_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, summary_path)
    print(f"\nModel summary saved: {summary_path}")

    _write_leaderboard(summary_output_dir, existing)


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

    # Generate command (legacy main arguments)
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
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
        "--template",
        type=str,
        default=None,
        help="Template for generation",
    )
    gen_parser.add_argument(
        "--markdown-renderer",
        type=str,
        default="pil",
        choices=["pil", "html2image"],
        help="Markdown rendering pipeline (pil or markdown->html->image via html2image)",
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
        "--add-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable noise effect (default: generator setting)",
    )
    gen_parser.add_argument(
        "--add-blur",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable blur effect (default: generator setting)",
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
            "transformers",
            "paddleocr",
        ],
        help="Inference backend (optional if model config exists)",
    )
    eval_parser.add_argument("--split", default="train", help="Dataset split")
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
        "--output-dir", default="./evaluation_results", help="Output directory"
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
