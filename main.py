"""Main entry point for generation and evaluation pipelines."""

import sys
import argparse
import os
import re
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

sys.path.insert(0, "src")
from env_utils import set_global_seed

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
        subset_scores = []
        subset_weights = []
        normalized_subsets = []
        for subset_entry in entry.get("subsets", []):
            metric_key = subset_entry.get("metric_key")
            metric_value = subset_entry.get("metric_value")
            normalized = _normalize_metric(metric_key, metric_value)
            normalized_subsets.append(
                {
                    **subset_entry,
                    "normalized_value": normalized,
                }
            )
            if isinstance(normalized, (int, float)):
                subset_scores.append(normalized)
                weight = subset_entry.get("total_samples")
                subset_weights.append(weight if isinstance(weight, (int, float)) else 1)

        if subset_scores:
            weight_total = sum(subset_weights)
            normalized_average = (
                sum(score * weight for score, weight in zip(subset_scores, subset_weights))
                / weight_total
                if weight_total
                else None
            )
        else:
            normalized_average = None

        leaderboard_entries.append(
            {
                "timestamp": entry.get("timestamp"),
                "protocol_version": entry.get("protocol_version"),
                "model_id": entry.get("model_id"),
                "backend": entry.get("backend"),
                "dataset": entry.get("dataset"),
                "split": entry.get("split"),
                "average_score": entry.get("average_score"),
                "normalized_average_score": normalized_average,
                "subsets": normalized_subsets,
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

    lines = ["# OCR Benchmark Leaderboard", "", "| Rank | Model | Backend | Dataset | Split | Normalized | Raw |", "|---:|---|---|---|---|---:|---:|"]
    for idx, entry in enumerate(leaderboard_entries, start=1):
        normalized_score = entry.get("normalized_average_score")
        raw_score = entry.get("average_score")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                idx,
                entry.get("model_id") or "-",
                entry.get("backend") or "-",
                entry.get("dataset") or "-",
                entry.get("split") or "-",
                f"{normalized_score:.4f}" if isinstance(normalized_score, (int, float)) else "-",
                f"{raw_score:.4f}" if isinstance(raw_score, (int, float)) else "-",
            )
        )

    leaderboard_md = output_dir / "leaderboard.md"
    with open(leaderboard_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
        "font_path": args.font_path,
        "output_dir": args.output_dir,
        "lang": args.lang,
        "corpus_size": args.corpus_size,
        "size": args.size,
        "typo_ratio": args.typo_ratio,
        "similarity_threshold": args.similarity_threshold,
        "similarity_top_k": args.similarity_top_k,
        "format": args.format,
        "template": args.template,
        "table_size": args.table_size,
        "mixed": args.mixed,
        "seed": args.seed,
    }
    pipeline(**pipeline_args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation command."""
    from evaluation.config import FormatType, InferenceBackend, ModelConfig, EvaluationConfig
    from evaluation.pipeline import EvaluationPipeline
    from evaluation.report import ReportGenerator

    model_specific_config = load_model_config(args.model_config)

    set_global_seed(args.seed)

    representative_metrics = {
        FormatType.SENTENCE.value: "avg_cer",
        FormatType.TABLE.value: "avg_teds",
        FormatType.DOCUMENT.value: "avg_overall_f1",
        FormatType.MARKDOWN.value: "avg_cer",
        FormatType.KIE.value: "avg_entity_f1",
    }

    backend_str = args.backend
    if not backend_str:
        backend_str = model_specific_config.backend

    def run_for_subset(subset: str, output_dir: Path) -> dict:
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
            output_dir=str(output_dir),
            seed=args.seed,
            batch_api=args.batch_api,
            batch_poll_seconds=args.batch_poll_seconds,
            batch_timeout_seconds=args.batch_timeout_seconds,
            batch_completion_window=args.batch_completion_window,
            model_config_path=args.model_config,
        )

        print("\nConfiguration:")
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
        output_path = output_dir

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

        print_results(output.metrics, format_type.value)

        print(
            f"\nSamples: {output.summary['successful']}/{output.summary['total_samples']}"
        )
        print(f"Avg Latency: {output.summary['avg_latency_ms']:.2f}ms")

        metric_key = representative_metrics.get(format_type.value)
        metric_value = output.metrics.get(metric_key) if metric_key else None
        if metric_key and metric_value is None:
            print(f"Warning: Missing representative metric '{metric_key}' for {subset}")

        return {
            "subset": subset,
            "format": format_type.value,
            "metric_key": metric_key,
            "metric_value": metric_value,
            "model_id": model_config.model_id,
            "backend": backend_str,
            "prompt_source": output.config.get("prompt_source"),
            "total_samples": output.summary.get("total_samples"),
        }

    def subset_output_dir(base_dir: Path, subset: str, use_subdir: bool) -> Path:
        if not use_subdir:
            return base_dir
        safe_subset = re.sub(r"[^A-Za-z0-9_-]", "_", subset)
        subset_hash = hashlib.sha256(subset.encode("utf-8")).hexdigest()[:8]
        return base_dir / f"{safe_subset}-{subset_hash}"

    default_subsets = [
        FormatType.SENTENCE.value,
        FormatType.TABLE.value,
        FormatType.DOCUMENT.value,
        FormatType.MARKDOWN.value,
        FormatType.KIE.value,
    ]

    if args.subset is None:
        subsets = default_subsets
        print("No --subset specified. Running for all default subsets.")
    else:
        subsets = [value.strip() for value in args.subset.split(",") if value.strip()]
        if not subsets:
            print("Error: --subset cannot be empty.", file=sys.stderr)
            sys.exit(1)

    summary_entries = []
    summary_output_dir = Path(args.output_dir)

    if len(subsets) == 1:
        subset = subsets[0]
        has_invalid_chars = bool(re.search(r"[^A-Za-z0-9_-]", subset))
        summary_entries.append(
            run_for_subset(
                subset,
                subset_output_dir(Path(args.output_dir), subset, has_invalid_chars),
            )
        )
    else:
        print("\n" + "=" * 60)
        print("Running subsets:", ", ".join(subsets))
        print("=" * 60)
        for subset in subsets:
            print("\n" + "=" * 60)
            print(f"Running subset: {subset}")
            print("=" * 60)
            summary_entries.append(
                run_for_subset(
                    subset, subset_output_dir(Path(args.output_dir), subset, True)
                )
            )

    metric_values = [
        entry["metric_value"]
        for entry in summary_entries
        if isinstance(entry["metric_value"], (int, float))
    ]
    average_score = sum(metric_values) / len(metric_values) if metric_values else None

    summary_entry = {
        "timestamp": _iso_timestamp(),
        "protocol_version": PROTOCOL_VERSION,
        "model_id": summary_entries[0]["model_id"] if summary_entries else None,
        "backend": summary_entries[0]["backend"] if summary_entries else None,
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "subsets": summary_entries,
        "average_score": average_score,
    }

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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
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
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="SSIM threshold for storing similar characters",
    )
    gen_parser.add_argument(
        "--similarity-top-k",
        type=int,
        default=8,
        help="Max similar characters to store per character",
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
        help="Table size range as 'min-max' applied to rows and cols",
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
    eval_parser.add_argument(
        "-s",
        "--subset",
        default=None,
        help=(
            "Dataset subset(s) and format type. Use comma-separated values to run multiple. "
            "If omitted, runs all default subsets"
        ),
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
