from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from src.cli.evaluation_outputs import persist_evaluation_outputs
from src.evaluation.config import EvaluationConfig, EvaluationMode, InferenceBackend, ModelConfig
from src.evaluation.leaderboard import update_leaderboards
from src.evaluation.model_config import ModelConfigLoader, ModelSpecificConfig


DEFAULT_DATASET_PREFIX = "junyeong-nero/synthetic-ocr-images"
DEFAULT_LANGUAGE = "ko"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 200
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"


def get_api_key(backend: str) -> Optional[str]:
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "upstage": "UPSTAGE_API_KEY",
    }
    env_var = key_map.get(backend)
    return None if env_var is None else __import__("os").environ.get(env_var)


def _resolve_execution_mode(args: argparse.Namespace) -> EvaluationMode:
    if args.inference_only:
        return EvaluationMode.INFERENCE_ONLY
    if args.evaluate_only:
        return EvaluationMode.EVALUATE_ONLY
    return EvaluationMode.ALL


def _resolve_evaluation_runtime(
    args: argparse.Namespace,
    model_specific_config: ModelSpecificConfig,
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
    model_config_path: Path,
    model_specific_config: ModelSpecificConfig,
) -> tuple[EvaluationConfig, ModelConfig, dict[str, Any], EvaluationMode]:
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
        model_config_path=str(model_config_path),
    )
    return config, model_config, runtime, execution_mode


def _save_evaluation_reports(output: Any, output_path: Path, report_format: str) -> None:
    from src.evaluation.report import ReportGenerator

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


def print_results(metrics: dict[str, Any], format_name: str) -> None:
    print("\n" + "=" * 60)
    print(f" {format_name.upper()} EVALUATION RESULTS ")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


def _resolve_config_path(args: argparse.Namespace) -> Path:
    loader = ModelConfigLoader()
    if args.model_config:
        return Path(args.model_config)

    model_ref = args.model_id or args.model_ref
    if not model_ref:
        raise ValueError("One of --model-config, --model-id, or positional model_ref is required")

    resolved = loader.resolve_config_path(model_ref)
    if resolved is None:
        raise FileNotFoundError(f"Config not found for model reference: {model_ref}")
    return resolved


def _load_model_config(config_path: Path) -> ModelSpecificConfig:
    loader = ModelConfigLoader()
    return loader.load_from_path(config_path)


def _default_wrapper_output_dir(model_specific_config: ModelSpecificConfig, language: str) -> Path:
    model_dir_name = model_specific_config.get_model_id().split("/")[-1]
    return PROJECT_ROOT / "evaluation_result" / model_dir_name / language


def _append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def add_evaluate_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_dir_default: Optional[str],
) -> argparse.ArgumentParser:
    parser.add_argument(
        "model_ref",
        nargs="?",
        help="Config name or model id reference when not using --model-config",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to model-specific config YAML",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        default=None,
        help="Config name or model id reference",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=False,
        default=None,
        help="Hugging Face dataset ID or local path",
    )
    parser.add_argument(
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
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split")
    parser.add_argument("--language", "-l", default=DEFAULT_LANGUAGE, help="Dataset language code")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible evaluation")
    parser.add_argument("--batch-api", action="store_true", default=False, help="Use OpenAI Batch API for evaluation")
    parser.add_argument("--batch-poll-seconds", type=int, default=60, help="Polling interval for batch status")
    parser.add_argument("--batch-timeout-seconds", type=int, default=86400, help="Max wait time for batch completion")
    parser.add_argument("--batch-completion-window", default="24h", help="Batch completion window")
    parser.add_argument("--output-dir", default=output_dir_default, help="Output directory")
    parser.add_argument(
        "--report-format",
        default="all",
        choices=["json", "markdown", "html", "all"],
        help="Report output format",
    )
    mode_group = parser.add_mutually_exclusive_group()
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
    override_group = parser.add_argument_group("config overrides", "Override values from model config")
    override_group.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides model config)")
    override_group.add_argument("--temperature", type=float, default=None, help="Generation temperature")
    override_group.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    override_group.add_argument("--api-base", default=None, help="Custom API base URL")
    override_group.add_argument("--tensor-parallel", type=int, default=None, help="Tensor parallel size")
    return parser


def add_evaluate_all_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-d", "--dataset", default=None, help="Dataset override for all configs")
    parser.add_argument("-l", "--language", default=DEFAULT_LANGUAGE, help="Dataset language code")
    parser.add_argument("--max-samples", "-n", type=int, default=DEFAULT_MAX_SAMPLES, help="Max samples per config")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split")
    return parser


def add_refresh_leaderboard_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="evaluation_result",
        help="Evaluation result directory containing report.json files",
    )
    return parser


def run_with_args(args: argparse.Namespace) -> None:
    from src.evaluation.pipeline import EvaluationPipeline

    config_path = _resolve_config_path(args)
    model_specific_config = _load_model_config(config_path)
    dataset = args.dataset
    if not dataset:
        raise ValueError("--dataset is required for evaluate")

    representative_metrics = {"markdown": "avg_markdown_overall_score"}
    if args.output_dir is None:
        args.output_dir = "./evaluation_result"

    config, model_config, runtime, execution_mode = _build_evaluation_config(
        args,
        config_path,
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
    print(f"  Config File: {config_path}")
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


def run_wrapper_with_args(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args)
    model_specific_config = _load_model_config(config_path)
    dependency_group = model_specific_config.dependency_group

    dataset = args.dataset or f"{DEFAULT_DATASET_PREFIX}-{args.language}"
    output_dir = Path(args.output_dir) if args.output_dir else _default_wrapper_output_dir(model_specific_config, args.language)

    command = ["uv", "run", "--group", "evaluate"]
    if dependency_group:
        command.extend(["--group", dependency_group])
    command.extend([
        "python",
        str(MAIN_SCRIPT),
        "evaluate",
        "--model-config",
        str(config_path),
        "--dataset",
        dataset,
        "--language",
        args.language,
        "--split",
        args.split,
        "--output-dir",
        str(output_dir),
        "--report-format",
        args.report_format,
    ])

    _append_flag(command, "--backend", args.backend)
    _append_flag(command, "--max-samples", args.max_samples)
    _append_flag(command, "--seed", args.seed)
    _append_flag(command, "--batch-poll-seconds", args.batch_poll_seconds)
    _append_flag(command, "--batch-timeout-seconds", args.batch_timeout_seconds)
    _append_flag(command, "--batch-completion-window", args.batch_completion_window)
    _append_flag(command, "--batch-size", args.batch_size)
    _append_flag(command, "--temperature", args.temperature)
    _append_flag(command, "--max-tokens", args.max_tokens)
    _append_flag(command, "--api-base", args.api_base)
    _append_flag(command, "--tensor-parallel", args.tensor_parallel)
    if args.batch_api:
        command.append("--batch-api")
    if args.inference_only:
        command.append("--inference-only")
    if args.evaluate_only:
        command.append("--evaluate-only")

    print(f"Config: {config_path.stem}")
    print(f"Model: {model_specific_config.model_id}")
    print(f"Dependency Group: {dependency_group or 'none'}")
    print("")
    print("Running evaluation")
    print(f"Output: {output_dir}")
    print(f"Language: {args.language}")

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def run_all_with_args(args: argparse.Namespace) -> int:
    loader = ModelConfigLoader()
    config_paths = loader.list_public_config_paths()
    dataset = args.dataset or f"{DEFAULT_DATASET_PREFIX}-{args.language}"

    summary_dir = PROJECT_ROOT / "evaluation_result" / "_runs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = summary_dir / f"run-all-{timestamp}.log"

    passed = 0
    failed = 0
    header_lines = [
        "==========================================",
        "Running evaluations for all model configs",
        "==========================================",
        f"Dataset: {dataset}",
        f"Language: {args.language}",
        f"Max Samples: {args.max_samples}",
        f"Split: {args.split}",
        f"Log: {summary_path}",
        "",
    ]
    summary_path.write_text("\n".join(header_lines), encoding="utf-8")
    print("\n".join(header_lines))

    for config_path in config_paths:
        config_name = config_path.stem
        section_header = "\n".join([
            "------------------------------------------",
            f"Config: {config_name}",
            "------------------------------------------",
        ])
        print(section_header)
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(section_header + "\n")

        command = [
            "uv",
            "run",
            "--group",
            "evaluate",
            "python",
            str(MAIN_SCRIPT),
            "evaluate-run",
            config_name,
            "--dataset",
            dataset,
            "--language",
            args.language,
            "--max-samples",
            str(args.max_samples),
            "--split",
            args.split,
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        combined_output = (completed.stdout or "") + (completed.stderr or "")
        print(combined_output, end="")
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(combined_output)

        if completed.returncode == 0:
            passed += 1
            result_line = "Result: PASSED"
        else:
            failed += 1
            result_line = "Result: FAILED"
        print(result_line)
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(result_line + "\n\n")

    footer = "\n".join([
        "==========================================",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "==========================================",
    ])
    print(footer)
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(footer + "\n")

    return 1 if failed > 0 else 0


def refresh_leaderboard_with_args(args: argparse.Namespace) -> None:
    update_leaderboards(Path(args.base_dir))
