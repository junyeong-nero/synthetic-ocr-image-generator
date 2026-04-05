"""Main evaluation pipeline orchestrator."""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from src.evaluation.checkpoint import build_checkpoint_context, resolve_checkpoint_path
from src.evaluation.config import DEFAULT_PROMPT, EvaluationConfig, EvaluationMode, ModelConfig
from src.evaluation.model_config import ModelConfigLoader, ModelSpecificConfig
from src.evaluation.runner import EvaluationRunner
from src.evaluation.types import EvaluationOutput, InferenceResult, RunnerState
from src.evaluation.strategies import MarkdownEvaluator
from src.models.registry import create_model
from src.env_utils import get_environment_metadata


class EvaluationPipeline:
    """
    Main evaluation pipeline orchestrator.

    Handles dataset loading, model inference, and metric computation.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.runner: Optional[EvaluationRunner] = None
        self.prompt: Optional[str] = None
        self.system_prompt: Optional[str] = None
        self.prompt_source: Optional[str] = None
        self.metric_views: Dict[str, Dict[str, float]] = {}
        self.dataset_fingerprint: Optional[str] = None
        self.dataset_info: Optional[Any] = None
        self.model_specific_config = self._load_model_specific_config()

        if self.config.execution_mode != EvaluationMode.EVALUATE_ONLY:
            self._ensure_runner()

    def _ensure_runner(self) -> EvaluationRunner:
        if self.runner is None:
            self.model = create_model(self.config.model)
            self.runner = EvaluationRunner(self.config, self.model)
        return self.runner

    def _load_checkpoint_results(self) -> List[InferenceResult]:
        output_dir = Path(self.config.output_dir)
        checkpoint_to_load = resolve_checkpoint_path(output_dir)
        if checkpoint_to_load is None:
            raise FileNotFoundError(
                f"Checkpoint file not found in {output_dir}. "
                "Run with --inference-only first or use full evaluation mode."
            )

        try:
            state = RunnerState.load(checkpoint_to_load)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_to_load}: {exc}") from exc

        expected_context = build_checkpoint_context(self.config)
        if state.context and state.context != expected_context:
            raise RuntimeError(
                "Checkpoint context mismatch. "
                f"Expected {expected_context}, got {state.context}."
            )

        return sorted(
            [InferenceResult(**row) for row in state.results],
            key=lambda result: result.index,
        )

    def _load_model_specific_config(self) -> Optional[ModelSpecificConfig]:
        loader = ModelConfigLoader()
        if self.config.model_config_path:
            return loader.load_from_path(Path(self.config.model_config_path))
        return loader.load(self.config.model.model_id)

    def _load_dataset(self) -> Dataset:
        """Load dataset from HuggingFace."""
        dataset = load_dataset(
            self.config.dataset_id,
            split=self.config.split,
        )

        self.dataset_fingerprint = getattr(dataset, "_fingerprint", None)
        self.dataset_info = getattr(dataset, "info", None)

        # Apply max_samples limit
        if self.config.max_samples:
            dataset = dataset.select(
                range(min(self.config.max_samples, len(dataset)))
            )

        return dataset

    def _resolve_prompt(self) -> tuple[str, Optional[str], str]:
        """Resolve prompt and system prompt with source labeling."""
        base_prompt: Optional[str] = None
        base_system_prompt: Optional[str] = None
        source: Optional[str] = None

        if self.config.prompt:
            base_prompt = self.config.prompt
            base_system_prompt = self.config.system_prompt
            source = "cli"

        if base_prompt is None and self.model_specific_config:
            prompt_config = self.model_specific_config.get_prompt()
            base_prompt = prompt_config.prompt
            base_system_prompt = prompt_config.system_prompt
            source = "model_config"

        if base_prompt is None:
            base_prompt = DEFAULT_PROMPT
            base_system_prompt = None
            source = "default"

        if base_prompt is None:
            raise RuntimeError("Prompt resolution failed")

        return base_prompt, base_system_prompt, source or "default"

    def _extract_ground_truths(self, dataset: Dataset) -> List[Any]:
        evaluator = MarkdownEvaluator()
        return evaluator.extract_ground_truths(dataset, self.config.target_column)

    def _compute_metrics(
        self, results: List[InferenceResult]
    ) -> Dict[str, float]:
        # Filter valid results
        valid_results = [
            r
            for r in results
            if r.error is None
            and r.prediction is not None
            and str(r.prediction).strip() != ""
        ]
        if not valid_results:
            self.metric_views = {"normalized": {}}
            return {}

        predictions = [str(r.prediction) for r in valid_results]
        ground_truths = [r.ground_truth for r in valid_results]

        evaluator = MarkdownEvaluator()
        metric_views = evaluator.compute_metric_views(predictions, ground_truths)
        self.metric_views = metric_views
        return metric_views.get("normalized", {})

    def _compute_quality_metrics(self, results: List[InferenceResult]) -> Dict[str, float]:
        total = len(results)
        if total == 0:
            return {
                "empty_count": 0.0,
                "empty_rate": 0.0,
                "parse_fail_count": 0.0,
                "parse_fail_rate": 0.0,
            }

        empty_count = float(
            sum(1 for r in results if str(r.prediction).strip() == "")
        )

        parse_fail_count = 0.0

        return {
            "empty_count": empty_count,
            "empty_rate": empty_count / float(total),
            "parse_fail_count": parse_fail_count,
            "parse_fail_rate": parse_fail_count / float(total),
        }

    def _prepare_run_inputs(self) -> tuple[List[Any], List[Any], List[str]]:
        print(f"Loading dataset: {self.config.dataset_id}")
        dataset = self._load_dataset()
        print(f"Loaded {len(dataset)} samples")
        print("Format: markdown")

        images = dataset[self.config.image_column]
        ground_truths = self._extract_ground_truths(dataset)
        prompt, system_prompt, prompt_source = self._resolve_prompt()
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.prompt_source = prompt_source
        prompts = [prompt] * len(images)
        return images, ground_truths, prompts

    @staticmethod
    def _build_summary(results: List[InferenceResult]) -> Dict[str, Any]:
        total = len(results)
        if total == 0:
            return {
                "total_samples": 0,
                "successful": 0,
                "failed": 0,
                "avg_latency_ms": 0.0,
            }

        successful = 0
        failed = 0
        total_latency = 0.0
        for result in results:
            total_latency += float(result.latency_ms)
            if result.error is None:
                successful += 1
            else:
                failed += 1

        return {
            "total_samples": total,
            "successful": successful,
            "failed": failed,
            "avg_latency_ms": total_latency / float(total),
        }

    def _build_output(self, results: List[InferenceResult]) -> EvaluationOutput:
        metrics = self._compute_metrics(results)
        quality_metrics = self._compute_quality_metrics(results)
        merged_metrics = {**metrics, **quality_metrics}
        summary = self._build_summary(results)

        summary.update(
            {
                "model_id": self.config.model.model_id,
                "backend": self.config.model.backend.value,
                "empty_count": int(quality_metrics["empty_count"]),
                "empty_rate": quality_metrics["empty_rate"],
                "parse_fail_count": int(quality_metrics["parse_fail_count"]),
                "parse_fail_rate": quality_metrics["parse_fail_rate"],
            }
        )

        return EvaluationOutput(
            config=self._config_to_dict(),
            metrics=merged_metrics,
            metric_views=self.metric_views,
            per_sample_results=[r.to_dict() for r in results],
            summary=summary,
        )


    async def run_async(self) -> EvaluationOutput:
        """Run the evaluation pipeline asynchronously."""
        images, ground_truths, prompts = self._prepare_run_inputs()
        runner = self._ensure_runner()
        results = await runner.run_async(images, ground_truths, prompts)
        return self._build_output(results)

    def run(self) -> EvaluationOutput:
        """Run the evaluation pipeline synchronously."""
        images, ground_truths, prompts = self._prepare_run_inputs()
        runner = self._ensure_runner()
        results = runner.run(images, ground_truths, prompts)
        return self._build_output(results)

    def run_inference_only(self) -> List[InferenceResult]:
        images, ground_truths, prompts = self._prepare_run_inputs()
        runner = self._ensure_runner()
        return runner.run(images, ground_truths, prompts)

    def run_evaluate_only(self) -> EvaluationOutput:
        results = self._load_checkpoint_results()
        return self._build_output(results)

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        dataset_info = None
        if self.dataset_info is not None:
            dataset_info = {
                "builder_name": getattr(self.dataset_info, "builder_name", None),
                "version": str(getattr(self.dataset_info, "version", ""))
                if getattr(self.dataset_info, "version", None)
                else None,
                "features": list(getattr(self.dataset_info, "features", {}).keys()),
            }
        return {
            "dataset_id": self.config.dataset_id,
            "split": self.config.split,
            "language": self.config.language,
            "format": "markdown",
            "batch_size": self.config.batch_size,
            "max_samples": self.config.max_samples,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "prompt_source": self.prompt_source,
            "seed": self.config.seed,
            "batch_api": self.config.batch_api,
            "batch_poll_seconds": self.config.batch_poll_seconds,
            "batch_timeout_seconds": self.config.batch_timeout_seconds,
            "batch_completion_window": self.config.batch_completion_window,
            "execution_mode": self.config.execution_mode.value,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_info": dataset_info,
            "environment": get_environment_metadata(),
            "model_config_path": self.config.model_config_path,
            "model": {
                "model_id": self.config.model.model_id,
                "backend": self.config.model.backend.value,
                "temperature": self.config.model.temperature,
                "max_tokens": self.config.model.max_tokens,
            },
        }


def evaluate_pipeline(
    dataset_id: str,
    model_id: str,
    backend: str,
    split: str = "test",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    output_dir: str = "./evaluation_result",
    **kwargs,
) -> EvaluationOutput:
    """
    Convenience function to run evaluation pipeline.

    Args:
        dataset_id: HuggingFace dataset ID.
        model_id: Model identifier.
        backend: Inference backend name.
        split: Dataset split.
        batch_size: Batch size.
        max_samples: Maximum samples to evaluate.
        output_dir: Output directory.
        **kwargs: Additional config options.

    Returns:
        EvaluationOutput with results.
    """
    from src.evaluation.config import InferenceBackend

    # Build model config
    model_config = ModelConfig(
        model_id=model_id,
        backend=InferenceBackend(backend),
        api_key=kwargs.pop("api_key", None)
        or os.environ.get(f"{backend.upper()}_API_KEY"),
        api_base=kwargs.pop("api_base", None),
        tensor_parallel_size=kwargs.pop("tensor_parallel_size", 1),
        **{k: v for k, v in kwargs.items() if hasattr(ModelConfig, k)},
    )

    # Build evaluation config
    config = EvaluationConfig(
        dataset_id=dataset_id,
        split=split,
        model=model_config,
        batch_size=batch_size,
        max_samples=max_samples,
        output_dir=output_dir,
    )

    # Run pipeline
    pipeline = EvaluationPipeline(config)
    return pipeline.run()
