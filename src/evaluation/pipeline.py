"""Main evaluation pipeline orchestrator."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from evaluation.config import (
    DEFAULT_PROMPTS,
    FORMAT_OUTPUT_CONTRACTS,
    EvaluationConfig,
    FormatType,
    ModelConfig,
)
from evaluation.model_config import ModelConfigLoader, ModelSpecificConfig
from evaluation.runner import EvaluationRunner
from evaluation.types import EvaluationOutput, InferenceResult
from evaluation.strategies import EvaluatorRegistry
from models.registry import create_model
from env_utils import get_environment_metadata


class EvaluationPipeline:
    """
    Main evaluation pipeline orchestrator.

    Handles dataset loading, model inference, and metric computation.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = create_model(config.model)
        self.runner = EvaluationRunner(config, self.model)
        self.prompt: Optional[str] = None
        self.system_prompt: Optional[str] = None
        self.prompt_source: Optional[str] = None
        self.metric_views: Dict[str, Dict[str, float]] = {}
        self.dataset_fingerprint: Optional[str] = None
        self.dataset_info: Optional[Any] = None

        # Load model-specific config if available
        self.model_specific_config: Optional[ModelSpecificConfig] = None
        if config.model_config_path:
            loader = ModelConfigLoader()
            self.model_specific_config = loader.load_from_path(
                Path(config.model_config_path)
            )
        else:
            # Try to auto-load based on model_id
            loader = ModelConfigLoader()
            self.model_specific_config = loader.load(config.model.model_id)

    def _load_dataset(self) -> Dataset:
        """Load dataset from HuggingFace."""
        if self.config.subset == "default":
            dataset = load_dataset(
                self.config.dataset_id,
                split=self.config.split,
            )
        else:
            dataset = load_dataset(
                self.config.dataset_id,
                name=self.config.subset,
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
            format_key = self.config.format_type.value
            subset_config = self.model_specific_config.subsets.get(self.config.subset)
            if subset_config and format_key in subset_config.prompts:
                prompt_config = subset_config.prompts[format_key]
                base_prompt = prompt_config.prompt
                base_system_prompt = prompt_config.system_prompt
                source = "model_config_subset"
            elif format_key in self.model_specific_config.prompts:
                prompt_config = self.model_specific_config.prompts[format_key]
                base_prompt = prompt_config.prompt
                base_system_prompt = prompt_config.system_prompt
                source = "model_config_format"

        if base_prompt is None:
            base_prompt = DEFAULT_PROMPTS.get(
                self.config.format_type,
                DEFAULT_PROMPTS[FormatType.SENTENCE],
            )
            base_system_prompt = None
            source = "default"

        contract = FORMAT_OUTPUT_CONTRACTS.get(self.config.format_type)
        if contract and contract.strip() not in base_prompt:
            base_prompt = f"{base_prompt.rstrip()}\n\n{contract}"
            source = f"{source}_with_contract"

        return base_prompt, base_system_prompt, source or "default"

    def _extract_ground_truths(self, dataset: Dataset) -> List[Any]:
        """Extract ground truths based on format type."""
        evaluator = EvaluatorRegistry.get_evaluator(self.config.format_type)
        return evaluator.extract_ground_truths(dataset, self.config.target_column)

    def _compute_metrics(
        self, results: List[InferenceResult]
    ) -> Dict[str, float]:
        """Compute metrics based on format type."""

        # Filter valid results
        valid_results = [
            r for r in results if r.error is None and r.prediction is not None
        ]
        if not valid_results:
            self.metric_views = {"raw": {}, "normalized": {}}
            return {}

        predictions = [str(r.prediction) for r in valid_results]
        ground_truths = [r.ground_truth for r in valid_results]

        evaluator = EvaluatorRegistry.get_evaluator(self.config.format_type)
        metric_views = evaluator.compute_metric_views(predictions, ground_truths)
        self.metric_views = metric_views
        return metric_views.get("normalized", {})


    async def run_async(self) -> EvaluationOutput:
        """Run the evaluation pipeline asynchronously."""
        # Load dataset
        print(f"Loading dataset: {self.config.dataset_id}")
        dataset = self._load_dataset()
        print(f"Loaded {len(dataset)} samples")

        # Extract data
        images = dataset[self.config.image_column]
        ground_truths = self._extract_ground_truths(dataset)
        prompt, system_prompt, prompt_source = self._resolve_prompt()
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.prompt_source = prompt_source
        prompts = [prompt] * len(images)

        # Run inference
        results = await self.runner.run_async(images, ground_truths, prompts)

        # Compute metrics
        metrics = self._compute_metrics(results)

        # Build summary
        successful = len([r for r in results if r.error is None])
        failed = len([r for r in results if r.error is not None])
        avg_latency = (
            sum(r.latency_ms for r in results) / len(results) if results else 0
        )

        # Build output
        return EvaluationOutput(
            config=self._config_to_dict(),
            metrics=metrics,
            metric_views=getattr(self, "metric_views", {}),
            per_sample_results=[r.to_dict() for r in results],
            summary={
                "total_samples": len(results),
                "successful": successful,
                "failed": failed,
                "avg_latency_ms": avg_latency,
                "model_id": self.config.model.model_id,
                "backend": self.config.model.backend.value,
            },
        )

    def run(self) -> EvaluationOutput:
        """Run the evaluation pipeline synchronously."""
        # Load dataset
        print(f"Loading dataset: {self.config.dataset_id}")
        dataset = self._load_dataset()
        print(f"Loaded {len(dataset)} samples")

        # Extract data
        images = dataset[self.config.image_column]
        ground_truths = self._extract_ground_truths(dataset)
        prompt, system_prompt, prompt_source = self._resolve_prompt()
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.prompt_source = prompt_source
        prompts = [prompt] * len(images)

        # Run inference
        results = self.runner.run(images, ground_truths, prompts)

        # Compute metrics
        metrics = self._compute_metrics(results)

        # Build summary
        successful = len([r for r in results if r.error is None])
        failed = len([r for r in results if r.error is not None])
        avg_latency = (
            sum(r.latency_ms for r in results) / len(results) if results else 0
        )

        # Build output
        return EvaluationOutput(
            config=self._config_to_dict(),
            metrics=metrics,
            metric_views=getattr(self, "metric_views", {}),
            per_sample_results=[r.to_dict() for r in results],
            summary={
                "total_samples": len(results),
                "successful": successful,
                "failed": failed,
                "avg_latency_ms": avg_latency,
                "model_id": self.config.model.model_id,
                "backend": self.config.model.backend.value,
            },
        )

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
            "subset": self.config.subset,
            "split": self.config.split,
            "format_type": self.config.format_type.value,
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
    format_type: str = "sentence",
    subset: str = "default",
    split: str = "test",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    output_dir: str = "./evaluation_results",
    **kwargs,
) -> EvaluationOutput:
    """
    Convenience function to run evaluation pipeline.

    Args:
        dataset_id: HuggingFace dataset ID.
        model_id: Model identifier.
        backend: Inference backend name.
        format_type: Evaluation format type.
        subset: Dataset subset.
        split: Dataset split.
        batch_size: Batch size.
        max_samples: Maximum samples to evaluate.
        output_dir: Output directory.
        **kwargs: Additional config options.

    Returns:
        EvaluationOutput with results.
    """
    from evaluation.config import InferenceBackend

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
        subset=subset,
        split=split,
        format_type=FormatType(format_type),
        model=model_config,
        batch_size=batch_size,
        max_samples=max_samples,
        output_dir=output_dir,
    )

    # Run pipeline
    pipeline = EvaluationPipeline(config)
    return pipeline.run()
