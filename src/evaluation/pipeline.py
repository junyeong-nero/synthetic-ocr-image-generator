"""Main evaluation pipeline orchestrator."""

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset, load_dataset
from PIL import Image

from evaluation.config import DEFAULT_PROMPTS, EvaluationConfig, FormatType, ModelConfig
from evaluation.model_config import ModelConfigLoader, ModelSpecificConfig
from evaluation.runner import EvaluationRunner
from evaluation.types import EvaluationOutput, InferenceResult
from evaluation.strategies import EvaluatorRegistry
from models.base import Model, VLMModel
from models.registry import create_model
from evaluation.utils import extract_html_table, parse_model_output_as_json


class EvaluationPipeline:
    """
    Main evaluation pipeline orchestrator.

    Handles dataset loading, model inference, and metric computation.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = create_model(config.model)
        self.runner = EvaluationRunner(config, self.model)

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

        # Apply max_samples limit
        if self.config.max_samples:
            dataset = dataset.select(
                range(min(self.config.max_samples, len(dataset)))
            )

        return dataset

    def _get_prompt(self) -> str:
        """Get prompt for the format type.

        Priority:
        1. Custom prompt from config
        2. Model-specific prompt (with subset override)
        3. Default prompt for format type
        """
        # 1. Custom prompt from config takes highest priority
        if self.config.prompt:
            return self.config.prompt

        # 2. Try model-specific prompt
        if self.model_specific_config:
            prompt_config = self.model_specific_config.get_prompt(
                self.config.format_type,
                subset=self.config.subset,
            )
            if prompt_config:
                return prompt_config.prompt

        # 3. Fall back to default prompt
        return DEFAULT_PROMPTS.get(
            self.config.format_type,
            DEFAULT_PROMPTS[FormatType.SENTENCE],
        )

    def _extract_ground_truths(self, dataset: Dataset) -> List[Any]:
        """Extract ground truths based on format type."""
        evaluator = EvaluatorRegistry.get_evaluator(self.config.format_type)
        return evaluator.extract_ground_truths(dataset, self.config.target_column)

    def _compute_metrics(
        self, results: List[InferenceResult]
    ) -> Dict[str, float]:
        """Compute metrics based on format type."""
        
        # Filter valid results
        valid_results = [r for r in results if r.error is None]
        predictions = [r.prediction for r in valid_results]
        ground_truths = [r.ground_truth for r in valid_results]

        evaluator = EvaluatorRegistry.get_evaluator(self.config.format_type)
        return evaluator.compute_metrics(predictions, ground_truths)


    async def run_async(self) -> EvaluationOutput:
        """Run the evaluation pipeline asynchronously."""
        # Load dataset
        print(f"Loading dataset: {self.config.dataset_id}")
        dataset = self._load_dataset()
        print(f"Loaded {len(dataset)} samples")

        # Extract data
        images = list(dataset[self.config.image_column])
        ground_truths = self._extract_ground_truths(dataset)
        prompt = self._get_prompt()
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
        images = list(dataset[self.config.image_column])
        ground_truths = self._extract_ground_truths(dataset)
        prompt = self._get_prompt()
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
        return {
            "dataset_id": self.config.dataset_id,
            "subset": self.config.subset,
            "split": self.config.split,
            "format_type": self.config.format_type.value,
            "batch_size": self.config.batch_size,
            "max_samples": self.config.max_samples,
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
