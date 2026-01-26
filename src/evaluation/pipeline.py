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
from evaluation.runner import EvaluationRunner
from evaluation.types import EvaluationOutput, InferenceResult
from models.base import Model, VLMModel
from models.registry import create_model


def parse_model_output_as_json(output: str) -> Optional[Dict]:
    """Parse model output as JSON, handling various formats."""
    if not isinstance(output, str):
        return output if isinstance(output, dict) else None

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    if "```json" in output:
        start = output.find("```json") + 7
        end = output.find("```", start)
        if end > start:
            try:
                return json.loads(output[start:end].strip())
            except json.JSONDecodeError:
                pass

    if "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        if end > start:
            try:
                return json.loads(output[start:end].strip())
            except json.JSONDecodeError:
                pass

    return None


def extract_html_table(output: str) -> str:
    """Extract HTML table from model output."""
    if not isinstance(output, str):
        return str(output)

    output = output.strip()

    if "<table" in output.lower():
        start = output.lower().find("<table")
        end = output.lower().find("</table>")
        if end > start:
            return output[start : end + 8]

    if "```html" in output:
        start = output.find("```html") + 7
        end = output.find("```", start)
        if end > start:
            return output[start:end].strip()

    if "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        if end > start:
            content = output[start:end].strip()
            if "<table" in content.lower():
                return content

    return output


class EvaluationPipeline:
    """
    Main evaluation pipeline orchestrator.

    Handles dataset loading, model inference, and metric computation.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = create_model(config.model)
        self.runner = EvaluationRunner(config, self.model)

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
        """Get prompt for the format type."""
        if self.config.prompt:
            return self.config.prompt
        return DEFAULT_PROMPTS.get(
            self.config.format_type,
            DEFAULT_PROMPTS[FormatType.SENTENCE],
        )

    def _extract_ground_truths(self, dataset: Dataset) -> List[Any]:
        """Extract ground truths based on format type."""
        format_type = self.config.format_type

        if format_type == FormatType.SENTENCE:
            return list(dataset[self.config.target_column])

        elif format_type == FormatType.TABLE:
            return [
                {
                    "html": dataset[i].get("html", ""),
                    "json": dataset[i].get("json", "{}"),
                }
                for i in range(len(dataset))
            ]

        elif format_type == FormatType.DOCUMENT:
            return [
                {"ground_truth": dataset[i].get("ground_truth", "{}")}
                for i in range(len(dataset))
            ]

        elif format_type == FormatType.MARKDOWN:
            return [dataset[i].get("markdown", "") for i in range(len(dataset))]

        elif format_type == FormatType.KIE:
            return [
                {
                    "entities": dataset[i].get("entities", {}),
                    "ground_truth": dataset[i].get("ground_truth", {}),
                    "document_type": dataset[i].get("document_type", ""),
                }
                for i in range(len(dataset))
            ]

        raise ValueError(f"Unknown format type: {format_type}")

    def _compute_metrics(
        self, results: List[InferenceResult]
    ) -> Dict[str, float]:
        """Compute metrics based on format type."""
        from metrics.edit_distance import cer

        # Filter valid results
        valid_results = [r for r in results if r.error is None]
        predictions = [r.prediction for r in valid_results]
        ground_truths = [r.ground_truth for r in valid_results]

        format_type = self.config.format_type

        if format_type == FormatType.SENTENCE:
            return self._compute_sentence_metrics(predictions, ground_truths)

        elif format_type == FormatType.TABLE:
            return self._compute_table_metrics(predictions, ground_truths)

        elif format_type == FormatType.DOCUMENT:
            return self._compute_document_metrics(predictions, ground_truths)

        elif format_type == FormatType.MARKDOWN:
            return self._compute_markdown_metrics(predictions, ground_truths)

        elif format_type == FormatType.KIE:
            return self._compute_kie_metrics(predictions, ground_truths)

        raise ValueError(f"Unknown format type: {format_type}")

    def _compute_sentence_metrics(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Compute sentence-level metrics."""
        from metrics.edit_distance import cer, wer

        cer_list = [cer(gt, pred) for gt, pred in zip(ground_truths, predictions)]
        wer_list = [wer(gt, pred) for gt, pred in zip(ground_truths, predictions)]

        return {
            "avg_cer": float(np.mean(cer_list)),
            "std_cer": float(np.std(cer_list)),
            "min_cer": float(np.min(cer_list)),
            "max_cer": float(np.max(cer_list)),
            "avg_wer": float(np.mean(wer_list)),
            "std_wer": float(np.std(wer_list)),
        }

    def _compute_table_metrics(
        self,
        predictions: List[str],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute table-level metrics."""
        from metrics.table_document_metrics import evaluate_table

        teds_scores = []
        cell_accuracies = []
        structure_f1_scores = []

        for pred, gt in zip(predictions, ground_truths):
            pred_html = extract_html_table(pred)
            pred_json = parse_model_output_as_json(pred) or {}

            true_html = gt.get("html", "")
            true_json = gt.get("json", {})
            if isinstance(true_json, str):
                true_json = json.loads(true_json) if true_json else {}

            metrics = evaluate_table(pred_html, pred_json, true_html, true_json)

            teds_scores.append(metrics.get("teds", 0.0))
            cell_accuracies.append(metrics.get("cell_accuracy", 0.0))
            structure_f1_scores.append(metrics.get("overall_structure_f1", 0.0))

        return {
            "avg_teds": float(np.mean(teds_scores)),
            "std_teds": float(np.std(teds_scores)),
            "avg_cell_accuracy": float(np.mean(cell_accuracies)),
            "std_cell_accuracy": float(np.std(cell_accuracies)),
            "avg_structure_f1": float(np.mean(structure_f1_scores)),
            "std_structure_f1": float(np.std(structure_f1_scores)),
        }

    def _compute_document_metrics(
        self,
        predictions: List[str],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute document-level metrics."""
        from metrics.table_document_metrics import evaluate_document

        layout_f1_scores = []
        reading_order_scores = []
        kv_f1_scores = []
        overall_f1_scores = []

        for pred, gt in zip(predictions, ground_truths):
            pred_json = parse_model_output_as_json(pred) or {}
            pred_elements = pred_json.get("elements", [])

            true_gt = gt.get("ground_truth", gt)
            if isinstance(true_gt, str):
                true_gt = json.loads(true_gt) if true_gt else {}
            true_elements = true_gt.get("elements", [])

            metrics = evaluate_document(pred_elements, true_elements)

            layout_metrics = metrics.get("layout_detection", {})
            reading_metrics = metrics.get("reading_order", {})
            kv_metrics = metrics.get("key_value_extraction", {})

            layout_f1_scores.append(layout_metrics.get("overall_f1", 0.0))
            reading_order_scores.append(reading_metrics.get("order_accuracy", 0.0))
            kv_f1_scores.append(kv_metrics.get("f1", 0.0))
            overall_f1_scores.append(metrics.get("overall_f1", 0.0))

        return {
            "avg_layout_f1": float(np.mean(layout_f1_scores)),
            "std_layout_f1": float(np.std(layout_f1_scores)),
            "avg_reading_order": float(np.mean(reading_order_scores)),
            "std_reading_order": float(np.std(reading_order_scores)),
            "avg_kv_f1": float(np.mean(kv_f1_scores)),
            "std_kv_f1": float(np.std(kv_f1_scores)),
            "avg_overall_f1": float(np.mean(overall_f1_scores)),
            "std_overall_f1": float(np.std(overall_f1_scores)),
        }

    def _compute_markdown_metrics(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Compute markdown-level metrics."""
        from metrics.edit_distance import cer

        cer_list = [cer(gt, pred) for gt, pred in zip(ground_truths, predictions)]

        exact_match_list = [
            1.0 if pred.strip() == gt.strip() else 0.0
            for pred, gt in zip(predictions, ground_truths)
        ]

        def normalize_markdown(text: str) -> str:
            lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
            return "\n".join(lines)

        normalized_match_list = [
            1.0 if normalize_markdown(pred) == normalize_markdown(gt) else 0.0
            for pred, gt in zip(predictions, ground_truths)
        ]

        return {
            "avg_cer": float(np.mean(cer_list)),
            "std_cer": float(np.std(cer_list)),
            "min_cer": float(np.min(cer_list)),
            "max_cer": float(np.max(cer_list)),
            "exact_match_rate": float(np.mean(exact_match_list)),
            "normalized_match_rate": float(np.mean(normalized_match_list)),
        }

    def _compute_kie_metrics(
        self,
        predictions: List[str],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute KIE (Key Information Extraction) metrics."""
        from metrics.kie_metrics import evaluate_kie, aggregate_kie_metrics

        all_results = []

        for pred, gt in zip(predictions, ground_truths):
            # Parse prediction
            pred_json = parse_model_output_as_json(pred) or {}
            pred_entities = pred_json.get("entities", {})
            pred_items = pred_json.get("line_items", [])

            # Handle different ground truth formats
            if isinstance(pred_entities, str):
                try:
                    pred_entities = json.loads(pred_entities)
                except (json.JSONDecodeError, TypeError):
                    pred_entities = {}

            # Extract ground truth entities
            true_entities = gt.get("entities", {})
            if isinstance(true_entities, str):
                try:
                    true_entities = json.loads(true_entities)
                except (json.JSONDecodeError, TypeError):
                    true_entities = {}

            # Extract ground truth from nested structure if needed
            true_gt = gt.get("ground_truth", {})
            if isinstance(true_gt, str):
                try:
                    true_gt = json.loads(true_gt)
                except (json.JSONDecodeError, TypeError):
                    true_gt = {}

            # Get line items from ground truth
            true_items = true_gt.get("line_items", [])

            # If entities not directly available, extract from ground_truth
            if not true_entities and true_gt:
                gt_entities = true_gt.get("entities", {})
                if isinstance(gt_entities, dict):
                    # Extract values from nested structure
                    true_entities = {
                        k: v.get("value", v) if isinstance(v, dict) else v
                        for k, v in gt_entities.items()
                    }

            # Evaluate this sample
            result = evaluate_kie(
                pred_entities=pred_entities,
                true_entities=true_entities,
                pred_items=pred_items,
                true_items=true_items,
            )
            all_results.append(result)

        # Aggregate results
        aggregated = aggregate_kie_metrics(all_results)

        return aggregated

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
