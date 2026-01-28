import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset

from evaluation.config import FormatType
from evaluation.types import InferenceResult
from evaluation.utils import extract_html_table, parse_model_output_as_json

class BaseEvaluator(ABC):
    """Abstract base class for format-specific evaluators."""

    @abstractmethod
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        """Extract ground truths from dataset."""
        pass

    @abstractmethod
    def compute_metrics(
        self, predictions: List[str], ground_truths: List[Any]
    ) -> Dict[str, float]:
        """Compute metrics comparing predictions and ground truths."""
        pass


class SentenceEvaluator(BaseEvaluator):
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        return list(dataset[target_column])

    def compute_metrics(
        self, predictions: List[str], ground_truths: List[str]
    ) -> Dict[str, float]:
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


class TableEvaluator(BaseEvaluator):
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        return [
            {
                "html": dataset[i].get("html", ""),
                "json": dataset[i].get("json", "{}"),
            }
            for i in range(len(dataset))
        ]

    def compute_metrics(
        self, predictions: List[str], ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
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


class DocumentEvaluator(BaseEvaluator):
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        return [
            {"ground_truth": dataset[i].get("ground_truth", "{}")}
            for i in range(len(dataset))
        ]

    def compute_metrics(
        self, predictions: List[str], ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
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


class MarkdownEvaluator(BaseEvaluator):
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        return [dataset[i].get("markdown", "") for i in range(len(dataset))]

    def compute_metrics(
        self, predictions: List[str], ground_truths: List[str]
    ) -> Dict[str, float]:
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


class KIEEvaluator(BaseEvaluator):
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        return [
            {
                "entities": dataset[i].get("entities", {}),
                "ground_truth": dataset[i].get("ground_truth", {}),
                "document_type": dataset[i].get("document_type", ""),
            }
            for i in range(len(dataset))
        ]

    def compute_metrics(
        self, predictions: List[str], ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
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


class EvaluatorRegistry:
    _evaluators: Dict[FormatType, BaseEvaluator] = {}

    @classmethod
    def register(cls, format_type: FormatType, evaluator: BaseEvaluator):
        cls._evaluators[format_type] = evaluator

    @classmethod
    def get_evaluator(cls, format_type: FormatType) -> BaseEvaluator:
        if format_type not in cls._evaluators:
            raise ValueError(f"No evaluator registered for format: {format_type}")
        return cls._evaluators[format_type]

# Register default evaluators
EvaluatorRegistry.register(FormatType.SENTENCE, SentenceEvaluator())
EvaluatorRegistry.register(FormatType.TABLE, TableEvaluator())
EvaluatorRegistry.register(FormatType.DOCUMENT, DocumentEvaluator())
EvaluatorRegistry.register(FormatType.MARKDOWN, MarkdownEvaluator())
EvaluatorRegistry.register(FormatType.KIE, KIEEvaluator())
