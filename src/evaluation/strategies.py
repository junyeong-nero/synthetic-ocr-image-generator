import json
import re

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from evaluation.config import FormatType
from evaluation.utils import extract_html_table, parse_model_output_as_json, table_html_to_json


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _canonical_key(key: str) -> str:
    normalized = key.strip()
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")

    aliases = {
        "title": "form_title",
        "document_title": "form_title",
        "formtitle": "form_title",
        "form_no": "form_number",
        "formnumber": "form_number",
        "receipt_no": "receipt_number",
        "receiptno": "receipt_number",
        "invoice_no": "invoice_number",
        "invoiceno": "invoice_number",
        "tel": "phone",
        "telephone": "phone",
        "phone_number": "phone",
        "mobile_number": "mobile",
        "signed_date": "signature_date",
        "signature": "signature_date",
        "birth_date": "signature_date",
        "header_date": "date",
        "qty": "quantity",
        "unitprice": "unit_price",
        "totalprice": "total_price",
    }
    return aliases.get(normalized, normalized)


def _extract_line_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for candidate in ("line_items", "items", "lineItems", "products"):
        raw = payload.get(candidate)
        if isinstance(raw, list):
            normalized_items: list[dict[str, Any]] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                normalized_items.append(
                    {
                        "name": _stringify_value(item.get("name", item.get("item", item.get("product", "")))),
                        "quantity": item.get("quantity", item.get("qty", "")),
                        "unit_price": item.get("unit_price", item.get("price", "")),
                        "total_price": item.get("total_price", item.get("total", "")),
                    }
                )
            return normalized_items
    return []


def _extract_entities(payload: Dict[str, Any]) -> Dict[str, str]:
    entity_candidates = [
        payload.get("entities"),
        payload.get("fields"),
        payload.get("kv_pairs"),
        payload.get("key_value_pairs"),
    ]
    for nested in ("result", "data", "output", "prediction", "response"):
        nested_value = payload.get(nested)
        if isinstance(nested_value, dict):
            entity_candidates.extend(
                [
                    nested_value.get("entities"),
                    nested_value.get("fields"),
                    nested_value.get("kv_pairs"),
                ]
            )

    entities: Dict[str, str] = {}
    for candidate in entity_candidates:
        if not isinstance(candidate, dict):
            continue
        for key, raw_value in candidate.items():
            canonical = _canonical_key(str(key))
            if not canonical:
                continue
            if isinstance(raw_value, dict):
                value = raw_value.get("value", raw_value.get("text", raw_value.get("content", "")))
            else:
                value = raw_value
            entities[canonical] = _stringify_value(value)
        if entities:
            return entities

    ignore_keys = {
        "line_items",
        "items",
        "lineitems",
        "result",
        "data",
        "output",
        "prediction",
        "response",
        "elements",
    }
    for key, value in payload.items():
        canonical = _canonical_key(str(key))
        if canonical in ignore_keys:
            continue
        if isinstance(value, (dict, list)):
            continue
        entities[canonical] = _stringify_value(value)
    return entities


def _normalize_kie_prediction(pred: str, normalize: bool) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    pred_json = parse_model_output_as_json(pred) or {}
    if not normalize:
        pred_entities = pred_json.get("entities", {})
        if isinstance(pred_entities, str):
            pred_entities = _as_dict(pred_entities)
        if not isinstance(pred_entities, dict):
            pred_entities = {}
        pred_items = pred_json.get("line_items", [])
        if not isinstance(pred_items, list):
            pred_items = []
        return {str(k): _stringify_value(v) for k, v in pred_entities.items()}, pred_items

    entities = _extract_entities(pred_json)
    line_items = _extract_line_items(pred_json)
    return entities, line_items


def _normalize_kie_ground_truth(gt: Any, normalize: bool) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    gt_dict = _as_dict(gt)
    true_entities = gt_dict.get("entities", {})
    true_gt = gt_dict.get("ground_truth", {})
    true_gt = _as_dict(true_gt)

    if not normalize:
        true_entities = _as_dict(true_entities)
        true_items = true_gt.get("line_items", [])
        if not isinstance(true_items, list):
            true_items = []
        return {str(k): _stringify_value(v) for k, v in true_entities.items()}, true_items

    merged = {}
    if isinstance(true_entities, dict):
        merged.update(true_entities)
    if isinstance(true_gt.get("entities"), dict):
        merged.update(true_gt.get("entities", {}))

    normalized_entities: Dict[str, str] = {}
    for k, v in merged.items():
        canonical = _canonical_key(str(k))
        if not canonical:
            continue
        if isinstance(v, dict):
            normalized_entities[canonical] = _stringify_value(v.get("value", v.get("text", v.get("content", ""))))
        else:
            normalized_entities[canonical] = _stringify_value(v)

    true_items = _extract_line_items(true_gt)
    return normalized_entities, true_items


def _normalize_table_prediction(pred: str, normalize: bool) -> Tuple[str, Dict[str, Any]]:
    pred_json = parse_model_output_as_json(pred) or {}
    pred_html = extract_html_table(pred)

    if not normalize:
        return pred_html, pred_json if isinstance(pred_json, dict) else {}

    if isinstance(pred_json.get("table"), dict):
        table_obj = pred_json.get("table", {})
        if not pred_html and isinstance(table_obj.get("html"), str):
            pred_html = extract_html_table(table_obj.get("html", ""))
        pred_json = table_obj

    if not pred_html and isinstance(pred_json.get("html"), str):
        pred_html = extract_html_table(pred_json.get("html", ""))

    if not isinstance(pred_json, dict):
        pred_json = {}

    if "cells" not in pred_json and "<table" in pred_html.lower():
        pred_json = {**pred_json, **table_html_to_json(pred_html)}

    pred_json.setdefault("cells", [])
    pred_json.setdefault("num_rows", 0)
    pred_json.setdefault("num_cols", 0)
    return pred_html, pred_json


def _normalize_table_ground_truth(gt: Any, normalize: bool) -> Tuple[str, Dict[str, Any]]:
    gt_dict = _as_dict(gt)
    true_html = gt_dict.get("html", "")
    true_json = gt_dict.get("json", {})
    true_json = _as_dict(true_json)

    if not normalize:
        return str(true_html), true_json

    if "<table" not in str(true_html).lower() and isinstance(true_json.get("html"), str):
        true_html = true_json.get("html", "")

    if "cells" not in true_json and isinstance(true_html, str) and "<table" in true_html.lower():
        true_json = {**true_json, **table_html_to_json(true_html)}

    true_json.setdefault("cells", [])
    true_json.setdefault("num_rows", 0)
    true_json.setdefault("num_cols", 0)
    return str(true_html), true_json


def _normalize_bbox(value: Any) -> Optional[List[float]]:
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "w", "h")):
            x = float(value.get("x", 0))
            y = float(value.get("y", 0))
            w = float(value.get("w", 0))
            h = float(value.get("h", 0))
            return [x, y, x + max(0.0, w), y + max(0.0, h)]
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            return [
                float(value.get("x1", 0)),
                float(value.get("y1", 0)),
                float(value.get("x2", 0)),
                float(value.get("y2", 0)),
            ]
        return None
    if isinstance(value, list) and len(value) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        if x2 < x1 or y2 < y1:
            return [x1, y1, x1 + max(0.0, x2), y1 + max(0.0, y2)]
        return [x1, y1, x2, y2]
    return None


def _normalize_document_elements(raw_elements: Any, normalize: bool) -> List[Dict[str, Any]]:
    if not isinstance(raw_elements, list):
        return []

    if not normalize:
        return [elem for elem in raw_elements if isinstance(elem, dict)]

    normalized: list[dict[str, Any]] = []
    for idx, elem in enumerate(raw_elements):
        if not isinstance(elem, dict):
            continue
        elem_type = elem.get("type", elem.get("label", elem.get("category", "text")))
        bbox = _normalize_bbox(elem.get("bounding_box", elem.get("bbox")))
        metadata = elem.get("metadata")
        table_html = _stringify_value(
            elem.get(
                "html",
                elem.get(
                    "table_html",
                    elem.get("table", {}).get("html", "") if isinstance(elem.get("table"), dict) else "",
                ),
            )
        )
        if not table_html and isinstance(metadata, dict):
            table_html = _stringify_value(
                metadata.get(
                    "html",
                    metadata.get(
                        "table_html",
                        metadata.get("table", {}).get("html", "")
                        if isinstance(metadata.get("table"), dict)
                        else "",
                    ),
                )
            )
        if metadata is None:
            key = elem.get("key")
            value = elem.get("value")
            if key is not None or value is not None:
                metadata = {"key": _stringify_value(key), "value": _stringify_value(value)}
        normalized.append(
            {
                "type": _stringify_value(elem_type) or "text",
                "text": _stringify_value(elem.get("text", elem.get("content", ""))),
                "latex": _stringify_value(elem.get("latex", elem.get("formula", ""))),
                "html": table_html,
                "bounding_box": bbox,
                "reading_order": elem.get("reading_order", elem.get("order", idx)),
                "metadata": metadata if isinstance(metadata, (dict, str)) else {},
            }
        )
    return normalized


def _normalize_document_prediction(pred: str, normalize: bool) -> List[Dict[str, Any]]:
    pred_json = parse_model_output_as_json(pred) or {}
    if not isinstance(pred_json, dict):
        return []
    elements = pred_json.get("elements")
    if elements is None:
        elements = pred_json.get("layout", pred_json.get("blocks", []))
    return _normalize_document_elements(elements, normalize=normalize)


def _normalize_document_ground_truth(gt: Any, normalize: bool) -> List[Dict[str, Any]]:
    gt_dict = _as_dict(gt)
    true_gt = gt_dict.get("ground_truth", gt_dict)
    true_gt = _as_dict(true_gt)
    elements = true_gt.get("elements", true_gt.get("layout", true_gt.get("blocks", [])))
    return _normalize_document_elements(elements, normalize=normalize)


def _is_formula_type(value: str) -> bool:
    lowered = value.strip().lower()
    return any(token in lowered for token in ("formula", "equation", "math"))


def _extract_formula_texts(elements: List[Dict[str, Any]]) -> List[str]:
    formulas: list[tuple[float, str]] = []
    for idx, elem in enumerate(elements):
        if not isinstance(elem, dict):
            continue
        elem_type = _stringify_value(elem.get("type", elem.get("label", elem.get("category", ""))))
        if not _is_formula_type(elem_type):
            continue
        candidate = _stringify_value(
            elem.get(
                "latex",
                elem.get(
                    "formula",
                    elem.get("text", elem.get("content", "")),
                ),
            )
        ).strip()
        if not candidate:
            metadata = elem.get("metadata")
            if isinstance(metadata, dict):
                candidate = _stringify_value(
                    metadata.get(
                        "latex",
                        metadata.get("formula", metadata.get("text", "")),
                    )
                ).strip()
        if not candidate:
            continue
        order = elem.get("reading_order", elem.get("order", idx))
        try:
            sort_order = float(order)
        except (TypeError, ValueError):
            sort_order = float(idx)
        formulas.append((sort_order, candidate))
    formulas.sort(key=lambda item: item[0])
    return [formula for _, formula in formulas]


def _formula_edit_distance(pred_elements: List[Dict[str, Any]], true_elements: List[Dict[str, Any]]) -> float:
    from metrics.edit_distance import cer

    pred_formulas = _extract_formula_texts(pred_elements)
    true_formulas = _extract_formula_texts(true_elements)

    if not pred_formulas and not true_formulas:
        return 0.0

    max_len = max(len(pred_formulas), len(true_formulas))
    if max_len == 0:
        return 0.0

    distances = []
    for idx in range(max_len):
        pred_formula = pred_formulas[idx] if idx < len(pred_formulas) else ""
        true_formula = true_formulas[idx] if idx < len(true_formulas) else ""
        if not pred_formula and not true_formula:
            continue
        if not pred_formula or not true_formula:
            distances.append(1.0)
            continue
        distances.append(cer(true_formula, pred_formula))

    if not distances:
        return 0.0
    return float(np.mean(distances))

class BaseEvaluator(ABC):
    """Abstract base class for format-specific evaluators."""

    @abstractmethod
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        """Extract ground truths from dataset."""
        pass

    @abstractmethod
    def compute_metrics(
        self, predictions: List[str], ground_truths: List[Any], normalize: bool = True
    ) -> Dict[str, float]:
        """Compute metrics comparing predictions and ground truths."""
        pass

    def compute_metric_views(
        self, predictions: List[str], ground_truths: List[Any]
    ) -> Dict[str, Dict[str, float]]:
        return {
            "raw": self.compute_metrics(predictions, ground_truths, normalize=False),
            "normalized": self.compute_metrics(predictions, ground_truths, normalize=True),
        }


class SentenceEvaluator(BaseEvaluator):
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        return list(dataset[target_column])

    def compute_metrics(
        self, predictions: List[str], ground_truths: List[str], normalize: bool = True
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
        self,
        predictions: List[str],
        ground_truths: List[Dict[str, Any]],
        normalize: bool = True,
    ) -> Dict[str, float]:
        from metrics.table_document_metrics import evaluate_table

        teds_scores = []
        cell_accuracies = []
        structure_f1_scores = []

        for pred, gt in zip(predictions, ground_truths):
            pred_html, pred_json = _normalize_table_prediction(pred, normalize=normalize)
            true_html, true_json = _normalize_table_ground_truth(gt, normalize=normalize)

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
        self,
        predictions: List[str],
        ground_truths: List[Dict[str, Any]],
        normalize: bool = True,
    ) -> Dict[str, float]:
        from metrics.table_document_metrics import evaluate_document

        layout_f1_scores = []
        reading_order_scores = []
        kv_f1_scores = []
        text_scores = []
        table_teds_scores = []
        overall_scores = []
        formula_edit_scores = []
        text_table_formula_scores = []

        for pred, gt in zip(predictions, ground_truths):
            pred_elements = _normalize_document_prediction(pred, normalize=normalize)
            true_elements = _normalize_document_ground_truth(gt, normalize=normalize)

            metrics = evaluate_document(pred_elements, true_elements)
            formula_edit_distance = _formula_edit_distance(pred_elements, true_elements)

            layout_metrics = metrics.get("layout_detection", {})
            reading_metrics = metrics.get("reading_order", {})
            kv_metrics = metrics.get("key_value_extraction", {})

            layout_f1_scores.append(layout_metrics.get("overall_f1", 0.0))
            reading_order_scores.append(reading_metrics.get("order_accuracy", 0.0))
            kv_f1_scores.append(kv_metrics.get("f1", 0.0))
            text_scores.append(metrics.get("text_score", 0.0))
            table_teds_scores.append(metrics.get("table_teds", 0.0))
            overall_scores.append(metrics.get("overall_score", metrics.get("overall_f1", 0.0)))
            formula_edit_scores.append(formula_edit_distance)
            text_table_formula_scores.append(
                (
                    metrics.get("text_score", 0.0)
                    + metrics.get("table_teds", 0.0)
                    + (1.0 - formula_edit_distance)
                )
                / 3.0
            )

        return {
            "avg_layout_f1": float(np.mean(layout_f1_scores)),
            "std_layout_f1": float(np.std(layout_f1_scores)),
            "avg_reading_order": float(np.mean(reading_order_scores)),
            "std_reading_order": float(np.std(reading_order_scores)),
            "avg_kv_f1": float(np.mean(kv_f1_scores)),
            "std_kv_f1": float(np.std(kv_f1_scores)),
            "avg_text_score": float(np.mean(text_scores)),
            "std_text_score": float(np.std(text_scores)),
            "avg_table_teds": float(np.mean(table_teds_scores)),
            "std_table_teds": float(np.std(table_teds_scores)),
            "avg_text_table_score": float(np.mean(overall_scores)),
            "std_text_table_score": float(np.std(overall_scores)),
            "avg_formula_edit_distance": float(np.mean(formula_edit_scores)),
            "std_formula_edit_distance": float(np.std(formula_edit_scores)),
            "avg_text_table_formula_score": float(np.mean(text_table_formula_scores)),
            "std_text_table_formula_score": float(np.std(text_table_formula_scores)),
            "avg_overall_f1": float(np.mean(overall_scores)),
            "std_overall_f1": float(np.std(overall_scores)),
        }


class MarkdownEvaluator(BaseEvaluator):
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        return [dataset[i].get("markdown", "") for i in range(len(dataset))]

    def compute_metrics(
        self, predictions: List[str], ground_truths: List[str], normalize: bool = True
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
        self,
        predictions: List[str],
        ground_truths: List[Dict[str, Any]],
        normalize: bool = True,
    ) -> Dict[str, float]:
        from metrics.kie_metrics import evaluate_kie, aggregate_kie_metrics

        all_results = []

        for pred, gt in zip(predictions, ground_truths):
            pred_entities, pred_items = _normalize_kie_prediction(pred, normalize=normalize)
            true_entities, true_items = _normalize_kie_ground_truth(gt, normalize=normalize)

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
