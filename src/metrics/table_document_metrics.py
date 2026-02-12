# -*- coding: utf-8 -*-
"""
Table and Document Evaluation Metrics.

This module provides metrics for evaluating OCR and document understanding results:
- Table metrics: TEDS, cell-level accuracy, row/column detection
- Document metrics: layout mAP, reading order accuracy, key-value extraction F1
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict
import json
from scipy import stats


# ==================== Table Metrics ====================

def cell_level_text_accuracy(pred_json: Dict, true_json: Dict) -> Dict[str, Any]:
    """
    Calculate cell-level text accuracy using Character Error Rate (CER).

    Returns:
        - avg_cer: Average CER across all cells
        - cell_acc: Percentage of cells with 0 CER (perfect match)
        - cell_details: List of per-cell CER scores
    """
    from .edit_distance import cer

    true_cells = {
        (cell.get("row"), cell.get("col")): cell
        for cell in true_json.get("cells", [])
        if isinstance(cell, dict)
    }
    pred_cells = {
        (cell.get("row"), cell.get("col")): cell
        for cell in pred_json.get("cells", [])
        if isinstance(cell, dict)
    }

    all_cer_scores = []
    perfect_cells = 0
    total_chars = 0
    errors = 0

    for cell_key in sorted(set(true_cells.keys()) & set(pred_cells.keys())):
        true_cell = true_cells.get(cell_key)
        pred_cell = pred_cells.get(cell_key)

        if true_cell and pred_cell:
            true_text = true_cell.get("text", "")
            pred_text = pred_cell.get("text", "")

            if true_text or pred_text:
                cell_cer = cer(true_text, pred_text)
                all_cer_scores.append(cell_cer)
                total_chars += len(true_text)

                if cell_cer == 0:
                    perfect_cells += 1
                else:
                    errors += cell_cer * len(true_text)

    avg_cer = errors / total_chars if total_chars > 0 else 0.0
    total_cells = len(all_cer_scores)
    cell_acc = perfect_cells / total_cells if total_cells > 0 else 0.0

    return {
        "avg_cer": avg_cer,
        "cell_accuracy": cell_acc,
        "perfect_cells": perfect_cells,
        "total_cells": total_cells,
        "cell_cer_scores": all_cer_scores,
    }


def row_column_detection_metrics(pred_json: Dict, true_json: Dict) -> Dict[str, Any]:
    """
    Calculate row and column detection metrics.

    For row detection: checks if predicted row boundaries match true boundaries
    For column detection: checks if predicted column boundaries match true boundaries

    Returns:
        - row_precision, row_recall, row_f1
        - col_precision, col_recall, col_f1
        - overall_f1
    """
    true_num_rows = true_json.get("num_rows", 0)
    pred_num_rows = pred_json.get("num_rows", 0)
    true_num_cols = true_json.get("num_cols", 0)
    pred_num_cols = pred_json.get("num_cols", 0)

    # Row detection metrics
    row_tp = min(true_num_rows, pred_num_rows)
    row_fp = max(0, pred_num_rows - true_num_rows)
    row_fn = max(0, true_num_rows - pred_num_rows)

    row_precision = row_tp / (row_tp + row_fp) if (row_tp + row_fp) > 0 else 0.0
    row_recall = row_tp / (row_tp + row_fn) if (row_tp + row_fn) > 0 else 0.0
    row_f1 = 2 * row_precision * row_recall / (row_precision + row_recall) if (row_precision + row_recall) > 0 else 0.0

    # Column detection metrics
    col_tp = min(true_num_cols, pred_num_cols)
    col_fp = max(0, pred_num_cols - true_num_cols)
    col_fn = max(0, true_num_cols - pred_num_cols)

    col_precision = col_tp / (col_tp + col_fp) if (col_tp + col_fp) > 0 else 0.0
    col_recall = col_tp / (col_tp + col_fn) if (col_tp + col_fn) > 0 else 0.0
    col_f1 = 2 * col_precision * col_recall / (col_precision + col_recall) if (col_precision + col_recall) > 0 else 0.0

    # Overall structure F1
    overall_f1 = 2 * row_f1 * col_f1 / (row_f1 + col_f1) if (row_f1 + col_f1) > 0 else 0.0

    return {
        "row_detection": {
            "true_count": true_num_rows,
            "pred_count": pred_num_rows,
            "precision": row_precision,
            "recall": row_recall,
            "f1": row_f1,
        },
        "column_detection": {
            "true_count": true_num_cols,
            "pred_count": pred_num_cols,
            "precision": col_precision,
            "recall": col_recall,
            "f1": col_f1,
        },
        "overall_structure_f1": overall_f1,
    }


def evaluate_table(
    pred_html: str,
    pred_json: Dict,
    true_html: str,
    true_json: Dict,
) -> Dict[str, Any]:
    """
    Comprehensive table evaluation.

    Combines TEDS, cell-level text accuracy, and row/column detection metrics.
    """
    from .table_edit_distance import TEDS

    # TEDS score
    teds_calculator = TEDS(structure_only=True)
    teds_result = teds_calculator.evaluate(pred_html, true_html)

    # Cell-level text accuracy
    cell_accuracy = cell_level_text_accuracy(pred_json, true_json)

    # Row/column detection
    row_col_metrics = row_column_detection_metrics(pred_json, true_json)

    return {
        "teds": teds_result.get("teds", 0.0),
        "teds_error": teds_result.get("error", None),
        **cell_accuracy,
        **row_col_metrics,
    }


# ==================== Document Metrics ====================

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Boxes are in format: (x1, y1, x2, y2)
    """
    if box1 is None or box2 is None:
        return 0.0

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _canonical_type(raw_type: Any) -> str:
    text = str(raw_type or "text").strip().lower()
    return text.replace("-", "_").replace(" ", "_")


def _is_formula_type(raw_type: Any) -> bool:
    elem_type = _canonical_type(raw_type)
    return any(token in elem_type for token in ("formula", "equation", "latex", "math"))


def _is_table_type(raw_type: Any) -> bool:
    return "table" in _canonical_type(raw_type)


def _is_text_type(raw_type: Any) -> bool:
    return not _is_table_type(raw_type) and not _is_formula_type(raw_type)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_table_html(element: Dict[str, Any]) -> str:
    html = _normalize_text(element.get("html", element.get("table_html", "")))
    if html:
        return html

    metadata = element.get("metadata", {})
    if isinstance(metadata, dict):
        html = _normalize_text(metadata.get("html", metadata.get("table_html", "")))
        if html:
            return html
        table_payload = metadata.get("table")
        if isinstance(table_payload, dict):
            html = _normalize_text(table_payload.get("html", ""))
            if html:
                return html

    table_payload = element.get("table")
    if isinstance(table_payload, dict):
        html = _normalize_text(table_payload.get("html", ""))
        if html:
            return html

    raw_text = _normalize_text(element.get("text", ""))
    return raw_text if "<table" in raw_text.lower() else ""


def _split_document_elements(elements: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    text_elements: list[dict[str, Any]] = []
    table_elements: list[dict[str, Any]] = []

    for element in elements:
        elem_type = element.get("type", "text")
        if _is_formula_type(elem_type):
            continue
        if _is_table_type(elem_type):
            table_elements.append(element)
        elif _is_text_type(elem_type):
            text_elements.append(element)

    return text_elements, table_elements


def _pair_elements_by_layout(
    pred_elements: List[Dict[str, Any]],
    true_elements: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], int, int]:
    if not pred_elements and not true_elements:
        return [], 0, 0

    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pred_used: set[int] = set()
    true_used: set[int] = set()

    for pred_idx, pred_elem in enumerate(pred_elements):
        pred_box = pred_elem.get("bounding_box")
        if pred_box is None:
            continue

        best_true_idx = -1
        best_iou = 0.0
        for true_idx, true_elem in enumerate(true_elements):
            if true_idx in true_used:
                continue
            true_box = true_elem.get("bounding_box")
            if true_box is None:
                continue
            iou = calculate_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
                best_true_idx = true_idx

        if best_true_idx >= 0 and best_iou >= iou_threshold:
            matched_pairs.append((pred_elem, true_elements[best_true_idx]))
            pred_used.add(pred_idx)
            true_used.add(best_true_idx)

    remaining_pred_idx = [idx for idx in range(len(pred_elements)) if idx not in pred_used]
    remaining_true_idx = [idx for idx in range(len(true_elements)) if idx not in true_used]

    no_bbox_pred_idx = [idx for idx in remaining_pred_idx if pred_elements[idx].get("bounding_box") is None]
    no_bbox_true_idx = [idx for idx in remaining_true_idx if true_elements[idx].get("bounding_box") is None]

    fallback_count = min(len(no_bbox_pred_idx), len(no_bbox_true_idx))
    for offset in range(fallback_count):
        pred_idx = no_bbox_pred_idx[offset]
        true_idx = no_bbox_true_idx[offset]
        matched_pairs.append((pred_elements[pred_idx], true_elements[true_idx]))
        pred_used.add(pred_idx)
        true_used.add(true_idx)

    unmatched_pred = len(pred_elements) - len(pred_used)
    unmatched_true = len(true_elements) - len(true_used)
    return matched_pairs, unmatched_pred, unmatched_true


def _compute_text_score(
    pred_text_elements: List[Dict[str, Any]],
    true_text_elements: List[Dict[str, Any]],
    iou_threshold: float,
) -> Dict[str, Any]:
    from .edit_distance import cer

    pairs, unmatched_pred, unmatched_true = _pair_elements_by_layout(
        pred_text_elements,
        true_text_elements,
        iou_threshold=iou_threshold,
    )

    if not pred_text_elements and not true_text_elements:
        return {
            "score": 1.0,
            "matched_pairs": 0,
            "unmatched_pred": 0,
            "unmatched_true": 0,
            "pair_scores": [],
        }

    pair_scores: list[float] = []
    for pred_elem, true_elem in pairs:
        pred_text = _normalize_text(pred_elem.get("text", ""))
        true_text = _normalize_text(true_elem.get("text", ""))
        if not pred_text and not true_text:
            pair_scores.append(1.0)
            continue
        pair_scores.append(max(0.0, 1.0 - cer(true_text, pred_text)))

    denominator = len(pairs) + unmatched_pred + unmatched_true
    score = float(sum(pair_scores) / denominator) if denominator else 1.0
    return {
        "score": score,
        "matched_pairs": len(pairs),
        "unmatched_pred": unmatched_pred,
        "unmatched_true": unmatched_true,
        "pair_scores": pair_scores,
    }


def _compute_table_teds_score(
    pred_table_elements: List[Dict[str, Any]],
    true_table_elements: List[Dict[str, Any]],
    iou_threshold: float,
) -> Dict[str, Any]:
    from .table_edit_distance import TEDS

    pairs, unmatched_pred, unmatched_true = _pair_elements_by_layout(
        pred_table_elements,
        true_table_elements,
        iou_threshold=iou_threshold,
    )

    if not pred_table_elements and not true_table_elements:
        return {
            "score": 1.0,
            "matched_pairs": 0,
            "unmatched_pred": 0,
            "unmatched_true": 0,
            "pair_scores": [],
        }

    teds = TEDS(structure_only=True)
    pair_scores: list[float] = []
    for pred_elem, true_elem in pairs:
        pred_html = _extract_table_html(pred_elem)
        true_html = _extract_table_html(true_elem)
        if not pred_html and not true_html:
            pair_scores.append(1.0)
            continue
        if not pred_html or not true_html:
            pair_scores.append(0.0)
            continue
        teds_result = teds.evaluate(pred_html, true_html)
        pair_scores.append(float(teds_result.get("teds", 0.0)))

    denominator = len(pairs) + unmatched_pred + unmatched_true
    score = float(sum(pair_scores) / denominator) if denominator else 1.0
    return {
        "score": score,
        "matched_pairs": len(pairs),
        "unmatched_pred": unmatched_pred,
        "unmatched_true": unmatched_true,
        "pair_scores": pair_scores,
    }


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(numeric) else numeric


def layout_detection_map(
    pred_elements: List[Dict],
    true_elements: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calculate mean Average Precision (mAP) for layout detection.

    Args:
        pred_elements: List of predicted elements with bounding_box and type
        true_elements: List of ground truth elements with bounding_box and type
        iou_threshold: IoU threshold for considering a detection as correct

    Returns:
        - map: Mean Average Precision
        - per_type_ap: AP for each element type
        - total_precision, total_recall, total_f1
    """
    if not true_elements:
        return {"map": 0.0, "error": "No ground truth elements"}

    type_to_true = defaultdict(list)
    for elem in true_elements:
        type_to_true[elem.get("type", "unknown")].append(elem)

    type_to_pred = defaultdict(list)
    for elem in pred_elements:
        type_to_pred[elem.get("type", "unknown")].append(elem)

    all_types = set(type_to_true.keys()) | set(type_to_pred.keys())
    per_type_ap = {}

    for elem_type in all_types:
        true_elems = type_to_true.get(elem_type, [])
        pred_elems = type_to_pred.get(elem_type, [])

        if not true_elems:
            # No ground truth for this type
            continue

        # Match predictions to ground truth
        true_matched = set()
        pred_scores = []  # (confidence, matched)

        for pred_idx, pred_elem in enumerate(pred_elems):
            best_iou = 0
            best_true_idx = -1

            for true_idx, true_elem in enumerate(true_elems):
                if true_idx in true_matched:
                    continue
                iou = calculate_iou(
                    pred_elem.get("bounding_box"),
                    true_elem.get("bounding_box"),
                )
                if iou > best_iou:
                    best_iou = iou
                    best_true_idx = true_idx

            matched = best_iou >= iou_threshold
            if matched:
                true_matched.add(best_true_idx)

            pred_scores.append((1.0, matched))  # Assuming confidence = 1.0

        # Calculate precision/recall for this type
        if pred_scores:
            tp = sum(1 for _, matched in pred_scores if matched)
            fp = len(pred_scores) - tp
            fn = len(true_elems) - len(true_matched)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_type_ap[elem_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_count": len(true_elems),
                "pred_count": len(pred_elems),
            }

    # Calculate mAP (simple average of APs)
    if per_type_ap:
        ap_values = [v.get("f1", 0.0) for v in per_type_ap.values()]
        mAP = np.mean(ap_values)

        # Overall metrics
        total_tp = sum(v.get("precision", 0) * v.get("pred_count", 0) for v in per_type_ap.values())
        total_pred = sum(v.get("pred_count", 0) for v in per_type_ap.values())
        total_fn = sum(v.get("recall", 0) * v.get("true_count", 0) for v in per_type_ap.values())
        total_true = sum(v.get("true_count", 0) for v in per_type_ap.values())

        overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
        overall_recall = total_fn / total_true if total_true > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    else:
        mAP = 0.0
        overall_precision = 0.0
        overall_recall = 0.0
        overall_f1 = 0.0

    return {
        "map": mAP,
        "per_type_metrics": per_type_ap,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
    }


def reading_order_accuracy(pred_elements: List[Dict], true_elements: List[Dict]) -> Dict[str, Any]:
    """
    Calculate reading order accuracy using Kendall's tau correlation.

    Args:
        pred_elements: Predicted elements with reading_order
        true_elements: Ground truth elements with reading_order

    Returns:
        - kendall_tau: Kendall's tau correlation coefficient
        - spearman_rho: Spearman's rho correlation
        - order_accuracy: Percentage of correctly ordered adjacent pairs
    """
    if not pred_elements or not true_elements:
        return {"kendall_tau": 0.0, "spearman_rho": 0.0, "order_accuracy": 0.0}

    # Extract reading orders
    pred_orders = [e.get("reading_order", i) for i, e in enumerate(pred_elements)]
    true_orders = [e.get("reading_order", i) for i, e in enumerate(true_elements)]

    # Ensure same length
    min_len = min(len(pred_orders), len(true_orders))
    pred_orders = pred_orders[:min_len]
    true_orders = true_orders[:min_len]

    if min_len < 2:
        return {"kendall_tau": 1.0, "spearman_rho": 1.0, "order_accuracy": 1.0}

    # Kendall's tau
    kendall_tau, _ = stats.kendalltau(pred_orders, true_orders)

    # Spearman's rho
    spearman_rho, _ = stats.spearmanr(pred_orders, true_orders)

    # Order accuracy (percentage of adjacent pairs in correct order)
    correct_pairs = 0
    total_pairs = min_len - 1

    for i in range(total_pairs):
        pred_diff = pred_orders[i + 1] - pred_orders[i]
        true_diff = true_orders[i + 1] - true_orders[i]

        # Check if the direction of change is the same
        if (pred_diff > 0 and true_diff > 0) or (pred_diff < 0 and true_diff < 0):
            correct_pairs += 1

    order_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 1.0

    return {
        "kendall_tau": _safe_float(kendall_tau),
        "spearman_rho": _safe_float(spearman_rho),
        "order_accuracy": order_accuracy,
    }


def key_value_extraction_f1(pred_elements: List[Dict], true_elements: List[Dict]) -> Dict[str, Any]:
    """
    Calculate F1 score for key-value extraction.

    Matches key-value pairs and calculates precision/recall/F1.

    Args:
        pred_elements: Predicted elements with key-value pairs
        true_elements: Ground truth elements with key-value pairs

    Returns:
        - precision: Key-value extraction precision
        - recall: Key-value extraction recall
        - f1: F1 score
        - matched_pairs: Number of correctly matched pairs
        - missing_pairs: Number of ground truth pairs not found
        - extra_pairs: Number of predicted pairs not matched
    """
    from .edit_distance import cer

    # Extract key-value pairs
    def extract_kv_pairs(elements: List[Dict]) -> List[Tuple[str, str, Tuple]]:
        pairs = []
        for elem in elements:
            # Look for key-value patterns in metadata
            metadata = elem.get("metadata", {})
            if isinstance(metadata, dict):
                key = metadata.get("key", elem.get("text", ""))
                value = metadata.get("value", "")
                bbox = elem.get("bounding_box")
                pairs.append((key, value, bbox))
            else:
                # Try to parse as JSON
                try:
                    kv = json.loads(metadata) if isinstance(metadata, str) else metadata
                    key = kv.get("key", elem.get("text", ""))
                    value = kv.get("value", "")
                    bbox = elem.get("bounding_box")
                    pairs.append((key, value, bbox))
                except (ValueError, json.JSONDecodeError):
                    # Use element text as key-value
                    key = elem.get("text", "")
                    value = ""
                    bbox = elem.get("bounding_box")
                    pairs.append((key, value, bbox))
        return pairs

    true_pairs = extract_kv_pairs(true_elements)
    pred_pairs = extract_kv_pairs(pred_elements)

    if not true_pairs:
        return {"precision": 1.0 if not pred_pairs else 0.0, "recall": 1.0, "f1": 1.0 if not pred_pairs else 0.0}

    # Match predictions to ground truth
    matched = set()
    true_used = set()
    pred_used = set()

    for pred_idx, (p_key, p_value, p_bbox) in enumerate(pred_pairs):
        best_match_idx = -1
        best_score = 0

        for true_idx, (t_key, t_value, t_bbox) in enumerate(true_pairs):
            if true_idx in true_used:
                continue

            # Check key similarity
            key_cer = cer(p_key, t_key) if p_key and t_key else 1.0
            value_cer = cer(p_value, t_value) if p_value and t_value else 1.0

            # Combined score (lower CER is better)
            avg_cer = (key_cer + value_cer) / 2
            match_score = 1.0 - avg_cer

            if match_score > best_score:
                best_score = match_score
                best_match_idx = true_idx

        if best_match_idx >= 0 and best_score >= 0.7:  # Threshold for matching
            matched.add((pred_idx, best_match_idx))
            true_used.add(best_match_idx)
            pred_used.add(pred_idx)

    # Calculate metrics
    tp = len(matched)
    fp = len(pred_pairs) - len(pred_used)
    fn = len(true_pairs) - len(true_used)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_pairs": tp,
        "missing_pairs": fn,
        "extra_pairs": fp,
        "true_pair_count": len(true_pairs),
        "pred_pair_count": len(pred_pairs),
    }


def evaluate_document(
    pred_elements: List[Dict],
    true_elements: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    pred_text_elements, pred_table_elements = _split_document_elements(pred_elements)
    true_text_elements, true_table_elements = _split_document_elements(true_elements)

    text_metrics = _compute_text_score(
        pred_text_elements,
        true_text_elements,
        iou_threshold=iou_threshold,
    )
    table_metrics = _compute_table_teds_score(
        pred_table_elements,
        true_table_elements,
        iou_threshold=iou_threshold,
    )

    total_text = len(pred_text_elements) + len(true_text_elements)
    total_table = len(pred_table_elements) + len(true_table_elements)
    score_weight = total_text + total_table
    if score_weight == 0:
        overall_score = 1.0
    else:
        overall_score = (
            text_metrics.get("score", 0.0) * total_text
            + table_metrics.get("score", 0.0) * total_table
        ) / score_weight

    filtered_pred = pred_text_elements + pred_table_elements
    filtered_true = true_text_elements + true_table_elements

    # Layout detection mAP
    layout_metrics = layout_detection_map(filtered_pred, filtered_true, iou_threshold=iou_threshold)

    # Reading order accuracy
    reading_order_metrics = reading_order_accuracy(filtered_pred, filtered_true)

    # Key-value extraction F1
    kv_metrics = key_value_extraction_f1(filtered_pred, filtered_true)

    # Combined metrics
    return {
        "layout_detection": layout_metrics,
        "reading_order": reading_order_metrics,
        "key_value_extraction": kv_metrics,
        "text": text_metrics,
        "table": table_metrics,
        "text_score": text_metrics.get("score", 0.0),
        "table_teds": table_metrics.get("score", 0.0),
        "overall_score": overall_score,
        "overall_f1": overall_score,
    }


# ==================== Batch Evaluation ====================

def evaluate_dataset(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    task_type: str = "table",
) -> Dict[str, Any]:
    """
    Evaluate a dataset of predictions against ground truths.

    Args:
        predictions: List of prediction dicts with html and json
        ground_truths: List of ground truth dicts with html and json
        task_type: Either "table" or "document"

    Returns:
        Aggregated metrics across the entire dataset
    """
    if len(predictions) != len(ground_truths):
        return {"error": f"Mismatched lengths: {len(predictions)} vs {len(ground_truths)}"}

    all_metrics = []

    for pred, gt in zip(predictions, ground_truths):
        if task_type == "table":
            metrics = evaluate_table(
                pred.get("html", ""),
                pred.get("json", {}),
                gt.get("html", ""),
                gt.get("json", {}),
            )
        else:
            metrics = evaluate_document(
                pred.get("elements", []),
                gt.get("elements", []),
            )
        all_metrics.append(metrics)

    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float)):
            values = [m.get(key, 0) for m in all_metrics]
            aggregated[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

    # Add count
    aggregated["sample_count"] = len(all_metrics)

    return aggregated


if __name__ == "__main__":
    # Example usage
    print("Evaluation metrics module loaded successfully.")

    # Example table evaluation
    true_table = {
        "html": """
        <table>
            <tr><th>A</th><th>B</th></tr>
            <tr><td>1</td><td>2</td></tr>
        </table>
        """,
        "json": {
            "num_rows": 2,
            "num_cols": 2,
            "cells": [
                {"row": 0, "col": 0, "text": "A", "is_header": True},
                {"row": 0, "col": 1, "text": "B", "is_header": True},
                {"row": 1, "col": 0, "text": "1"},
                {"row": 1, "col": 1, "text": "2"},
            ],
        },
    }

    pred_table = {
        "html": """
        <table>
            <tr><th>A</th><th>B</th></tr>
            <tr><td>1</td><td>3</td></tr>
        </table>
        """,
        "json": {
            "num_rows": 2,
            "num_cols": 2,
            "cells": [
                {"row": 0, "col": 0, "text": "A", "is_header": True},
                {"row": 0, "col": 1, "text": "B", "is_header": True},
                {"row": 1, "col": 0, "text": "1"},
                {"row": 1, "col": 1, "text": "3"},
            ],
        },
    }

    result = evaluate_table(
        pred_table["html"],
        pred_table["json"],
        true_table["html"],
        true_table["json"],
    )
    print(f"Table evaluation result: {result}")
