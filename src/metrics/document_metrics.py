import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
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


def _compute_text_score(pred_text_elements: List[Dict[str, Any]], true_text_elements: List[Dict[str, Any]], iou_threshold: float) -> Dict[str, Any]:
    from .edit_distance import cer
    pairs, unmatched_pred, unmatched_true = _pair_elements_by_layout(pred_text_elements, true_text_elements, iou_threshold=iou_threshold)
    if not pred_text_elements and not true_text_elements:
        return {"score": 1.0, "matched_pairs": 0, "unmatched_pred": 0, "unmatched_true": 0, "pair_scores": []}
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
    return {"score": score, "matched_pairs": len(pairs), "unmatched_pred": unmatched_pred, "unmatched_true": unmatched_true, "pair_scores": pair_scores}


def _compute_table_teds_score(pred_table_elements: List[Dict[str, Any]], true_table_elements: List[Dict[str, Any]], iou_threshold: float) -> Dict[str, Any]:
    from .table_edit_distance import TEDS
    pairs, unmatched_pred, unmatched_true = _pair_elements_by_layout(pred_table_elements, true_table_elements, iou_threshold=iou_threshold)
    if not pred_table_elements and not true_table_elements:
        return {"score": 1.0, "matched_pairs": 0, "unmatched_pred": 0, "unmatched_true": 0, "pair_scores": []}
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
    return {"score": score, "matched_pairs": len(pairs), "unmatched_pred": unmatched_pred, "unmatched_true": unmatched_true, "pair_scores": pair_scores}


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(numeric) else numeric


def _rankdata(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        avg_rank = (index + end - 1) / 2 + 1
        for offset in range(index, end):
            original_index = indexed[offset][0]
            ranks[original_index] = avg_rank
        index = end
    return ranks


def _pearson_correlation(left: List[float], right: List[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((l - left_mean) * (r - right_mean) for l, r in zip(left, right))
    left_variance = sum((l - left_mean) ** 2 for l in left)
    right_variance = sum((r - right_mean) ** 2 for r in right)
    denominator = (left_variance * right_variance) ** 0.5
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _fallback_kendall_tau(left: List[float], right: List[float]) -> float:
    concordant = 0
    discordant = 0
    n = len(left)
    for i in range(n):
        for j in range(i + 1, n):
            left_diff = left[j] - left[i]
            right_diff = right[j] - right[i]
            product = left_diff * right_diff
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def layout_detection_map(pred_elements: List[Dict], true_elements: List[Dict], iou_threshold: float = 0.5) -> Dict[str, Any]:
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
            continue
        true_matched = set()
        pred_scores = []
        for pred_elem in pred_elems:
            best_iou = 0
            best_true_idx = -1
            for true_idx, true_elem in enumerate(true_elems):
                if true_idx in true_matched:
                    continue
                iou = calculate_iou(pred_elem.get("bounding_box"), true_elem.get("bounding_box"))
                if iou > best_iou:
                    best_iou = iou
                    best_true_idx = true_idx
            matched = best_iou >= iou_threshold
            if matched:
                true_matched.add(best_true_idx)
            pred_scores.append((1.0, matched))
        if pred_scores:
            tp = sum(1 for _, matched in pred_scores if matched)
            fp = len(pred_scores) - tp
            fn = len(true_elems) - len(true_matched)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_type_ap[elem_type] = {"precision": precision, "recall": recall, "f1": f1, "true_count": len(true_elems), "pred_count": len(pred_elems)}
    if per_type_ap:
        ap_values = [v.get("f1", 0.0) for v in per_type_ap.values()]
        mAP = np.mean(ap_values)
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
    return {"map": mAP, "per_type_metrics": per_type_ap, "overall_precision": overall_precision, "overall_recall": overall_recall, "overall_f1": overall_f1}


def reading_order_accuracy(pred_elements: List[Dict], true_elements: List[Dict]) -> Dict[str, Any]:
    if not pred_elements or not true_elements:
        return {"kendall_tau": 0.0, "spearman_rho": 0.0, "order_accuracy": 0.0}
    pred_orders = [e.get("reading_order", i) for i, e in enumerate(pred_elements)]
    true_orders = [e.get("reading_order", i) for i, e in enumerate(true_elements)]
    min_len = min(len(pred_orders), len(true_orders))
    pred_orders = pred_orders[:min_len]
    true_orders = true_orders[:min_len]
    if min_len < 2:
        return {"kendall_tau": 1.0, "spearman_rho": 1.0, "order_accuracy": 1.0}
    try:
        from scipy import stats

        kendall_tau, _ = stats.kendalltau(pred_orders, true_orders)
        spearman_rho, _ = stats.spearmanr(pred_orders, true_orders)
    except Exception:
        kendall_tau = _fallback_kendall_tau(pred_orders, true_orders)
        spearman_rho = _pearson_correlation(_rankdata(pred_orders), _rankdata(true_orders))
    correct_pairs = 0
    total_pairs = min_len - 1
    for i in range(total_pairs):
        pred_diff = pred_orders[i + 1] - pred_orders[i]
        true_diff = true_orders[i + 1] - true_orders[i]
        if (pred_diff > 0 and true_diff > 0) or (pred_diff < 0 and true_diff < 0):
            correct_pairs += 1
    order_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 1.0
    return {"kendall_tau": _safe_float(kendall_tau), "spearman_rho": _safe_float(spearman_rho), "order_accuracy": order_accuracy}


def key_value_extraction_f1(pred_elements: List[Dict], true_elements: List[Dict]) -> Dict[str, Any]:
    from .edit_distance import cer

    def extract_kv_pairs(elements: List[Dict]) -> List[Tuple[str, str, Tuple]]:
        pairs = []
        for elem in elements:
            metadata = elem.get("metadata", {})
            if isinstance(metadata, dict):
                key = metadata.get("key", elem.get("text", ""))
                value = metadata.get("value", "")
                bbox = elem.get("bounding_box")
                pairs.append((key, value, bbox))
            else:
                try:
                    kv = json.loads(metadata) if isinstance(metadata, str) else metadata
                    key = kv.get("key", elem.get("text", ""))
                    value = kv.get("value", "")
                    bbox = elem.get("bounding_box")
                    pairs.append((key, value, bbox))
                except (ValueError, json.JSONDecodeError):
                    pairs.append((elem.get("text", ""), "", elem.get("bounding_box")))
        return pairs

    true_pairs = extract_kv_pairs(true_elements)
    pred_pairs = extract_kv_pairs(pred_elements)
    if not true_pairs:
        return {"precision": 1.0 if not pred_pairs else 0.0, "recall": 1.0, "f1": 1.0 if not pred_pairs else 0.0}
    matched = set()
    true_used = set()
    pred_used = set()
    for pred_idx, (p_key, p_value, _p_bbox) in enumerate(pred_pairs):
        best_match_idx = -1
        best_score = 0
        for true_idx, (t_key, t_value, _t_bbox) in enumerate(true_pairs):
            if true_idx in true_used:
                continue
            key_cer = cer(p_key, t_key) if p_key and t_key else 1.0
            value_cer = cer(p_value, t_value) if p_value and t_value else 1.0
            match_score = 1.0 - ((key_cer + value_cer) / 2)
            if match_score > best_score:
                best_score = match_score
                best_match_idx = true_idx
        if best_match_idx >= 0 and best_score >= 0.7:
            matched.add((pred_idx, best_match_idx))
            true_used.add(best_match_idx)
            pred_used.add(pred_idx)
    tp = len(matched)
    fp = len(pred_pairs) - len(pred_used)
    fn = len(true_pairs) - len(true_used)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matched_pairs": tp, "missing_pairs": fn, "extra_pairs": fp, "true_pair_count": len(true_pairs), "pred_pair_count": len(pred_pairs)}


def evaluate_document(pred_elements: List[Dict], true_elements: List[Dict], iou_threshold: float = 0.5) -> Dict[str, Any]:
    pred_text_elements, pred_table_elements = _split_document_elements(pred_elements)
    true_text_elements, true_table_elements = _split_document_elements(true_elements)
    text_metrics = _compute_text_score(pred_text_elements, true_text_elements, iou_threshold=iou_threshold)
    table_metrics = _compute_table_teds_score(pred_table_elements, true_table_elements, iou_threshold=iou_threshold)
    total_text = len(pred_text_elements) + len(true_text_elements)
    total_table = len(pred_table_elements) + len(true_table_elements)
    score_weight = total_text + total_table
    if score_weight == 0:
        overall_score = 1.0
    else:
        overall_score = ((text_metrics.get("score", 0.0) * total_text) + (table_metrics.get("score", 0.0) * total_table)) / score_weight
    filtered_pred = pred_text_elements + pred_table_elements
    filtered_true = true_text_elements + true_table_elements
    layout_metrics = layout_detection_map(filtered_pred, filtered_true, iou_threshold=iou_threshold)
    reading_order_metrics = reading_order_accuracy(filtered_pred, filtered_true)
    kv_metrics = key_value_extraction_f1(filtered_pred, filtered_true)
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
