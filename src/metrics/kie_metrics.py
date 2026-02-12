"""KIE (Key Information Extraction) Evaluation Metrics.

This module provides metrics for evaluating KIE results:
- Entity-level F1: Precision/Recall/F1 for each entity type
- Field Accuracy: Per-field extraction accuracy
- Normalized Edit Distance: Character-level similarity
- End-to-end F1: Overall extraction performance
"""

from typing import Any, Dict, List, Optional
import numpy as np
from collections import defaultdict


def normalized_edit_distance(pred: str, target: str) -> float:
    """
    Calculate Normalized Edit Distance (NED) between two strings.

    Returns value in [0, 1] where 0 means identical.
    """
    if not target and not pred:
        return 0.0
    if not target:
        return 1.0
    if not pred:
        return 1.0

    from .edit_distance import cer
    return cer(target, pred)


def entity_level_metrics(
    pred_entities: Dict[str, str],
    true_entities: Dict[str, str],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Calculate entity-level metrics for KIE evaluation.

    Args:
        pred_entities: Predicted entities {field_name: value}
        true_entities: Ground truth entities {field_name: value}
        threshold: NED threshold for considering a match (lower = stricter)

    Returns:
        Dict containing precision, recall, f1, and per-field metrics.
    """
    if not true_entities:
        return {
            "precision": 1.0 if not pred_entities else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not pred_entities else 0.0,
            "avg_ned": 0.0,
            "avg_accuracy": 1.0 if not pred_entities else 0.0,
            "matched_fields": 0,
            "total_true_fields": 0,
            "total_pred_fields": len(pred_entities),
            "per_field": {},
        }

    per_field_metrics = {}
    matched_fields = 0
    total_ned = 0.0

    # All fields from both prediction and ground truth
    all_fields = set(true_entities.keys()) | set(pred_entities.keys())

    for field in all_fields:
        true_value = true_entities.get(field, "")
        pred_value = pred_entities.get(field, "")

        # Calculate NED for this field
        ned = normalized_edit_distance(str(pred_value), str(true_value))

        # Determine if this is a match based on threshold
        is_match = ned <= threshold

        per_field_metrics[field] = {
            "true_value": true_value,
            "pred_value": pred_value,
            "ned": ned,
            "accuracy": 1.0 - ned,
            "is_match": is_match,
            "in_ground_truth": field in true_entities,
            "in_prediction": field in pred_entities,
        }

        if field in true_entities:
            total_ned += ned
            if is_match:
                matched_fields += 1

    # Calculate precision, recall, F1
    true_field_count = len(true_entities)
    pred_field_count = len(pred_entities)

    # Precision: Of predicted fields, how many match ground truth
    precision_matches = sum(
        1 for f in pred_entities
        if f in true_entities and per_field_metrics[f]["is_match"]
    )
    precision = precision_matches / pred_field_count if pred_field_count > 0 else 0.0

    # Recall: Of ground truth fields, how many were correctly predicted
    recall = matched_fields / true_field_count if true_field_count > 0 else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Average NED across true fields
    avg_ned = total_ned / true_field_count if true_field_count > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_ned": avg_ned,
        "avg_accuracy": 1.0 - avg_ned,
        "matched_fields": matched_fields,
        "total_true_fields": true_field_count,
        "total_pred_fields": pred_field_count,
        "per_field": per_field_metrics,
    }


def line_item_metrics(
    pred_items: List[Dict[str, Any]],
    true_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate metrics for line item extraction (receipts/invoices).

    Args:
        pred_items: Predicted line items
        true_items: Ground truth line items

    Returns:
        Dict containing item-level metrics.
    """
    if not true_items:
        return {
            "precision": 1.0 if not pred_items else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not pred_items else 0.0,
            "item_accuracy": 1.0 if not pred_items else 0.0,
        }

    matched_items = 0
    total_item_ned = 0.0

    # Match items by name similarity
    true_matched = set()

    for pred_item in pred_items:
        pred_name = str(pred_item.get("name", ""))
        best_match_idx = -1
        best_ned = 1.0

        for true_idx, true_item in enumerate(true_items):
            if true_idx in true_matched:
                continue

            true_name = str(true_item.get("name", ""))
            ned = normalized_edit_distance(pred_name, true_name)

            if ned < best_ned:
                best_ned = ned
                best_match_idx = true_idx

        if best_match_idx >= 0 and best_ned <= 0.3:
            true_matched.add(best_match_idx)
            matched_items += 1

            # Check quantity and price
            true_item = true_items[best_match_idx]

            qty_match = pred_item.get("quantity") == true_item.get("quantity")
            price_match = pred_item.get("total_price") == true_item.get("total_price")

            # NED for full item match (name + quantity + price)
            if qty_match and price_match:
                total_item_ned += best_ned
            else:
                total_item_ned += 0.5 + best_ned * 0.5

    # Calculate metrics
    true_count = len(true_items)
    pred_count = len(pred_items)

    precision = matched_items / pred_count if pred_count > 0 else 0.0
    recall = matched_items / true_count if true_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_item_accuracy = (
        1.0 - (total_item_ned / true_count) if true_count > 0 and matched_items > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_items": matched_items,
        "total_true_items": true_count,
        "total_pred_items": pred_count,
        "item_accuracy": avg_item_accuracy,
    }


def category_metrics(
    pred_entities: Dict[str, str],
    true_entities: Dict[str, Any],
    true_categories: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics grouped by entity category.

    Categories typically include: header, total, payment, customer, etc.

    Args:
        pred_entities: Predicted entities {field_name: value}
        true_entities: Ground truth entities with category info
        true_categories: Optional mapping of field -> category

    Returns:
        Dict of category -> metrics
    """
    # Group fields by category
    category_fields = defaultdict(list)

    if true_categories:
        for field, category in true_categories.items():
            category_fields[category].append(field)
    else:
        # Try to infer categories from common field patterns
        category_mapping = {
            "header": ["company", "address", "date", "time", "receipt_number", "invoice_number"],
            "customer": ["customer_name", "customer_address", "name", "phone", "email"],
            "total": ["subtotal", "tax", "total", "discount"],
            "payment": ["payment_method", "card_number"],
            "personal": ["name", "phone", "email", "address"],
            "organization": ["company", "department", "position"],
        }

        for category, fields in category_mapping.items():
            for field in fields:
                if field in true_entities:
                    category_fields[category].append(field)

    # Calculate metrics per category
    category_results = {}

    for category, fields in category_fields.items():
        cat_true = {f: true_entities.get(f, "") for f in fields if f in true_entities}
        cat_pred = {f: pred_entities.get(f, "") for f in fields}

        if cat_true:
            metrics = entity_level_metrics(cat_pred, cat_true)
            category_results[category] = {
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["avg_accuracy"],
                "field_count": len(cat_true),
            }

    return category_results


def evaluate_kie(
    pred_entities: Dict[str, str],
    true_entities: Dict[str, str],
    pred_items: Optional[List[Dict[str, Any]]] = None,
    true_items: Optional[List[Dict[str, Any]]] = None,
    true_categories: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive KIE evaluation.

    Args:
        pred_entities: Predicted entities {field_name: value}
        true_entities: Ground truth entities {field_name: value}
        pred_items: Predicted line items (for receipts/invoices)
        true_items: Ground truth line items
        true_categories: Optional field -> category mapping

    Returns:
        Complete evaluation results.
    """
    # Entity-level metrics
    entity_metrics = entity_level_metrics(pred_entities, true_entities)

    # Line item metrics (if applicable)
    item_metrics = None
    if true_items is not None:
        pred_items = pred_items or []
        item_metrics = line_item_metrics(pred_items, true_items)

    # Category metrics
    cat_metrics = category_metrics(pred_entities, true_entities, true_categories)

    # Overall score
    scores = [entity_metrics["f1"]]
    if item_metrics:
        scores.append(item_metrics["f1"])

    overall_f1 = np.mean(scores)

    return {
        "entity_metrics": entity_metrics,
        "line_item_metrics": item_metrics,
        "category_metrics": cat_metrics,
        "overall_f1": overall_f1,
        "overall_accuracy": entity_metrics["avg_accuracy"],
    }


def aggregate_kie_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate KIE metrics across multiple samples.

    Args:
        results: List of evaluate_kie results

    Returns:
        Aggregated metrics.
    """
    if not results:
        return {}

    # Collect all scalar metrics
    entity_f1s = []
    entity_precisions = []
    entity_recalls = []
    entity_accuracies = []
    item_f1s = []
    overall_f1s = []

    per_field_scores = defaultdict(list)

    for r in results:
        entity = r.get("entity_metrics", {})
        entity_f1s.append(entity.get("f1", 0.0))
        entity_precisions.append(entity.get("precision", 0.0))
        entity_recalls.append(entity.get("recall", 0.0))
        entity_accuracies.append(entity.get("avg_accuracy", 0.0))

        # Per-field scores
        for field, field_data in entity.get("per_field", {}).items():
            if field_data.get("in_ground_truth"):
                per_field_scores[field].append(field_data.get("accuracy", 0.0))

        items = r.get("line_item_metrics")
        if items:
            item_f1s.append(items.get("f1", 0.0))

        overall_f1s.append(r.get("overall_f1", 0.0))

    # Build aggregated result
    aggregated = {
        "avg_entity_f1": float(np.mean(entity_f1s)),
        "std_entity_f1": float(np.std(entity_f1s)),
        "avg_entity_precision": float(np.mean(entity_precisions)),
        "avg_entity_recall": float(np.mean(entity_recalls)),
        "avg_entity_accuracy": float(np.mean(entity_accuracies)),
        "std_entity_accuracy": float(np.std(entity_accuracies)),
        "avg_overall_f1": float(np.mean(overall_f1s)),
        "std_overall_f1": float(np.std(overall_f1s)),
    }

    if item_f1s:
        aggregated["avg_item_f1"] = float(np.mean(item_f1s))
        aggregated["std_item_f1"] = float(np.std(item_f1s))

    # Per-field aggregated scores
    per_field_aggregated = {}
    for field, scores in per_field_scores.items():
        per_field_aggregated[field] = {
            "avg_accuracy": float(np.mean(scores)),
            "sample_count": len(scores),
        }

    aggregated["per_field_metrics"] = per_field_aggregated

    return aggregated
