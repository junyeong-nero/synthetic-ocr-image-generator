from typing import Any, Dict


def cell_level_text_accuracy(pred_json: Dict, true_json: Dict) -> Dict[str, Any]:
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
    true_num_rows = true_json.get("num_rows", 0)
    pred_num_rows = pred_json.get("num_rows", 0)
    true_num_cols = true_json.get("num_cols", 0)
    pred_num_cols = pred_json.get("num_cols", 0)

    row_tp = min(true_num_rows, pred_num_rows)
    row_fp = max(0, pred_num_rows - true_num_rows)
    row_fn = max(0, true_num_rows - pred_num_rows)
    row_precision = row_tp / (row_tp + row_fp) if (row_tp + row_fp) > 0 else 0.0
    row_recall = row_tp / (row_tp + row_fn) if (row_tp + row_fn) > 0 else 0.0
    row_f1 = 2 * row_precision * row_recall / (row_precision + row_recall) if (row_precision + row_recall) > 0 else 0.0

    col_tp = min(true_num_cols, pred_num_cols)
    col_fp = max(0, pred_num_cols - true_num_cols)
    col_fn = max(0, true_num_cols - pred_num_cols)
    col_precision = col_tp / (col_tp + col_fp) if (col_tp + col_fp) > 0 else 0.0
    col_recall = col_tp / (col_tp + col_fn) if (col_tp + col_fn) > 0 else 0.0
    col_f1 = 2 * col_precision * col_recall / (col_precision + col_recall) if (col_precision + col_recall) > 0 else 0.0

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
    from .table_edit_distance import TEDS

    teds_calculator = TEDS(structure_only=True)
    teds_result = teds_calculator.evaluate(pred_html, true_html)
    cell_accuracy = cell_level_text_accuracy(pred_json, true_json)
    row_col_metrics = row_column_detection_metrics(pred_json, true_json)
    return {
        "teds": teds_result.get("teds", 0.0),
        "teds_error": teds_result.get("error", None),
        **cell_accuracy,
        **row_col_metrics,
    }
