from typing import Any, Dict, List

import numpy as np

from .document_metrics import evaluate_document
from .table_metrics import evaluate_table


def evaluate_dataset(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    task_type: str = "table",
) -> Dict[str, Any]:
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

    aggregated["sample_count"] = len(all_metrics)
    return aggregated


if __name__ == "__main__":
    print("Evaluation metrics module loaded successfully.")
