from typing import Any, Dict, List

import numpy as np
from datasets import Dataset


class MarkdownEvaluator:
    def extract_ground_truths(self, dataset: Dataset, target_column: str) -> List[Any]:
        return [
            dataset[i].get("GT_markdown", dataset[i].get("markdown", ""))
            for i in range(len(dataset))
        ]

    def compute_metrics(
        self, predictions: List[str], ground_truths: List[str], normalize: bool = True
    ) -> Dict[str, float]:
        from metrics.markdown_block_metrics import evaluate_markdown_blocks

        per_sample = [
            evaluate_markdown_blocks(pred, gt)
            for pred, gt in zip(predictions, ground_truths)
        ]

        def aggregate(key: str) -> tuple[float, float]:
            values = [float(item.get(key, 0.0)) for item in per_sample]
            if not values:
                return 0.0, 0.0
            return float(np.mean(values)), float(np.std(values))

        avg_text, std_text = aggregate("markdown_text_score")
        avg_table, std_table = aggregate("markdown_table_teds")
        avg_formula, std_formula = aggregate("markdown_formula_score")
        avg_order, std_order = aggregate("markdown_order_score")
        avg_overall, std_overall = aggregate("markdown_overall_score")

        return {
            "avg_markdown_text_score": avg_text,
            "std_markdown_text_score": std_text,
            "avg_markdown_table_teds": avg_table,
            "std_markdown_table_teds": std_table,
            "avg_markdown_formula_score": avg_formula,
            "std_markdown_formula_score": std_formula,
            "avg_markdown_order_score": avg_order,
            "std_markdown_order_score": std_order,
            "avg_markdown_overall_score": avg_overall,
            "std_markdown_overall_score": std_overall,
        }

    def compute_metric_views(
        self, predictions: List[str], ground_truths: List[str]
    ) -> Dict[str, Dict[str, float]]:
        return {
            "raw": self.compute_metrics(predictions, ground_truths, normalize=False),
            "normalized": self.compute_metrics(predictions, ground_truths, normalize=True),
        }
