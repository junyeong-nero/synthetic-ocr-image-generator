"""Furniture filtering is sample-specific and preserves body lines."""

from types import SimpleNamespace

from src.evaluation.page_furniture import strip_page_furniture
from src.evaluation.pipeline import EvaluationPipeline
from src.evaluation.types import InferenceResult


def test_normalized_and_fuzzy_whole_lines_only():
    furniture = {"header": "Annual financial report\n연구 보고서", "footer": "Confidential", "page_number": "- 3 -"}
    prediction = "# Annual financial rep0rt\n연구  보고서\nBody Confidential details\nConfidential\n-3-\n30\n| 3 |"
    assert strip_page_furniture(prediction, furniture) == "Body Confidential details\n30\n| 3 |"
    assert strip_page_furniture(prediction, None) == prediction


def test_short_numbers_require_exact_match():
    assert strip_page_furniture("3\n4\n13\n3. Item", {"page_number": "3"}) == "4\n13\n3. Item"


def test_pipeline_filters_by_result_index_without_changing_raw_prediction():
    pipeline = object.__new__(EvaluationPipeline)
    pipeline.config = SimpleNamespace(target_column="GT_markdown")
    pipeline._extract_ground_truths([
        {"GT_markdown": "Body", "page_furniture": {"header": "Report title"}},
        {"GT_markdown": "Report title\n\nBody"},
    ])
    results = [
        InferenceResult(index=1, prediction="Report title\n\nBody", ground_truth="Report title\n\nBody", latency_ms=0),
        InferenceResult(index=0, prediction="Report title\n\nBody", ground_truth="Body", latency_ms=0),
    ]
    assert pipeline._compute_metrics(results)["avg_markdown_overall_score"] == 1.0
    assert results[1].prediction == "Report title\n\nBody"
