import sys
from pathlib import Path
from types import ModuleType

from evaluation.config import EvaluationConfig, InferenceBackend, ModelConfig
from evaluation.pipeline import EvaluationPipeline
from evaluation.utils import extract_html_table, parse_model_output_as_json
from evaluation.types import InferenceResult
from metrics.markdown_block_metrics import normalize_markdown_text
from metrics.table_document_metrics import evaluate_document


def test_parse_model_output_as_json_recovers_fenced_candidate_with_trailing_comma() -> None:
    payload = """
    Here is the result:
    ```json
    {
      "entities": {"total": "100"},
    }
    ```
    """
    parsed = parse_model_output_as_json(payload)
    assert parsed == {"entities": {"total": "100"}}


def test_extract_html_table_converts_markdown_table() -> None:
    output = "| Item | Qty |\n| --- | --- |\n| Pen | 2 |\n"
    extracted = extract_html_table(output)
    assert "<table>" in extracted
    assert "<th>Item</th>" in extracted
    assert "<td>Pen</td>" in extracted


def test_normalize_markdown_text_uses_omnidoc_style_cleaning() -> None:
    text = "Hello,\tWorld!\n안녕? 123"
    assert normalize_markdown_text(text) == "HelloWorld안녕123"


def test_pipeline_exposes_metric_views(tmp_path: Path, monkeypatch) -> None:
    class DummyModel:
        def run(self, prompts, images):
            return []

    monkeypatch.setattr("evaluation.pipeline.create_model", lambda _config: DummyModel())
    config = EvaluationConfig(
        dataset_id="dummy",
        split="train",
        model=ModelConfig(model_id="dummy-model", backend=InferenceBackend.OPENAI),
        output_dir=str(tmp_path),
    )
    pipeline = EvaluationPipeline(config)

    results = [
        InferenceResult(index=0, prediction="abc", ground_truth="abc", latency_ms=1.0),
    ]
    metrics = pipeline._compute_metrics(results)

    assert metrics["avg_markdown_text_score"] == 1.0
    assert pipeline.metric_views["normalized"]["avg_markdown_text_score"] == 1.0
    assert "raw" not in pipeline.metric_views


def test_evaluate_document_uses_text_table_scores_and_ignores_formula_elements(monkeypatch) -> None:
    module = ModuleType("metrics.table_edit_distance")

    class StubTEDS:
        def __init__(self, structure_only=True):
            self.structure_only = structure_only

        def evaluate(self, pred_html, true_html):
            if pred_html == true_html:
                return {"teds": 1.0}
            return {"teds": 0.0}

    module.TEDS = StubTEDS
    monkeypatch.setitem(sys.modules, "metrics.table_edit_distance", module)

    pred_elements = [
        {
            "type": "text",
            "text": "Invoice No 123",
            "bounding_box": [0, 0, 100, 20],
            "reading_order": 0,
        },
        {
            "type": "table",
            "html": "<table><tr><th>Item</th></tr><tr><td>Pen</td></tr></table>",
            "bounding_box": [0, 30, 120, 100],
            "reading_order": 1,
        },
        {
            "type": "formula",
            "text": "E = mc^2",
            "bounding_box": [0, 110, 80, 130],
            "reading_order": 2,
        },
    ]
    true_elements = [
        {
            "type": "text",
            "text": "Invoice No 124",
            "bounding_box": [0, 0, 100, 20],
            "reading_order": 0,
        },
        {
            "type": "table",
            "html": "<table><tr><th>Item</th></tr><tr><td>Pen</td></tr></table>",
            "bounding_box": [0, 30, 120, 100],
            "reading_order": 1,
        },
        {
            "type": "equation_inline",
            "text": "a^2+b^2=c^2",
            "bounding_box": [0, 110, 80, 130],
            "reading_order": 2,
        },
    ]

    metrics = evaluate_document(pred_elements, true_elements)

    assert 0.0 <= metrics["text_score"] < 1.0
    assert metrics["table_teds"] == 1.0
    assert metrics["overall_score"] == metrics["overall_f1"]
