import sys
from pathlib import Path
from types import ModuleType

from evaluation.config import EvaluationConfig, FormatType, InferenceBackend, ModelConfig
from evaluation.pipeline import EvaluationPipeline
from evaluation.strategies import TableEvaluator
from evaluation.utils import extract_html_table, parse_model_output_as_json
from evaluation.types import InferenceResult


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


def test_table_evaluator_normalizes_nested_table_payload(monkeypatch) -> None:
    module = ModuleType("metrics.table_document_metrics")
    captured = {}

    def evaluate_table(pred_html, pred_json, true_html, true_json):
        captured["pred_html"] = pred_html
        captured["pred_json"] = pred_json
        captured["true_html"] = true_html
        captured["true_json"] = true_json
        return {"teds": 1.0, "cell_accuracy": 1.0, "overall_structure_f1": 1.0}

    module.evaluate_table = evaluate_table
    monkeypatch.setitem(sys.modules, "metrics.table_document_metrics", module)

    pred = '{"table": {"html": "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"}}'
    gt = {"html": "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>", "json": {}}
    metrics = TableEvaluator().compute_metrics([pred], [gt], normalize=True)

    assert metrics["avg_teds"] == 1.0
    assert captured["pred_json"]["num_rows"] == 2
    assert captured["pred_json"]["num_cols"] == 1


def test_pipeline_exposes_metric_views(tmp_path: Path, monkeypatch) -> None:
    class DummyModel:
        def run(self, prompts, images):
            return []

    monkeypatch.setattr("evaluation.pipeline.create_model", lambda _config: DummyModel())
    config = EvaluationConfig(
        dataset_id="dummy",
        subset="sentence",
        split="train",
        format_type=FormatType.SENTENCE,
        model=ModelConfig(model_id="dummy-model", backend=InferenceBackend.OPENAI),
        output_dir=str(tmp_path),
    )
    pipeline = EvaluationPipeline(config)

    results = [
        InferenceResult(index=0, prediction="abc", ground_truth="abc", latency_ms=1.0),
    ]
    metrics = pipeline._compute_metrics(results)

    assert metrics["avg_cer"] == 0.0
    assert pipeline.metric_views["raw"]["avg_cer"] == 0.0
    assert pipeline.metric_views["normalized"]["avg_cer"] == 0.0
