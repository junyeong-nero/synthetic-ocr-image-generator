import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from evaluation.config import EvaluationConfig, FormatType, InferenceBackend, ModelConfig
from evaluation.pipeline import EvaluationPipeline
from evaluation.runner import EvaluationRunner
from evaluation.strategies import DocumentEvaluator, TableEvaluator
from evaluation.types import InferenceResult


class FixedOutputModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def run(self, prompts, images):
        return list(self.outputs)


def make_config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        dataset_id="dummy-dataset",
        subset="sentence",
        split="train",
        format_type=FormatType.SENTENCE,
        model=ModelConfig(model_id="dummy-model", backend=InferenceBackend.OPENAI),
        batch_size=2,
        output_dir=str(tmp_path),
        resume_from_checkpoint=True,
    )


def test_runner_ignores_checkpoint_from_different_context(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    stale = {
        "completed": [0],
        "results": [
            {
                "index": 0,
                "prediction": "stale",
                "ground_truth": "stale",
                "latency_ms": 1.0,
                "error": None,
            }
        ],
        "context": {
            "dataset_id": "other-dataset",
            "subset": "sentence",
            "split": "train",
            "format_type": "sentence",
            "model_id": "dummy-model",
            "backend": "openai",
        },
    }
    checkpoint_path.write_text(json.dumps(stale), encoding="utf-8")

    runner = EvaluationRunner(make_config(tmp_path), FixedOutputModel(["ok", "ok"]))

    assert runner.state.completed == []
    assert runner.state.results == []
    assert runner.state.context["dataset_id"] == "dummy-dataset"


def test_runner_moves_corrupt_checkpoint_as_backup(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text("{broken-json", encoding="utf-8")

    runner = EvaluationRunner(make_config(tmp_path), FixedOutputModel(["ok", "ok"]))

    assert checkpoint_path.exists() is False
    assert (tmp_path / "checkpoint.corrupt.json").exists()
    assert runner.state.completed == []


def test_runner_marks_all_batch_items_failed_on_prediction_length_mismatch(
    tmp_path: Path,
) -> None:
    runner = EvaluationRunner(make_config(tmp_path), FixedOutputModel(["only-one"]))
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    ground_truths = ["a", "b"]
    prompts = ["p", "p"]

    results = runner.run(images, ground_truths, prompts)

    assert len(results) == 2
    assert runner.state.completed == []
    assert all(r.error is not None for r in results)
    assert all("mismatched batch size" in (r.error or "") for r in results)


@pytest.fixture
def stub_table_document_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("metrics.table_document_metrics")

    def evaluate_table(pred_html, pred_json, true_html, true_json):
        return {"teds": 1.0, "cell_accuracy": 1.0, "overall_structure_f1": 1.0}

    def evaluate_document(pred_elements, true_elements):
        return {
            "layout_detection": {"overall_f1": 1.0},
            "reading_order": {"order_accuracy": 1.0},
            "key_value_extraction": {"f1": 1.0},
            "text_score": 1.0,
            "table_teds": 1.0,
            "overall_score": 1.0,
            "overall_f1": 1.0,
        }

    module.evaluate_table = evaluate_table
    module.evaluate_document = evaluate_document
    monkeypatch.setitem(sys.modules, "metrics.table_document_metrics", module)


def test_table_evaluator_handles_malformed_ground_truth_json(
    stub_table_document_metrics: None,
) -> None:
    evaluator = TableEvaluator()
    metrics = evaluator.compute_metrics(
        predictions=["<table><tr><td>a</td></tr></table>"],
        ground_truths=[{"html": "<table><tr><td>a</td></tr></table>", "json": "{bad json"}],
    )

    assert metrics["avg_teds"] == 1.0
    assert metrics["avg_cell_accuracy"] == 1.0


def test_document_evaluator_handles_malformed_nested_ground_truth_json(
    stub_table_document_metrics: None,
) -> None:
    evaluator = DocumentEvaluator()
    metrics = evaluator.compute_metrics(
        predictions=['{"elements": []}'],
        ground_truths=[{"ground_truth": "{broken json"}],
    )

    assert metrics["avg_overall_f1"] == 1.0
    assert metrics["avg_layout_f1"] == 1.0
    assert metrics["avg_text_table_score"] == 1.0
    assert metrics["avg_formula_edit_distance"] == 0.0
    assert metrics["avg_text_table_formula_score"] == 1.0


def test_pipeline_compute_metrics_skips_none_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evaluation.pipeline.create_model", lambda _config: FixedOutputModel([]))
    config = make_config(tmp_path)
    pipeline = EvaluationPipeline(config)

    results = [
        InferenceResult(index=0, prediction=None, ground_truth="abc", latency_ms=1.0),
        InferenceResult(index=1, prediction="abc", ground_truth="abc", latency_ms=1.0),
    ]

    metrics = pipeline._compute_metrics(results)

    assert metrics["avg_cer"] == 0.0
    assert metrics["avg_wer"] == 0.0
