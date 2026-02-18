import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from evaluation.config import EvaluationConfig, EvaluationMode, InferenceBackend, ModelConfig
from evaluation.checkpoint import build_checkpoint_context, resolve_checkpoint_path
from evaluation.pipeline import EvaluationPipeline
from evaluation.runner import EvaluationRunner
from evaluation.strategies import MarkdownEvaluator
from evaluation.types import InferenceResult


class FixedOutputModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def run(self, prompts, images):
        return list(self.outputs)


class FixedBatchModel:
    def __init__(self, predictions: dict[str, str]):
        self.predictions = predictions

    async def run_batch_async(
        self,
        prompts,
        images,
        custom_ids,
        output_dir,
        completion_window,
        poll_interval,
        timeout,
    ):
        return {
            custom_id: self.predictions.get(custom_id, "")
            for custom_id in custom_ids
        }


class RetryThenSuccessModel:
    def __init__(self):
        self.sync_calls = 0
        self.async_calls = 0

    def run(self, prompts, images):
        self.sync_calls += 1
        if self.sync_calls == 1:
            return [""] * len(prompts)
        return ["ok"] * len(prompts)

    async def run_async(self, prompts, images):
        self.async_calls += 1
        if self.async_calls == 1:
            return [""] * len(prompts)
        return ["ok"] * len(prompts)


def make_config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        dataset_id="dummy-dataset",
        split="train",
        model=ModelConfig(model_id="dummy-model", backend=InferenceBackend.OPENAI),
        batch_size=2,
        output_dir=str(tmp_path),
        resume_from_checkpoint=True,
    )


def test_checkpoint_helpers_build_context_and_resolve_path(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    expected = {
        "dataset_id": "dummy-dataset",
        "split": "train",
        "model_id": "dummy-model",
        "backend": "openai",
    }
    assert build_checkpoint_context(config) == expected

    legacy_path = tmp_path / "checkpoint.json"
    plural_path = tmp_path / "checkpoints.json"

    assert resolve_checkpoint_path(tmp_path) is None

    legacy_path.write_text("{}", encoding="utf-8")
    assert resolve_checkpoint_path(tmp_path) == legacy_path

    plural_path.write_text("{}", encoding="utf-8")
    assert resolve_checkpoint_path(tmp_path) == plural_path


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
            "split": "train",
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


def test_runner_ignores_invalid_batch_error_file(tmp_path: Path) -> None:
    runner = EvaluationRunner(make_config(tmp_path), FixedOutputModel(["ok"]))
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "batch_errors.json").write_text("{bad-json", encoding="utf-8")

    assert runner._load_batch_errors(batch_dir) == {}


def test_runner_batch_api_ignores_invalid_batch_metadata_and_submits_new_batch(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path).model_copy(update={"batch_api": True, "batch_size": 1})
    runner = EvaluationRunner(config, FixedBatchModel({"0": "predicted"}))

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "batch_info.json").write_text("{broken-json", encoding="utf-8")

    images = [Image.new("RGB", (4, 4))]
    ground_truths = ["gt"]
    prompts = ["prompt"]

    results = asyncio.run(runner.run_async(images, ground_truths, prompts))

    assert len(results) == 1
    assert results[0].prediction == "predicted"
    assert results[0].error is None


def test_runner_empty_prediction_retry_uses_sync_model_retry_path(tmp_path: Path) -> None:
    config = make_config(tmp_path).model_copy(update={"batch_size": 1})
    model = RetryThenSuccessModel()
    runner = EvaluationRunner(config, model)
    runner._empty_prediction_max_retries = 2
    runner._empty_prediction_retry_backoff_seconds = 0

    images = [Image.new("RGB", (4, 4))]
    ground_truths = ["gt"]
    prompts = ["prompt"]

    results = runner.run(images, ground_truths, prompts)

    assert len(results) == 1
    assert results[0].prediction == "ok"
    assert results[0].error is None
    assert model.sync_calls == 2


def test_runner_empty_prediction_retry_uses_async_model_retry_path(tmp_path: Path) -> None:
    config = make_config(tmp_path).model_copy(update={"batch_size": 1})
    model = RetryThenSuccessModel()
    runner = EvaluationRunner(config, model)
    runner._empty_prediction_max_retries = 2
    runner._empty_prediction_retry_backoff_seconds = 0

    images = [Image.new("RGB", (4, 4))]
    ground_truths = ["gt"]
    prompts = ["prompt"]

    results = asyncio.run(runner.run_async(images, ground_truths, prompts))

    assert len(results) == 1
    assert results[0].prediction == "ok"
    assert results[0].error is None
    assert model.async_calls == 2


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

    assert metrics["avg_markdown_text_score"] == 1.0
    assert metrics["avg_markdown_overall_score"] == 1.0


def test_markdown_evaluator_returns_block_scores() -> None:
    evaluator = MarkdownEvaluator()
    metrics = evaluator.compute_metrics(
        predictions=["Text before\n\n|A|B|\n|---|---|\n|1|2|\n\n$x+y$"],
        ground_truths=["Text before\n\n|A|B|\n|---|---|\n|1|2|\n\n$x+y$"],
    )

    assert metrics["avg_markdown_text_score"] == 1.0
    assert metrics["avg_markdown_table_teds"] == 1.0
    assert metrics["avg_markdown_formula_score"] == 1.0
    assert metrics["avg_markdown_order_score"] == 1.0
    assert metrics["avg_markdown_overall_score"] == 1.0


def test_pipeline_evaluate_only_uses_checkpoint_without_model_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "checkpoints.json"
    checkpoint_payload = {
        "completed": [0],
        "results": [
            {
                "index": 0,
                "prediction": "abc",
                "ground_truth": "abc",
                "latency_ms": 3.0,
                "error": None,
            }
        ],
        "context": {
            "dataset_id": "dummy-dataset",
            "split": "train",
            "model_id": "dummy-model",
            "backend": "openai",
        },
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    monkeypatch.setattr(
        "evaluation.pipeline.create_model",
        lambda _config: (_ for _ in ()).throw(RuntimeError("should not initialize model")),
    )

    config = EvaluationConfig(
        dataset_id="dummy-dataset",
        split="train",
        model=ModelConfig(model_id="dummy-model", backend=InferenceBackend.OPENAI),
        output_dir=str(tmp_path),
        execution_mode=EvaluationMode.EVALUATE_ONLY,
    )

    pipeline = EvaluationPipeline(config)
    output = pipeline.run_evaluate_only()

    assert output.summary["total_samples"] == 1
    assert output.summary["successful"] == 1
    assert output.metrics["avg_markdown_overall_score"] == 1.0


def test_pipeline_evaluate_only_fails_on_context_mismatch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints.json"
    checkpoint_payload = {
        "completed": [0],
        "results": [
            {
                "index": 0,
                "prediction": "abc",
                "ground_truth": "abc",
                "latency_ms": 3.0,
                "error": None,
            }
        ],
        "context": {
            "dataset_id": "other-dataset",
            "split": "train",
            "model_id": "dummy-model",
            "backend": "openai",
        },
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    config = EvaluationConfig(
        dataset_id="dummy-dataset",
        split="train",
        model=ModelConfig(model_id="dummy-model", backend=InferenceBackend.OPENAI),
        output_dir=str(tmp_path),
        execution_mode=EvaluationMode.EVALUATE_ONLY,
    )
    pipeline = EvaluationPipeline(config)

    with pytest.raises(RuntimeError, match="Checkpoint context mismatch"):
        pipeline.run_evaluate_only()
