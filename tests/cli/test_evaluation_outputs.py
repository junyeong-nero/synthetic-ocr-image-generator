import json
from types import SimpleNamespace

import pytest

from cli.evaluation_outputs import (
    build_summary_entry,
    load_summary_entries,
    persist_evaluation_outputs,
    save_summary_entries,
    write_leaderboard,
    write_protocol_snapshot,
)


def _make_output() -> SimpleNamespace:
    return SimpleNamespace(
        config={
            "format": "markdown",
            "prompt": "extract markdown",
            "prompt_source": "model_config",
            "system_prompt": "system",
        },
        summary={
            "total_samples": 10,
            "successful": 9,
            "failed": 1,
            "avg_latency_ms": 12.5,
            "empty_rate": 0.1,
            "parse_fail_rate": 0.0,
        },
        metrics={
            "avg_markdown_overall_score": 0.9,
            "avg_markdown_text_score": 0.8,
            "avg_markdown_table_teds": 1.0,
            "avg_markdown_formula_score": 0.7,
            "avg_markdown_order_score": 0.95,
        },
    )


def test_write_protocol_snapshot_writes_expected_keys(tmp_path) -> None:
    output = _make_output()
    path = write_protocol_snapshot(tmp_path, output, "all")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "protocol.json"
    assert payload["protocol_version"] == "1.0"
    assert payload["command"] == "evaluate"
    assert payload["report_format"] == "all"
    assert payload["config"]["prompt"] == "extract markdown"
    assert payload["prompt_source"] == "model_config"
    assert payload["system_prompt"] == "system"
    assert payload["summary"]["total_samples"] == 10
    assert isinstance(payload["timestamp"], str)


def test_summary_load_save_round_trip_and_single_dict_upgrade(tmp_path) -> None:
    summary_path = tmp_path / "model_summary.json"
    entries = [{"model_id": "a"}, {"model_id": "b"}]
    save_summary_entries(summary_path, entries)
    assert load_summary_entries(summary_path) == entries

    summary_path.write_text(json.dumps({"model_id": "legacy"}), encoding="utf-8")
    assert load_summary_entries(summary_path) == [{"model_id": "legacy"}]


def test_load_summary_entries_moves_corrupt_file(tmp_path) -> None:
    summary_path = tmp_path / "model_summary.json"
    summary_path.write_text("{broken-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Corrupt summary file moved"):
        load_summary_entries(summary_path)

    assert not summary_path.exists()
    assert (tmp_path / "model_summary.corrupt.json").exists()


def test_write_leaderboard_outputs_sorted_json_and_markdown(tmp_path) -> None:
    entries = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "protocol_version": "1.0",
            "model_id": "model-low",
            "backend": "openai",
            "dataset": "repo/data-ko",
            "split": "train",
            "language": "ko",
            "metric_key": "avg_markdown_overall_score",
            "metric_value": 0.3,
            "average_score": 0.3,
            "empty_rate": 0.0,
            "parse_fail_rate": 0.0,
            "metrics": {"avg_markdown_text_score": 0.3},
        },
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "protocol_version": "1.0",
            "model_id": "model-high",
            "backend": "openai",
            "dataset": "repo/data-ko",
            "split": "train",
            "language": "ko",
            "metric_key": "avg_markdown_overall_score",
            "metric_value": 0.9,
            "average_score": 0.9,
            "empty_rate": 0.0,
            "parse_fail_rate": 0.0,
            "metrics": {"avg_markdown_text_score": 0.9},
        },
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "protocol_version": "1.0",
            "model_id": "model-ja",
            "backend": "openai",
            "dataset": "repo/data-ja",
            "split": "train",
            "language": "ja",
            "metric_key": "avg_markdown_overall_score",
            "metric_value": 0.5,
            "average_score": 0.5,
            "empty_rate": 0.1,
            "parse_fail_rate": 0.0,
            "metrics": {"avg_markdown_text_score": 0.5},
        },
    ]

    write_leaderboard(tmp_path, entries)

    json_payload = json.loads((tmp_path / "leaderboard.json").read_text(encoding="utf-8"))
    assert [row["language"] for row in json_payload] == ["ja", "ko", "ko"]
    assert [row["model_id"] for row in json_payload] == ["model-ja", "model-high", "model-low"]

    md_text = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert "# OCR Benchmark Leaderboard" in md_text
    assert "## ja" in md_text
    assert "## ko" in md_text
    assert "model-high" in md_text


def test_persist_evaluation_outputs_writes_all_artifacts(tmp_path) -> None:
    output = _make_output()
    protocol_path, summary_path = persist_evaluation_outputs(
        output_dir=tmp_path,
        output=output,
        report_format="all",
        model_id="model-a",
        backend="openai",
        dataset="repo/data-ko",
        split="train",
        language="ko",
        seed=123,
        resolved_format="markdown",
        metric_key="avg_markdown_overall_score",
        metric_value=0.9,
    )

    assert protocol_path == tmp_path / "protocol.json"
    assert summary_path == tmp_path / "model_summary.json"
    assert protocol_path.exists()
    assert summary_path.exists()
    assert (tmp_path / "leaderboard.json").exists()
    assert (tmp_path / "leaderboard.md").exists()

    saved_entries = load_summary_entries(summary_path)
    assert len(saved_entries) == 1
    saved = saved_entries[0]
    assert saved["model_id"] == "model-a"
    assert saved["metric_value"] == 0.9
    assert saved["seed"] == 123


def test_build_summary_entry_filters_non_numeric_metrics() -> None:
    output = _make_output()
    output.metrics["note"] = "n/a"
    entry = build_summary_entry(
        output=output,
        model_id="model-a",
        backend="openai",
        dataset="repo/data-ko",
        split="train",
        language="ko",
        seed=None,
        resolved_format="markdown",
        metric_key="avg_markdown_overall_score",
        metric_value=0.9,
    )

    assert "note" not in entry["metrics"]
    assert entry["metrics"]["avg_markdown_overall_score"] == 0.9
