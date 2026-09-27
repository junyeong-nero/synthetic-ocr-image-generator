import json
import math

import pytest

from src.evaluation.rank_correlation import (
    DEFAULT_METRIC,
    ModelRankRow,
    compute_rank_correlation,
    load_aliases,
    load_real_scores,
    load_synthetic_scores,
    normalize_model_id,
    render_markdown_report,
)


# ---------------------------------------------------------------------------
# normalize_model_id / aliases
# ---------------------------------------------------------------------------


def test_normalize_model_id_strips_org_prefix_and_punctuation() -> None:
    assert normalize_model_id("lightonai/LightOnOCR-2-1B") == "lightonocr21b"
    assert normalize_model_id("./weights/DotsOCR") == "dotsocr"
    assert normalize_model_id("  GPT-5 Mini  ") == "gpt5mini"


def test_normalize_model_id_handles_empty() -> None:
    assert normalize_model_id("") == ""
    assert normalize_model_id(None) == ""


def test_load_aliases_normalizes_both_sides(tmp_path) -> None:
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(json.dumps({"ChatGPT-4o": "openai/gpt-4o-2024-08-06"}), encoding="utf-8")

    aliases = load_aliases(alias_path)

    assert aliases == {"chatgpt4o": "gpt4o20240806"}


def test_load_aliases_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_aliases(tmp_path / "missing.json")


def test_load_aliases_none_returns_empty_dict() -> None:
    assert load_aliases(None) == {}


# ---------------------------------------------------------------------------
# load_real_scores
# ---------------------------------------------------------------------------


def test_load_real_scores_csv_with_named_columns(tmp_path) -> None:
    csv_path = tmp_path / "real.csv"
    csv_path.write_text("model,score\nModelA,0.9\nModelB,0.8\n", encoding="utf-8")

    scores = load_real_scores(csv_path)

    assert scores == {"ModelA": 0.9, "ModelB": 0.8}


def test_load_real_scores_csv_falls_back_to_first_two_columns(tmp_path) -> None:
    csv_path = tmp_path / "real.csv"
    csv_path.write_text("name,accuracy\nModelA,0.9\nModelB,0.8\n", encoding="utf-8")

    scores = load_real_scores(csv_path)

    assert scores == {"ModelA": 0.9, "ModelB": 0.8}


def test_load_real_scores_json_dict(tmp_path) -> None:
    json_path = tmp_path / "real.json"
    json_path.write_text(json.dumps({"ModelA": 0.9, "ModelB": 0.8}), encoding="utf-8")

    scores = load_real_scores(json_path)

    assert scores == {"ModelA": 0.9, "ModelB": 0.8}


def test_load_real_scores_json_list_of_objects(tmp_path) -> None:
    json_path = tmp_path / "real.json"
    json_path.write_text(
        json.dumps([{"model": "ModelA", "score": 0.9}, {"model": "ModelB", "score": 0.8}]),
        encoding="utf-8",
    )

    scores = load_real_scores(json_path)

    assert scores == {"ModelA": 0.9, "ModelB": 0.8}


def test_load_real_scores_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_real_scores(tmp_path / "missing.csv")


def test_load_real_scores_unsupported_extension_raises(tmp_path) -> None:
    bad_path = tmp_path / "real.txt"
    bad_path.write_text("ModelA 0.9", encoding="utf-8")
    with pytest.raises(ValueError):
        load_real_scores(bad_path)


# ---------------------------------------------------------------------------
# load_synthetic_scores
# ---------------------------------------------------------------------------


def _leaderboard_payload(languages: list[dict]) -> dict:
    return {"generated_at_utc": "2026-01-01T00:00:00Z", "count": len(languages), "languages": languages}


def test_load_synthetic_scores_from_leaderboard_json_single_language(tmp_path) -> None:
    payload = _leaderboard_payload(
        [
            {
                "language": "ko",
                "metric_key": DEFAULT_METRIC,
                "rows": [
                    {"model_id": "org/ModelA", "metrics": {DEFAULT_METRIC: 0.9}},
                    {"model_id": "org/ModelB", "metrics": {DEFAULT_METRIC: 0.8}},
                ],
            }
        ]
    )
    path = tmp_path / "leaderboard.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    scores = load_synthetic_scores(path, metric=DEFAULT_METRIC)

    assert scores == {"org/ModelA": 0.9, "org/ModelB": 0.8}


def test_load_synthetic_scores_from_leaderboard_json_requires_language_when_ambiguous(tmp_path) -> None:
    payload = _leaderboard_payload(
        [
            {"language": "ko", "rows": [{"model_id": "org/ModelA", "metrics": {DEFAULT_METRIC: 0.9}}]},
            {"language": "ja", "rows": [{"model_id": "org/ModelA", "metrics": {DEFAULT_METRIC: 0.5}}]},
        ]
    )
    path = tmp_path / "leaderboard.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="multiple languages"):
        load_synthetic_scores(path, metric=DEFAULT_METRIC)


def test_load_synthetic_scores_from_leaderboard_json_with_language_filter(tmp_path) -> None:
    payload = _leaderboard_payload(
        [
            {"language": "ko", "rows": [{"model_id": "org/ModelA", "metrics": {DEFAULT_METRIC: 0.9}}]},
            {"language": "ja", "rows": [{"model_id": "org/ModelA", "metrics": {DEFAULT_METRIC: 0.5}}]},
        ]
    )
    path = tmp_path / "leaderboard.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    scores = load_synthetic_scores(path, metric=DEFAULT_METRIC, language="ja")

    assert scores == {"org/ModelA": 0.5}


def test_load_synthetic_scores_from_single_report_json(tmp_path) -> None:
    payload = {
        "config": {"model": {"model_id": "org/ModelA"}},
        "metrics": {DEFAULT_METRIC: 0.77},
        "summary": {},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    scores = load_synthetic_scores(path, metric=DEFAULT_METRIC)

    assert scores == {"org/ModelA": 0.77}


def test_load_synthetic_scores_from_single_report_json_falls_back_to_summary_model_id(tmp_path) -> None:
    payload = {
        "config": {},
        "metrics": {DEFAULT_METRIC: 0.5},
        "summary": {"model_id": "paddleocr/paddle-ocr"},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    scores = load_synthetic_scores(path, metric=DEFAULT_METRIC)

    assert scores == {"paddleocr/paddle-ocr": 0.5}


def test_load_synthetic_scores_from_directory_scans_report_json(tmp_path) -> None:
    for model_dir, model_id, score in (("ModelA", "org/ModelA", 0.9), ("ModelB", "org/ModelB", 0.6)):
        report_dir = tmp_path / model_dir / "ko"
        report_dir.mkdir(parents=True)
        payload = {
            "config": {"model": {"model_id": model_id}, "language": "ko", "dataset_id": "ds"},
            "metrics": {DEFAULT_METRIC: score},
            "summary": {},
        }
        (report_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    scores = load_synthetic_scores(tmp_path, metric=DEFAULT_METRIC, language="ko")

    assert scores == {"org/ModelA": 0.9, "org/ModelB": 0.6}


def test_load_synthetic_scores_missing_source_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_synthetic_scores(tmp_path / "missing.json", metric=DEFAULT_METRIC)


def test_load_synthetic_scores_unrecognized_format_raises(tmp_path) -> None:
    path = tmp_path / "weird.json"
    path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized"):
        load_synthetic_scores(path, metric=DEFAULT_METRIC)


def test_load_synthetic_scores_missing_metric_raises(tmp_path) -> None:
    payload = _leaderboard_payload(
        [{"language": "ko", "rows": [{"model_id": "org/ModelA", "metrics": {"other_metric": 0.9}}]}]
    )
    path = tmp_path / "leaderboard.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="No models found"):
        load_synthetic_scores(path, metric=DEFAULT_METRIC)


# ---------------------------------------------------------------------------
# compute_rank_correlation
# ---------------------------------------------------------------------------


def test_compute_rank_correlation_perfect_agreement_no_ties() -> None:
    synthetic = {"org/A": 0.9, "org/B": 0.8, "org/C": 0.7, "org/D": 0.6}
    real = {"A": 95, "B": 85, "C": 75, "D": 65}

    result = compute_rank_correlation(synthetic, real)

    assert result.n == 4
    assert result.spearman_rho == pytest.approx(1.0)
    assert result.kendall_tau == pytest.approx(1.0)
    assert result.unmatched_synthetic == []
    assert result.unmatched_real == []


def test_compute_rank_correlation_perfect_inverse_agreement() -> None:
    synthetic = {"org/A": 0.9, "org/B": 0.8, "org/C": 0.7, "org/D": 0.6}
    real = {"A": 10, "B": 20, "C": 30, "D": 40}

    result = compute_rank_correlation(synthetic, real)

    assert result.spearman_rho == pytest.approx(-1.0)
    assert result.kendall_tau == pytest.approx(-1.0)


def test_compute_rank_correlation_with_ties_matches_scipy_reference() -> None:
    # Known reference values computed directly via scipy.stats for this
    # dataset (B and C tie on the synthetic side).
    synthetic = {"org/A": 0.90, "org/B": 0.80, "org/C": 0.80, "org/D": 0.60, "org/E": 0.50}
    real = {"A": 88, "B": 75, "C": 79, "D": 65, "E": 50}

    result = compute_rank_correlation(synthetic, real, n_bootstrap=0)

    assert result.spearman_rho == pytest.approx(0.9746794344808963)
    assert result.kendall_tau == pytest.approx(0.9486832980505138)
    assert result.spearman_pvalue == pytest.approx(0.004818230468198566)
    assert result.kendall_pvalue == pytest.approx(0.022977401503206065)
    # B and C are tied on the synthetic side -> average rank 2.5 each.
    row_by_model = {row.synthetic_model_id: row for row in result.rows}
    assert row_by_model["org/B"].synthetic_rank == pytest.approx(2.5)
    assert row_by_model["org/C"].synthetic_rank == pytest.approx(2.5)


def test_compute_rank_correlation_bootstrap_ci_brackets_point_estimate() -> None:
    synthetic = {"org/A": 0.90, "org/B": 0.80, "org/C": 0.70, "org/D": 0.55, "org/E": 0.50, "org/F": 0.40}
    real = {"A": 88, "B": 60, "C": 79, "D": 50, "E": 65, "F": 30}

    result = compute_rank_correlation(synthetic, real, n_bootstrap=500, seed=7)

    assert result.spearman_ci[0] <= result.spearman_rho <= result.spearman_ci[1] or math.isnan(
        result.spearman_ci[0]
    )
    assert result.kendall_ci[0] <= result.kendall_tau <= result.kendall_ci[1] or math.isnan(
        result.kendall_ci[0]
    )


def test_compute_rank_correlation_alias_matching() -> None:
    synthetic = {"lightonai/LightOnOCR-2-1B": 0.9, "org/OtherModel": 0.5}
    real = {"lighton-2-1b": 80, "othermodel": 40}
    aliases = {"lighton-2-1b": "lightonai/LightOnOCR-2-1B"}

    result = compute_rank_correlation(synthetic, real, aliases=_load_aliases_dict(aliases))

    assert result.n == 2
    assert result.unmatched_synthetic == []
    assert result.unmatched_real == []


def _load_aliases_dict(raw: dict) -> dict:
    from src.evaluation.rank_correlation import normalize_model_id as _norm

    return {_norm(k): _norm(v) for k, v in raw.items()}


def test_compute_rank_correlation_reports_unmatched_models() -> None:
    synthetic = {"org/A": 0.9, "org/B": 0.8, "org/OnlySynthetic": 0.1}
    real = {"A": 90, "B": 80, "OnlyReal": 5}

    result = compute_rank_correlation(synthetic, real)

    assert result.n == 2
    assert result.unmatched_synthetic == ["org/OnlySynthetic"]
    assert result.unmatched_real == ["OnlyReal"]


def test_compute_rank_correlation_raises_when_fewer_than_two_matches() -> None:
    synthetic = {"org/A": 0.9, "org/Other": 0.1}
    real = {"A": 90, "Different": 5}

    with pytest.raises(ValueError, match="at least 2"):
        compute_rank_correlation(synthetic, real)


def test_compute_rank_correlation_ambiguous_match_key_raises() -> None:
    synthetic = {"org1/Model": 0.9, "org2/Model": 0.5}
    real = {"model": 50}

    with pytest.raises(ValueError, match="Ambiguous"):
        compute_rank_correlation(synthetic, real)


# ---------------------------------------------------------------------------
# render_markdown_report
# ---------------------------------------------------------------------------


def test_render_markdown_report_contains_per_model_table() -> None:
    synthetic = {"org/A": 0.9, "org/B": 0.8, "org/OnlySynthetic": 0.1}
    real = {"A": 90, "B": 80, "OnlyReal": 5}
    result = compute_rank_correlation(synthetic, real, n_bootstrap=0)

    report = render_markdown_report(result, synthetic_source="lb.json", real_source="real.csv")

    assert "# Synthetic vs Real Rank Correlation" in report
    assert f"`{DEFAULT_METRIC}`" in report
    assert "Spearman rho" in report
    assert "Kendall tau" in report
    assert "| Real Rank | Synthetic Model | Real Model |" in report
    assert "org/A" in report
    assert "org/B" in report
    assert "## Unmatched (synthetic only)" in report
    assert "org/OnlySynthetic" in report
    assert "## Unmatched (real only)" in report
    assert "OnlyReal" in report
    assert "lb.json" in report
    assert "real.csv" in report


def test_model_rank_row_is_a_dataclass_with_expected_fields() -> None:
    row = ModelRankRow(
        match_key="a",
        synthetic_model_id="org/A",
        real_model_name="A",
        synthetic_score=0.9,
        real_score=90.0,
        synthetic_rank=1.0,
        real_rank=1.0,
    )
    assert row.synthetic_model_id == "org/A"
