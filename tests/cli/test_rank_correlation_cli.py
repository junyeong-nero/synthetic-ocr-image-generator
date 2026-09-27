import argparse
import json

from src.cli.rank_correlation import add_arguments, run_with_args


def _build_args(**overrides) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    defaults = {
        "synthetic": None,
        "real": None,
        "metric": None,
        "aliases": None,
        "language": None,
        "output": None,
        "bootstrap_iterations": 0,
        "seed": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_leaderboard(path, rows) -> None:
    payload = {
        "generated_at_utc": "2026-01-01T00:00:00Z",
        "count": 1,
        "languages": [
            {
                "language": "ko",
                "metric_key": "avg_markdown_overall_score",
                "rows": rows,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_with_args_prints_report_and_writes_output(tmp_path, capsys) -> None:
    synthetic_path = tmp_path / "leaderboard.json"
    _write_leaderboard(
        synthetic_path,
        [
            {"model_id": "org/ModelA", "metrics": {"avg_markdown_overall_score": 0.9}},
            {"model_id": "org/ModelB", "metrics": {"avg_markdown_overall_score": 0.8}},
            {"model_id": "org/ModelC", "metrics": {"avg_markdown_overall_score": 0.7}},
        ],
    )
    real_path = tmp_path / "real.csv"
    real_path.write_text("model,score\nModelA,95\nModelB,85\nModelC,75\n", encoding="utf-8")
    output_path = tmp_path / "report.md"

    args = _build_args(synthetic=str(synthetic_path), real=str(real_path), output=str(output_path))
    exit_code = run_with_args(args)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Synthetic vs Real Rank Correlation" in captured.out
    assert output_path.exists()
    assert "Spearman rho" in output_path.read_text(encoding="utf-8")


def test_run_with_args_returns_error_code_for_missing_synthetic_file(tmp_path, capsys) -> None:
    real_path = tmp_path / "real.csv"
    real_path.write_text("model,score\nModelA,95\n", encoding="utf-8")

    args = _build_args(synthetic=str(tmp_path / "missing.json"), real=str(real_path))
    exit_code = run_with_args(args)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err


def test_run_with_args_returns_error_code_when_too_few_matches(tmp_path, capsys) -> None:
    synthetic_path = tmp_path / "leaderboard.json"
    _write_leaderboard(
        synthetic_path,
        [{"model_id": "org/ModelA", "metrics": {"avg_markdown_overall_score": 0.9}}],
    )
    real_path = tmp_path / "real.csv"
    real_path.write_text("model,score\nOtherModel,95\n", encoding="utf-8")

    args = _build_args(synthetic=str(synthetic_path), real=str(real_path))
    exit_code = run_with_args(args)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "at least 2" in captured.err


def test_run_with_args_uses_aliases_file(tmp_path, capsys) -> None:
    synthetic_path = tmp_path / "leaderboard.json"
    _write_leaderboard(
        synthetic_path,
        [
            {"model_id": "lightonai/LightOnOCR-2-1B", "metrics": {"avg_markdown_overall_score": 0.9}},
            {"model_id": "org/ModelB", "metrics": {"avg_markdown_overall_score": 0.8}},
        ],
    )
    real_path = tmp_path / "real.csv"
    real_path.write_text("model,score\nlighton-2-1b,90\nModelB,80\n", encoding="utf-8")
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text(
        json.dumps({"lighton-2-1b": "lightonai/LightOnOCR-2-1B"}), encoding="utf-8"
    )

    args = _build_args(synthetic=str(synthetic_path), real=str(real_path), aliases=str(aliases_path))
    exit_code = run_with_args(args)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "0 synthetic-only, 0 real-only" in captured.out


def test_add_arguments_registers_required_flags() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["--synthetic", "lb.json", "--real", "real.csv"])
    assert args.synthetic == "lb.json"
    assert args.real == "real.csv"
    assert args.metric is None
    assert args.output is None
