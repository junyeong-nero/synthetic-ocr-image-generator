"""Tests for the `distribution discriminate` CLI subcommand."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from src.cli import distribution


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    distribution.configure_parser(parser)
    return parser.parse_args(argv)


def _write_stats_with_rows(path, feature_mean: float, n: int, prefix: str, seed: int) -> None:
    rng = np.random.RandomState(seed)
    rows = [
        {"feat": float(v), "path": f"{prefix}{i}.png"}
        for i, v in enumerate(rng.normal(feature_mean, 1.0, n))
    ]
    path.write_text(
        json.dumps({"source": prefix, "count": n, "metrics": {}, "rows": rows}),
        encoding="utf-8",
    )


def test_discriminate_subcommand_is_registered() -> None:
    args = _parse(
        ["discriminate", "--reference", "ref.json", "--candidate", "cand.json"]
    )
    assert args.handler is distribution.run_discriminate


def test_run_discriminate_writes_report(tmp_path) -> None:
    reference_path = tmp_path / "real.json"
    candidate_path = tmp_path / "synthetic.json"
    _write_stats_with_rows(reference_path, 0.0, 60, "real_", seed=10)
    _write_stats_with_rows(candidate_path, 4.0, 60, "syn_", seed=11)

    output = tmp_path / "report.md"
    args = _parse(
        [
            "discriminate",
            "--reference",
            str(reference_path),
            "--candidate",
            str(candidate_path),
            "--top",
            "5",
            "--output",
            str(output),
        ]
    )
    rc = distribution.run_discriminate(args)
    assert rc == 0

    report = output.read_text(encoding="utf-8")
    assert "AUC" in report
    assert "feat" in report
    assert "syn_" in report  # top-candidate paths


def test_run_discriminate_errors_without_saved_rows(tmp_path) -> None:
    reference_path = tmp_path / "real.json"
    candidate_path = tmp_path / "synthetic.json"
    # No `rows` key: as if `distribution measure` ran without --save-rows.
    reference_path.write_text(json.dumps({"source": "real", "count": 1, "metrics": {}}), encoding="utf-8")
    candidate_path.write_text(json.dumps({"source": "syn", "count": 1, "metrics": {}}), encoding="utf-8")

    args = _parse(
        ["discriminate", "--reference", str(reference_path), "--candidate", str(candidate_path)]
    )
    rc = distribution.run_discriminate(args)
    assert rc == 1
