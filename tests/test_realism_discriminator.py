"""Tests for the real-vs-synthetic classifier two-sample test (C2ST).

`src/realism/discriminator.py` trains a small numpy-only L2 logistic
regression on per-image statistic rows (as produced by
`distribution measure --save-rows`) to tell whether a reference (real) set
and a candidate (synthetic) set are distinguishable on those statistics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.realism.discriminator import run_discriminator


def _rows(feature_values: dict, path_prefix: str, n: int) -> list[dict]:
    """Build `n` rows with the given per-feature arrays and a `path` field."""
    rows = []
    for i in range(n):
        row = {name: float(values[i]) for name, values in feature_values.items()}
        row["path"] = f"{path_prefix}{i}.png"
        rows.append(row)
    return rows


def test_separable_feature_gives_high_auc_and_is_ranked_first() -> None:
    rng = np.random.RandomState(1)
    n = 60
    reference_rows = _rows(
        {"feat_sep": rng.normal(0.0, 1.0, n), "feat_noise": rng.normal(0.0, 1.0, n)},
        "ref_",
        n,
    )
    candidate_rows = _rows(
        {"feat_sep": rng.normal(6.0, 1.0, n), "feat_noise": rng.normal(0.0, 1.0, n)},
        "cand_",
        n,
    )

    result = run_discriminator(reference_rows, candidate_rows, seed=0)

    assert result["auc_mean"] > 0.9
    assert result["coefficients"][0]["feature"] == "feat_sep"


def test_identical_distributions_give_auc_near_half() -> None:
    rng = np.random.RandomState(2)
    n = 100

    def make(prefix: str) -> list[dict]:
        return _rows(
            {"feat_a": rng.normal(0.0, 1.0, n), "feat_b": rng.normal(0.0, 1.0, n)},
            prefix,
            n,
        )

    reference_rows = make("ref_")
    candidate_rows = make("cand_")

    result = run_discriminator(reference_rows, candidate_rows, seed=0)

    assert 0.3 < result["auc_mean"] < 0.7


def test_run_discriminator_is_deterministic() -> None:
    rng = np.random.RandomState(3)
    n = 40
    reference_rows = _rows({"feat": rng.normal(0.0, 1.0, n)}, "ref_", n)
    candidate_rows = _rows({"feat": rng.normal(1.0, 1.0, n)}, "cand_", n)

    result_a = run_discriminator(reference_rows, candidate_rows, seed=7)
    result_b = run_discriminator(reference_rows, candidate_rows, seed=7)

    assert result_a == result_b


def test_top_candidates_are_sorted_by_probability_and_carry_paths() -> None:
    rng = np.random.RandomState(4)
    n = 50
    reference_rows = _rows({"feat": rng.normal(0.0, 1.0, n)}, "ref_", n)
    candidate_rows = _rows({"feat": rng.normal(5.0, 1.0, n)}, "cand_", n)

    result = run_discriminator(reference_rows, candidate_rows, seed=0, top_n=5)

    top = result["top_candidates"]
    assert len(top) == 5
    probs = [item["probability"] for item in top]
    assert probs == sorted(probs, reverse=True)
    assert all(item["path"].startswith("cand_") for item in top)


def test_fold_spread_is_reported() -> None:
    rng = np.random.RandomState(5)
    n = 60
    reference_rows = _rows({"feat": rng.normal(0.0, 1.0, n)}, "ref_", n)
    candidate_rows = _rows({"feat": rng.normal(2.0, 1.0, n)}, "cand_", n)

    result = run_discriminator(reference_rows, candidate_rows, seed=0, n_splits=5)

    assert len(result["fold_aucs"]) == result["n_splits"]
    assert len(result["fold_accuracies"]) == result["n_splits"]
    assert result["auc_std"] >= 0.0


def test_small_class_size_clamps_fold_count() -> None:
    reference_rows = _rows({"feat": [0.0, 0.1, 0.2]}, "ref_", 3)
    candidate_rows = _rows({"feat": [5.0, 5.1, 5.2]}, "cand_", 3)

    result = run_discriminator(reference_rows, candidate_rows, seed=0, n_splits=5)

    assert 2 <= result["n_splits"] <= 3


def test_raises_when_no_shared_numeric_features() -> None:
    reference_rows = [{"path": "ref_0.png"}]
    candidate_rows = [{"path": "cand_0.png"}]

    with pytest.raises(ValueError):
        run_discriminator(reference_rows, candidate_rows, seed=0)


def test_balances_unequal_class_sizes() -> None:
    rng = np.random.RandomState(6)
    reference_rows = _rows({"feat": rng.normal(0.0, 1.0, 20)}, "ref_", 20)
    candidate_rows = _rows({"feat": rng.normal(3.0, 1.0, 200)}, "cand_", 200)

    result = run_discriminator(reference_rows, candidate_rows, seed=0)

    assert result["n_balanced_per_class"] == 20
    assert result["n_reference"] == 20
    assert result["n_candidate"] == 200
