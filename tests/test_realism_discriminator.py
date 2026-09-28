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


def test_top_candidates_are_not_biased_toward_in_sample_rows() -> None:
    """Regression test for PR #11 review: candidate rows that happened to be
    drawn into the balanced training subsample must be scored with their
    cross-validation out-of-fold probability, not the final model's
    in-sample (inflated) probability -- otherwise those rows systematically
    dominate the top-N ranking, exactly the list users act on.

    With fully overlapping (statistically identical) distributions and a
    candidate set 10x the reference, only `n_reference` of the candidates
    ever touch training. If in-sample rows were scored in-sample, an
    overfit-prone small-sample fit would push most/all of the top-N ranking
    to be in-sample rows; with proper out-of-fold scoring, the top-N should
    reflect the true (~10%) population share of in-sample candidates.
    """
    rng = np.random.RandomState(11)
    n_reference = 40
    n_candidate = 400
    reference_rows = _rows(
        {"feat_a": rng.normal(0.0, 1.0, n_reference), "feat_b": rng.normal(0.0, 1.0, n_reference)},
        "ref_",
        n_reference,
    )
    candidate_rows = _rows(
        {"feat_a": rng.normal(0.0, 1.0, n_candidate), "feat_b": rng.normal(0.0, 1.0, n_candidate)},
        "cand_",
        n_candidate,
    )

    result = run_discriminator(reference_rows, candidate_rows, seed=0, top_n=50)

    assert result["n_balanced_per_class"] == n_reference

    # Every top-N entry must say where its probability came from.
    sources = {item["score_source"] for item in result["top_candidates"]}
    assert sources <= {"cv_out_of_fold", "final_model_holdout"}

    in_sample_share_in_top = sum(
        1 for item in result["top_candidates"] if item["score_source"] == "cv_out_of_fold"
    ) / len(result["top_candidates"])

    # Population share of in-sample candidates is n_reference / n_candidate = 10%.
    # A bias toward in-sample rows would push this close to 1.0.
    assert in_sample_share_in_top <= 0.3


def test_in_sample_candidate_probability_matches_its_out_of_fold_score() -> None:
    """Every candidate row that was part of the balanced training subsample
    must report its cross-validation out-of-fold probability, not a score
    from the final model that was fit on that very row."""
    rng = np.random.RandomState(12)
    n_reference = 15
    n_candidate = 15
    reference_rows = _rows({"feat": rng.normal(0.0, 1.0, n_reference)}, "ref_", n_reference)
    candidate_rows = _rows({"feat": rng.normal(2.0, 1.0, n_candidate)}, "cand_", n_candidate)

    # n_reference == n_candidate, so every candidate row is in the balanced
    # training subsample: all of them must be tagged "cv_out_of_fold".
    result = run_discriminator(reference_rows, candidate_rows, seed=0, top_n=n_candidate)

    assert result["n_balanced_per_class"] == n_candidate
    assert all(item["score_source"] == "cv_out_of_fold" for item in result["top_candidates"])


def test_in_sample_candidate_probability_equals_reproduced_out_of_fold_value() -> None:
    """White-box golden test: for a fully in-sample candidate set
    (n_reference == n_candidate, so every candidate row is drawn into the
    balanced training subsample), each reported probability must exactly
    equal the out-of-fold prediction computed by independently replaying the
    same cross-validation loop with the same seed -- not the final (in-
    sample) model's prediction on that row.
    """
    from src.realism import discriminator as d

    rng_data = np.random.RandomState(13)
    n = 12
    reference_rows = _rows({"feat": rng_data.normal(0.0, 1.0, n)}, "ref_", n)
    candidate_rows = _rows({"feat": rng_data.normal(1.5, 1.0, n)}, "cand_", n)

    seed = 42
    result = run_discriminator(reference_rows, candidate_rows, seed=seed, n_splits=4, top_n=n)

    # Independently replicate the internal CV loop. No rng draws happen
    # before the k-fold split when n_reference == n_candidate (neither
    # `ref_idx` nor `cand_idx` sampling consumes the RNG in that case), so a
    # fresh RandomState(seed) here matches the one `run_discriminator` used
    # for `_stratified_kfold_indices`.
    feature_keys = d.select_feature_keys(reference_rows, candidate_rows)
    X_reference = d._build_feature_matrix(reference_rows, feature_keys)
    X_candidate = d._build_feature_matrix(candidate_rows, feature_keys)
    X_balanced = np.vstack([X_reference, X_candidate])
    y_balanced = np.concatenate([np.zeros(n), np.ones(n)])

    rng = np.random.RandomState(seed)
    n_splits_eff = max(2, min(4, n))
    folds = d._stratified_kfold_indices(y_balanced, n_splits_eff, rng)
    oof = np.full(2 * n, np.nan)
    for k in range(n_splits_eff):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_splits_eff) if j != k])
        X_train, X_test = d._standardize(X_balanced[train_idx], X_balanced[test_idx])
        weights = d.fit_logistic_regression(X_train, y_balanced[train_idx], l2=1.0, n_iter=50)
        oof[test_idx] = d.predict_proba(X_test, weights)

    expected_by_index = {i: round(float(oof[n + i]), 6) for i in range(n)}
    reported_by_index = {item["index"]: item["probability"] for item in result["top_candidates"]}
    assert reported_by_index == expected_by_index
