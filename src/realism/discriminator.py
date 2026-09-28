"""Real-vs-synthetic discriminator: a classifier two-sample test (C2ST).

Trains a small, numpy-only, L2-regularised logistic regression to tell a
reference (real) set of per-image statistic rows apart from a candidate
(synthetic) set -- the rows produced by `distribution measure --save-rows`
(pixel-texture metrics from `src/realism/image_stats.py` merged with layout
metrics from `src/realism/layout_stats.py`, see
`src.realism.distribution_stats.measure_images`).

**How to read the result.** If the classifier cannot beat chance under
cross-validation (AUC close to 0.5), the two sets are statistically
indistinguishable on these features -- a good sign for the synthetic
generator. An AUC approaching 1.0 means the sets are easy to tell apart; the
ranked `coefficients` say *which* statistics the model leans on most (the
biggest gaps to close), and `top_candidates` lists the synthetic images the
model is most confident about, so you can look at the worst offenders
directly instead of only reading aggregate numbers. Every `top_candidates`
entry is a genuine held-out prediction -- see `score_source` below -- so the
ranking is not biased towards whichever rows happened to be used for
training.

**Caveat.** Like the skew and layout estimators it consumes
(`image_stats.estimate_skew`, `layout_stats.compute_layout_stats`), a high
AUC driven mostly by `margin_*_frac` / `column_count` / `text_line_*` on a
photographed-heavy set may reflect that those metrics are unreliable on
photographs (perspective distortion, desk background) rather than a real
distribution gap. Check which features dominate `coefficients` before
concluding the generator itself is off.

No scikit-learn or torch: everything here is numpy plus the standard
library, per the project's dependency-group rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

# Row keys that are identifiers, not statistics, and must never be fed to the
# classifier or averaged into a metric summary.
NON_FEATURE_KEYS = {"path"}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def fit_logistic_regression(
    X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, n_iter: int = 50
) -> np.ndarray:
    """L2-regularised logistic regression via Newton-Raphson (IRLS).

    `X` must already be standardized (see `run_discriminator`); the
    intercept (`weights[0]`) is not regularised. Newton steps converge in a
    handful of iterations on the small, standardized feature matrices this
    module works with; `n_iter` is a generous ceiling, not a tuned budget.
    Returns `weights` of shape `(d + 1,)`: `[intercept, w_1, ..., w_d]`.
    """
    n, d = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])
    weights = np.zeros(d + 1)
    reg = l2 * np.eye(d + 1)
    reg[0, 0] = 0.0
    for _ in range(n_iter):
        z = X_aug @ weights
        p = _sigmoid(z)
        w_diag = np.clip(p * (1.0 - p), 1e-6, None)
        hessian = X_aug.T @ (X_aug * w_diag[:, None]) + reg
        grad = X_aug.T @ (p - y) + l2 * np.concatenate(([0.0], weights[1:]))
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        weights = weights - step
    return weights


def predict_proba(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """P(class=1) for standardized rows `X` under a fitted `weights` vector."""
    X_aug = np.hstack([np.ones((len(X), 1)), X])
    return _sigmoid(X_aug @ weights)


def _average_rank(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given the average rank of their group."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = len(values)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (rank-based, tie-aware).

    Returns 0.5 (chance) if either class is empty in `y_true`, so a
    degenerate fold never produces a NaN that would poison an average.
    """
    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = _average_rank(y_score)
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def select_feature_keys(reference_rows: Sequence[Dict[str, Any]], candidate_rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Numeric keys present on every row of both sides, excluding identifiers."""
    if not reference_rows or not candidate_rows:
        return []

    def numeric_keys(rows: Sequence[Dict[str, Any]]) -> set:
        keys = {k for k, v in rows[0].items() if k not in NON_FEATURE_KEYS and isinstance(v, (int, float))}
        for row in rows[1:]:
            keys &= {k for k in keys if k in row and isinstance(row[k], (int, float))}
        return keys

    shared = numeric_keys(reference_rows) & numeric_keys(candidate_rows)
    return sorted(shared)


def _build_feature_matrix(rows: Sequence[Dict[str, Any]], feature_keys: Sequence[str]) -> np.ndarray:
    return np.array([[float(row[key]) for key in feature_keys] for row in rows], dtype=np.float64)


def _stratified_kfold_indices(y: np.ndarray, n_splits: int, rng: np.random.RandomState) -> List[np.ndarray]:
    """Indices (into `y`) for each of `n_splits` stratified folds."""
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    parts0 = np.array_split(idx0, n_splits)
    parts1 = np.array_split(idx1, n_splits)
    return [np.concatenate([p0, p1]) for p0, p1 in zip(parts0, parts1)]


def _standardize(train: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return tuple((arr - mean) / std for arr in (train, *others))


def run_discriminator(
    reference_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    *,
    n_splits: int = 5,
    l2: float = 1.0,
    top_n: int = 20,
    seed: int = 0,
    n_iter: int = 50,
) -> Dict[str, Any]:
    """Run the classifier two-sample test.

    Class 0 = reference (real), class 1 = candidate (synthetic). Classes are
    balanced by subsampling the larger side down to the smaller side's count
    (seeded, so the same inputs + seed always draw the same subsample).
    Cross-validated AUC/accuracy come from stratified k-fold on that balanced
    set; `coefficients` come from one final model fit on the whole balanced
    set.

    Every `top_candidates` probability is a held-out prediction, never a
    score from a model fit on that same row (`score_source` says which of
    two ways it was obtained):
    - A candidate row drawn into the balanced training subsample was, by
      construction, part of the final model's training data, so its final-
      model score would be in-sample and inflated. It is instead scored
      with its cross-validation **out-of-fold** probability (`"cv_out_of_fold"`)
      -- the prediction from the one fold where that row was held out.
    - A candidate row *not* drawn into the balanced subsample never touched
      any training (neither a CV fold nor the final refit), so the final
      model's prediction on it is a genuine holdout score
      (`"final_model_holdout"`).

    This avoids ranking `top_candidates` by a mix of honest and in-sample
    (overconfident) scores, which would otherwise systematically bias the
    top-N ranking towards whichever candidate rows happened to be drawn into
    the balanced training subsample -- worse the larger the candidate set is
    relative to the reference set.

    Raises `ValueError` if the two sides share no common numeric feature key,
    or if either side has fewer than 2 rows.
    """
    feature_keys = select_feature_keys(reference_rows, candidate_rows)
    if not feature_keys:
        raise ValueError("No shared numeric feature keys between reference and candidate rows")

    n_reference = len(reference_rows)
    n_candidate = len(candidate_rows)
    if n_reference < 2 or n_candidate < 2:
        raise ValueError("Need at least 2 rows on each side to run the discriminator")

    rng = np.random.RandomState(seed)

    X_reference = _build_feature_matrix(reference_rows, feature_keys)
    X_candidate = _build_feature_matrix(candidate_rows, feature_keys)

    n_balanced = min(n_reference, n_candidate)
    ref_idx = rng.choice(n_reference, size=n_balanced, replace=False) if n_reference > n_balanced else np.arange(n_reference)
    cand_idx = rng.choice(n_candidate, size=n_balanced, replace=False) if n_candidate > n_balanced else np.arange(n_candidate)

    X_balanced = np.vstack([X_reference[ref_idx], X_candidate[cand_idx]])
    y_balanced = np.concatenate([np.zeros(n_balanced), np.ones(n_balanced)])

    n_splits_eff = max(2, min(n_splits, n_balanced))
    folds = _stratified_kfold_indices(y_balanced, n_splits_eff, rng)

    fold_aucs: List[float] = []
    fold_accuracies: List[float] = []
    # Out-of-fold probability for every row of the balanced set (filled
    # exactly once each, since the folds partition `range(2 * n_balanced)`).
    # Needed below so a candidate row that was drawn into the balanced
    # training subsample is scored with its held-out CV prediction, never
    # with the final model that was fit on that very row.
    oof_proba = np.full(2 * n_balanced, np.nan)
    for k in range(n_splits_eff):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_splits_eff) if j != k])
        X_train, X_test = _standardize(X_balanced[train_idx], X_balanced[test_idx])
        y_train, y_test = y_balanced[train_idx], y_balanced[test_idx]

        weights = fit_logistic_regression(X_train, y_train, l2=l2, n_iter=n_iter)
        proba_test = predict_proba(X_test, weights)
        oof_proba[test_idx] = proba_test

        fold_aucs.append(compute_auc(y_test, proba_test))
        fold_accuracies.append(float(np.mean((proba_test > 0.5).astype(np.float64) == y_test)))

    # One final model on the whole balanced set: feature coefficients for the
    # report, and probabilities for every candidate row that never touched
    # training (not just the balanced subsample), so a larger candidate set
    # is still fully covered by the top-N ranking.
    X_balanced_std, X_candidate_std = _standardize(X_balanced, X_candidate)
    final_weights = fit_logistic_regression(X_balanced_std, y_balanced, l2=l2, n_iter=n_iter)

    coefficients = sorted(
        (
            {"feature": key, "coefficient": float(weight)}
            for key, weight in zip(feature_keys, final_weights[1:])
        ),
        key=lambda item: -abs(item["coefficient"]),
    )

    # Default: the final model's prediction, which is a genuine holdout
    # score for any candidate row never drawn into the balanced training
    # subsample (`cand_idx`). For rows that WERE drawn in, the final model
    # was fit on them, so that score is in-sample and inflated -- overwrite
    # those with their cross-validation out-of-fold probability instead,
    # which is a proper held-out estimate for exactly those rows.
    candidate_proba = predict_proba(X_candidate_std, final_weights)
    score_source = np.full(n_candidate, "final_model_holdout", dtype=object)
    candidate_oof = oof_proba[n_balanced:]  # same order as `cand_idx`
    for position, original_index in enumerate(cand_idx):
        candidate_proba[original_index] = candidate_oof[position]
        score_source[original_index] = "cv_out_of_fold"

    ranked = sorted(
        (
            {
                "path": row.get("path"),
                "probability": round(float(proba), 6),
                "index": index,
                "score_source": source,
            }
            for index, (row, proba, source) in enumerate(zip(candidate_rows, candidate_proba, score_source))
        ),
        key=lambda item: -item["probability"],
    )

    return {
        "n_reference": n_reference,
        "n_candidate": n_candidate,
        "n_balanced_per_class": n_balanced,
        "n_splits": n_splits_eff,
        "feature_keys": feature_keys,
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "fold_aucs": [float(v) for v in fold_aucs],
        "fold_accuracies": [float(v) for v in fold_accuracies],
        "coefficients": coefficients,
        "top_candidates": ranked[: max(0, top_n)],
    }


def load_rows(path: Union[str, Path]) -> Tuple[List[Dict[str, Any]], str]:
    """Load `{"rows": [...]}` from a `distribution measure --save-rows` JSON file.

    Raises `ValueError` with an actionable message if the file has no
    (non-empty) `rows` list -- the common cause is measuring without
    `--save-rows`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("rows")
    if not rows:
        raise ValueError(
            f"{path} has no per-image rows. Re-run `distribution measure --save-rows` "
            "to include them."
        )
    return rows, str(data.get("source") or path)


def format_discriminator_markdown(
    result: Dict[str, Any], reference_source: str, candidate_source: str
) -> str:
    lines = [
        f"# Real vs synthetic discriminator: `{candidate_source}` vs `{reference_source}`",
        "",
        "AUC 0.5 = the two sets are indistinguishable on these statistics; "
        "AUC 1.0 = perfectly separable.",
        "",
        f"- Reference: {result['n_reference']} images",
        f"- Candidate: {result['n_candidate']} images",
        f"- Balanced per class for training: {result['n_balanced_per_class']} "
        "(the larger side is subsampled down to the smaller)",
        f"- Folds: {result['n_splits']}",
        "",
        f"**Mean AUC: {result['auc_mean']:.3f} +/- {result['auc_std']:.3f}**  "
        f"Mean accuracy: {result['accuracy_mean']:.3f} +/- {result['accuracy_std']:.3f}",
        "",
        "| Fold | AUC | Accuracy |",
        "|---:|---:|---:|",
    ]
    for i, (auc, acc) in enumerate(zip(result["fold_aucs"], result["fold_accuracies"]), start=1):
        lines.append(f"| {i} | {auc:.3f} | {acc:.3f} |")

    lines += [
        "",
        "## Top separating features",
        "",
        "Coefficients of the final logistic regression on standardized "
        "features (positive = pushes towards \"candidate/synthetic\").",
        "",
        "| Feature | Coefficient |",
        "|---|---:|",
    ]
    for item in result["coefficients"][:15]:
        lines.append(f"| {item['feature']} | {item['coefficient']:.3f} |")

    lines += [
        "",
        f"## Top {len(result['top_candidates'])} candidate images most confidently synthetic",
        "",
        "Every probability is a held-out prediction (`Source` column): "
        "`cv_out_of_fold` for a row drawn into the balanced training "
        "subsample (scored from the fold where it was held out), "
        "`final_model_holdout` for a row never used in training.",
        "",
        "| Rank | Probability | Source | Path |",
        "|---:|---:|---|---|",
    ]
    for rank, item in enumerate(result["top_candidates"], start=1):
        path = item["path"] if item["path"] is not None else f"(row {item['index']}, no path saved)"
        lines.append(f"| {rank} | {item['probability']:.3f} | {item['score_source']} | {path} |")

    lines.append("")
    return "\n".join(lines) + "\n"
