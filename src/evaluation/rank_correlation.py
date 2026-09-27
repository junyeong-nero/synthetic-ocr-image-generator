"""Correlate model rankings from this repo's synthetic evaluation outputs
against an external "real" benchmark ranking.

Synthetic scores are read from the existing evaluation output formats
(`evaluation_result/leaderboard.json`, a single `report.json`, or a directory
scanned recursively for `report.json` files -- see `src/evaluation/leaderboard.py`
for the canonical parsing of those files, reused here). Real scores are a
CSV or JSON mapping of model name -> score supplied by the caller.

Models are matched by a normalized id (org prefix / path stripped, case and
punctuation folded) with an optional alias file for names that do not
normalize to the same string on both sides.
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats

from src.evaluation.leaderboard import _collect_latest_rows, _load_json

DEFAULT_METRIC = "avg_markdown_overall_score"
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 12345

_MODEL_KEYS = ("model", "model_id", "name")
_SCORE_KEYS = ("score", "real_score", "value")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def normalize_model_id(name: str) -> str:
    """Fold a model identifier to a comparable key: strip any org/path
    prefix (text before the final "/"), lowercase, and drop punctuation."""
    text = str(name or "").strip().replace("\\", "/")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def load_aliases(path: Path | None) -> dict[str, str]:
    """Load a JSON/YAML object mapping alternate model names to a shared
    canonical name. Both sides are normalized; either the real or the
    synthetic name may be used as the alias key, matching whichever side
    does not already normalize the same as its counterpart."""
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Alias file not found: {path}")

    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Alias file must contain a JSON/YAML object: {path}")

    return {normalize_model_id(key): normalize_model_id(value) for key, value in raw.items()}


def _match_key(name: str, aliases: dict[str, str]) -> str:
    normalized = normalize_model_id(name)
    return aliases.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Real score loading
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _pick_csv_columns(fieldnames: list[str]) -> tuple[str, str]:
    lowered = {name.lower().strip(): name for name in fieldnames}
    model_col = next((lowered[key] for key in _MODEL_KEYS if key in lowered), None)
    score_col = next((lowered[key] for key in _SCORE_KEYS if key in lowered), None)
    if model_col and score_col:
        return model_col, score_col
    if len(fieldnames) >= 2:
        return fieldnames[0], fieldnames[1]
    raise ValueError(f"CSV must have at least two columns (model, score); got: {fieldnames}")


def _load_real_scores_csv(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return scores
        model_col, score_col = _pick_csv_columns(list(reader.fieldnames))
        for row in reader:
            name = str(row.get(model_col) or "").strip()
            value = _to_float(row.get(score_col))
            if name and value is not None:
                scores[name] = value
    return scores


def _load_real_scores_mapping(payload: Any) -> dict[str, float]:
    scores: dict[str, float] = {}
    if isinstance(payload, dict):
        for name, value in payload.items():
            numeric = _to_float(value)
            if numeric is not None:
                scores[str(name)] = numeric
    elif isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            name = next((entry[key] for key in _MODEL_KEYS if key in entry), None)
            value = next((entry[key] for key in _SCORE_KEYS if key in entry), None)
            numeric = _to_float(value)
            if name and numeric is not None:
                scores[str(name)] = numeric
    else:
        raise ValueError("Real score JSON/YAML must be an object or a list of objects")
    return scores


def load_real_scores(path: Path) -> dict[str, float]:
    """Load model -> score from a CSV (model,score columns, matched by name
    or by position) or JSON/YAML (object of name->score, or a list of
    objects with model/score-like keys)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Real score file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        scores = _load_real_scores_csv(path)
    elif suffix == ".json":
        scores = _load_real_scores_mapping(json.loads(path.read_text(encoding="utf-8")))
    elif suffix in {".yaml", ".yml"}:
        scores = _load_real_scores_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    else:
        raise ValueError(f"Unsupported real score file extension: {path.suffix}")

    if not scores:
        raise ValueError(f"No model scores parsed from {path}")
    return scores


# ---------------------------------------------------------------------------
# Synthetic score loading (this repo's evaluation_result formats)
# ---------------------------------------------------------------------------


def _model_id_from_report(payload: dict[str, Any], source_path: Path) -> str:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    model_id = str(model_cfg.get("model_id") or summary.get("model_id") or "").strip()
    return model_id or source_path.parent.name


def _scores_from_leaderboard_payload(
    payload: dict[str, Any], metric: str, language: str | None
) -> dict[str, float]:
    languages_block = payload.get("languages") or []
    available = sorted(
        {str(block.get("language")) for block in languages_block if isinstance(block, dict)}
    )

    if language is not None:
        blocks = [
            block
            for block in languages_block
            if isinstance(block, dict) and str(block.get("language")) == language
        ]
        if not blocks:
            raise ValueError(f"Language {language!r} not found in leaderboard; available: {available}")
    elif len(available) == 1:
        blocks = languages_block
    else:
        raise ValueError(
            f"Leaderboard has multiple languages ({available}); pass --language to select one."
        )

    scores: dict[str, float] = {}
    for block in blocks:
        rows = block.get("rows") if isinstance(block.get("rows"), list) else []
        for row in rows:
            model_id = row.get("model_id")
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            value = _to_float(metrics.get(metric))
            if model_id and value is not None:
                scores[str(model_id)] = value
    return scores


def load_synthetic_scores(
    source: Path,
    metric: str = DEFAULT_METRIC,
    language: str | None = None,
) -> dict[str, float]:
    """Load model -> `metric` value from an evaluation_result leaderboard.json,
    a single report.json, or a directory scanned recursively for report.json
    files (the same "latest per model+language" logic as `leaderboard.py`)."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Synthetic score source not found: {source}")

    if source.is_dir():
        rows = _collect_latest_rows(source)
        if language is not None:
            rows = [row for row in rows if str(row.get("language")) == language]
        scores = {}
        for row in rows:
            model_id = row.get("model_id")
            value = _to_float((row.get("metrics") or {}).get(metric))
            if model_id and value is not None:
                scores[str(model_id)] = value
    else:
        payload = _load_json(source)
        if isinstance(payload.get("languages"), list):
            scores = _scores_from_leaderboard_payload(payload, metric, language)
        elif "config" in payload and "metrics" in payload:
            model_id = _model_id_from_report(payload, source)
            value = _to_float((payload.get("metrics") or {}).get(metric))
            scores = {model_id: value} if value is not None else {}
        else:
            raise ValueError(
                f"Unrecognized synthetic score source format: {source}. Expected a "
                "leaderboard.json (with a 'languages' list), a report.json (with "
                "'config'/'metrics'), or a directory containing report.json files."
            )

    if not scores:
        language_note = f" for language={language!r}" if language else ""
        raise ValueError(f"No models found with metric {metric!r} in {source}{language_note}")
    return scores


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


@dataclass
class ModelRankRow:
    match_key: str
    synthetic_model_id: str
    real_model_name: str
    synthetic_score: float
    real_score: float
    synthetic_rank: float
    real_rank: float


@dataclass
class RankCorrelationResult:
    metric: str
    n: int
    spearman_rho: float
    spearman_pvalue: float
    spearman_ci: tuple[float, float]
    kendall_tau: float
    kendall_pvalue: float
    kendall_ci: tuple[float, float]
    bootstrap_iterations: int
    confidence: float
    rows: list[ModelRankRow]
    unmatched_synthetic: list[str]
    unmatched_real: list[str]


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    statistic,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    n = len(x)
    if n < 2 or n_bootstrap <= 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            value = statistic(x[idx], y[idx])
            if value is not None and not np.isnan(value):
                samples.append(float(value))

    if len(samples) < 2:
        return (float("nan"), float("nan"))

    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha)))


def _spearman_statistic(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = stats.spearmanr(a, b)
    return rho


def _kendall_statistic(a: np.ndarray, b: np.ndarray) -> float:
    tau, _ = stats.kendalltau(a, b)
    return tau


def compute_rank_correlation(
    synthetic_scores: dict[str, float],
    real_scores: dict[str, float],
    *,
    aliases: dict[str, str] | None = None,
    metric: str = DEFAULT_METRIC,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> RankCorrelationResult:
    """Match synthetic and real models by normalized id, then compute
    Spearman rho and Kendall tau (with a bootstrap CI over models) between
    the two score sets."""
    aliases = aliases or {}

    synthetic_keys: dict[str, str] = {}
    for model_id in synthetic_scores:
        key = _match_key(model_id, aliases)
        if key in synthetic_keys and synthetic_keys[key] != model_id:
            raise ValueError(
                f"Ambiguous synthetic model match key {key!r}: "
                f"{synthetic_keys[key]!r} and {model_id!r} both normalize to it. "
                "Use --aliases to disambiguate."
            )
        synthetic_keys[key] = model_id

    real_keys: dict[str, str] = {}
    for name in real_scores:
        key = _match_key(name, aliases)
        if key in real_keys and real_keys[key] != name:
            raise ValueError(
                f"Ambiguous real model match key {key!r}: "
                f"{real_keys[key]!r} and {name!r} both normalize to it. "
                "Use --aliases to disambiguate."
            )
        real_keys[key] = name

    matched_keys = sorted(set(synthetic_keys) & set(real_keys))
    unmatched_synthetic = sorted(
        model_id for key, model_id in synthetic_keys.items() if key not in real_keys
    )
    unmatched_real = sorted(name for key, name in real_keys.items() if key not in synthetic_keys)

    if len(matched_keys) < 2:
        raise ValueError(
            f"Need at least 2 matched models to compute rank correlation; got {len(matched_keys)}. "
            f"Unmatched synthetic: {unmatched_synthetic}; unmatched real: {unmatched_real}."
        )

    synthetic_values = np.array([synthetic_scores[synthetic_keys[k]] for k in matched_keys], dtype=float)
    real_values = np.array([real_scores[real_keys[k]] for k in matched_keys], dtype=float)

    spearman_rho, spearman_p = stats.spearmanr(synthetic_values, real_values)
    kendall_tau, kendall_p = stats.kendalltau(synthetic_values, real_values)

    spearman_ci = _bootstrap_ci(
        synthetic_values, real_values, _spearman_statistic, n_bootstrap, confidence, seed
    )
    kendall_ci = _bootstrap_ci(
        synthetic_values, real_values, _kendall_statistic, n_bootstrap, confidence, seed
    )

    # Rank 1 = best (highest score). Ties get the average rank.
    synthetic_ranks = stats.rankdata(-synthetic_values, method="average")
    real_ranks = stats.rankdata(-real_values, method="average")

    rows = [
        ModelRankRow(
            match_key=key,
            synthetic_model_id=synthetic_keys[key],
            real_model_name=real_keys[key],
            synthetic_score=float(synthetic_values[i]),
            real_score=float(real_values[i]),
            synthetic_rank=float(synthetic_ranks[i]),
            real_rank=float(real_ranks[i]),
        )
        for i, key in enumerate(matched_keys)
    ]
    rows.sort(key=lambda row: (row.real_rank, row.match_key))

    return RankCorrelationResult(
        metric=metric,
        n=len(matched_keys),
        spearman_rho=float(spearman_rho),
        spearman_pvalue=float(spearman_p),
        spearman_ci=spearman_ci,
        kendall_tau=float(kendall_tau),
        kendall_pvalue=float(kendall_p),
        kendall_ci=kendall_ci,
        bootstrap_iterations=n_bootstrap,
        confidence=confidence,
        rows=rows,
        unmatched_synthetic=unmatched_synthetic,
        unmatched_real=unmatched_real,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown_report(
    result: RankCorrelationResult,
    *,
    synthetic_source: str | None = None,
    real_source: str | None = None,
) -> str:
    lines = ["# Synthetic vs Real Rank Correlation", "", f"- Metric: `{result.metric}`"]
    if synthetic_source:
        lines.append(f"- Synthetic source: `{synthetic_source}`")
    if real_source:
        lines.append(f"- Real source: `{real_source}`")

    lines.extend(
        [
            f"- Matched models: {result.n} "
            f"({len(result.unmatched_synthetic)} synthetic-only, {len(result.unmatched_real)} real-only unmatched)",
            f"- Spearman rho: {_fmt(result.spearman_rho)} "
            f"(95% CI [{_fmt(result.spearman_ci[0])}, {_fmt(result.spearman_ci[1])}], "
            f"p={_fmt(result.spearman_pvalue)})",
            f"- Kendall tau: {_fmt(result.kendall_tau)} "
            f"(95% CI [{_fmt(result.kendall_ci[0])}, {_fmt(result.kendall_ci[1])}], "
            f"p={_fmt(result.kendall_pvalue)})",
            f"- Bootstrap: {result.bootstrap_iterations} resamples, {int(round(result.confidence * 100))}% CI",
            "",
            "## Per-model Ranks",
            "",
            "| Real Rank | Synthetic Model | Real Model | Synthetic Score | Real Score | Synthetic Rank | Rank Diff |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )

    for row in result.rows:
        rank_diff = row.real_rank - row.synthetic_rank
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {:+g} |".format(
                f"{row.real_rank:g}",
                row.synthetic_model_id,
                row.real_model_name,
                _fmt(row.synthetic_score),
                _fmt(row.real_score),
                f"{row.synthetic_rank:g}",
                rank_diff,
            )
        )
    lines.append("")

    if result.unmatched_synthetic:
        lines.append("## Unmatched (synthetic only)")
        lines.append("")
        lines.extend(f"- {name}" for name in result.unmatched_synthetic)
        lines.append("")

    if result.unmatched_real:
        lines.append("## Unmatched (real only)")
        lines.append("")
        lines.extend(f"- {name}" for name in result.unmatched_real)
        lines.append("")

    return "\n".join(lines)
