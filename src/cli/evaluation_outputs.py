import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PROTOCOL_VERSION = "1.0"


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_metric(metric_key: Optional[str], metric_value: Optional[float]) -> Optional[float]:
    if metric_key is None or metric_value is None:
        return None
    lower_better = {"avg_cer", "avg_wer"}
    normalized = 1.0 - metric_value if metric_key in lower_better else metric_value
    return max(0.0, min(1.0, normalized))


def _safe_numeric(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def write_protocol_snapshot(
    output_dir: Path,
    output: Any,
    report_format: str,
) -> Path:
    snapshot = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": _iso_timestamp(),
        "command": "evaluate",
        "report_format": report_format,
        "config": output.config,
        "summary": output.summary,
        "prompt": output.config.get("prompt"),
        "prompt_source": output.config.get("prompt_source"),
        "system_prompt": output.config.get("system_prompt"),
    }
    path = output_dir / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return path


def write_leaderboard(output_dir: Path, summary_entries: list[dict[str, Any]]) -> None:
    leaderboard_entries = []
    for entry in summary_entries:
        metric_key = entry.get("metric_key")
        metric_value = entry.get("metric_value")
        normalized_average = _normalize_metric(metric_key, metric_value)

        entry_metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
        leaderboard_entries.append(
            {
                "timestamp": entry.get("timestamp"),
                "protocol_version": entry.get("protocol_version"),
                "model_id": entry.get("model_id"),
                "backend": entry.get("backend"),
                "dataset": entry.get("dataset"),
                "split": entry.get("split"),
                "language": entry.get("language") or "unknown",
                "average_score": entry.get("average_score", metric_value),
                "normalized_average_score": normalized_average,
                "average_empty_rate": entry.get("average_empty_rate", entry.get("empty_rate")),
                "average_parse_fail_rate": entry.get(
                    "average_parse_fail_rate", entry.get("parse_fail_rate")
                ),
                "markdown_text_score": _safe_numeric(entry_metrics.get("avg_markdown_text_score")),
                "markdown_table_teds": _safe_numeric(entry_metrics.get("avg_markdown_table_teds")),
                "markdown_formula_score": _safe_numeric(entry_metrics.get("avg_markdown_formula_score")),
                "markdown_order_score": _safe_numeric(entry_metrics.get("avg_markdown_order_score")),
            }
        )

    leaderboard_entries.sort(
        key=lambda item: (
            item.get("language") or "",
            item.get("normalized_average_score") is None,
            -(item.get("normalized_average_score") or 0),
        ),
    )

    leaderboard_path = output_dir / "leaderboard.json"
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard_entries, f, ensure_ascii=False, indent=2)

    by_language: dict[str, list[dict[str, Any]]] = {}
    for entry in leaderboard_entries:
        language = str(entry.get("language") or "unknown")
        if language not in by_language:
            by_language[language] = []
        by_language[language].append(entry)

    lines = ["# OCR Benchmark Leaderboard", ""]
    for language in sorted(by_language.keys()):
        lines.extend(
            [
                f"## {language}",
                "",
                "| Rank | Model | Backend | Dataset | Split | Normalized | Raw | Text | Table | Formula | Order | Empty Rate | Parse Fail Rate |",
                "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for idx, entry in enumerate(by_language[language], start=1):
            normalized_score = entry.get("normalized_average_score")
            raw_score = entry.get("average_score")
            empty_rate = entry.get("average_empty_rate")
            parse_fail_rate = entry.get("average_parse_fail_rate")
            text_score = entry.get("markdown_text_score")
            table_score = entry.get("markdown_table_teds")
            formula_score = entry.get("markdown_formula_score")
            order_score = entry.get("markdown_order_score")
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    idx,
                    entry.get("model_id") or "-",
                    entry.get("backend") or "-",
                    entry.get("dataset") or "-",
                    entry.get("split") or "-",
                    f"{normalized_score:.4f}" if isinstance(normalized_score, (int, float)) else "-",
                    f"{raw_score:.4f}" if isinstance(raw_score, (int, float)) else "-",
                    f"{text_score:.4f}" if isinstance(text_score, (int, float)) else "-",
                    f"{table_score:.4f}" if isinstance(table_score, (int, float)) else "-",
                    f"{formula_score:.4f}" if isinstance(formula_score, (int, float)) else "-",
                    f"{order_score:.4f}" if isinstance(order_score, (int, float)) else "-",
                    f"{empty_rate:.4f}" if isinstance(empty_rate, (int, float)) else "-",
                    f"{parse_fail_rate:.4f}" if isinstance(parse_fail_rate, (int, float)) else "-",
                )
            )

        lines.append("")

    leaderboard_md = output_dir / "leaderboard.md"
    with open(leaderboard_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_summary_entry(
    *,
    output: Any,
    model_id: str,
    backend: str,
    dataset: str,
    split: str,
    language: str,
    seed: Optional[int],
    resolved_format: str,
    metric_key: Optional[str],
    metric_value: Optional[float],
) -> dict[str, Any]:
    return {
        "timestamp": _iso_timestamp(),
        "protocol_version": PROTOCOL_VERSION,
        "model_id": model_id,
        "backend": backend,
        "dataset": dataset,
        "split": split,
        "language": language,
        "seed": seed,
        "format": resolved_format,
        "metric_key": metric_key,
        "metric_value": metric_value,
        "average_score": metric_value,
        "empty_rate": output.summary.get("empty_rate"),
        "parse_fail_rate": output.summary.get("parse_fail_rate"),
        "total_samples": output.summary.get("total_samples"),
        "prompt_source": output.config.get("prompt_source"),
        "metrics": {
            key: float(value)
            for key, value in output.metrics.items()
            if isinstance(value, (int, float))
        },
    }


def load_summary_entries(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []

    try:
        with open(summary_path, encoding="utf-8") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        corrupt_path = summary_path.with_suffix(".corrupt.json")
        try:
            if summary_path.exists():
                summary_path.replace(corrupt_path)
        finally:
            raise RuntimeError(
                f"Corrupt summary file moved to {corrupt_path}"
            ) from exc

    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        return [loaded]
    return []


def save_summary_entries(summary_path: Path, entries: list[dict[str, Any]]) -> None:
    tmp_path = summary_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, summary_path)


def persist_evaluation_outputs(
    *,
    output_dir: Path,
    output: Any,
    report_format: str,
    model_id: str,
    backend: str,
    dataset: str,
    split: str,
    language: str,
    seed: Optional[int],
    resolved_format: str,
    metric_key: Optional[str],
    metric_value: Optional[float],
) -> tuple[Path, Path]:
    protocol_path = write_protocol_snapshot(output_dir, output, report_format)

    summary_entry = build_summary_entry(
        output=output,
        model_id=model_id,
        backend=backend,
        dataset=dataset,
        split=split,
        language=language,
        seed=seed,
        resolved_format=resolved_format,
        metric_key=metric_key,
        metric_value=metric_value,
    )

    summary_path = output_dir / "model_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_summary_entries(summary_path)
    existing.append(summary_entry)
    save_summary_entries(summary_path, existing)
    write_leaderboard(output_dir, existing)

    return protocol_path, summary_path
