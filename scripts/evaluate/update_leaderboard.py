#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


MAIN_METRIC_PREFERENCES: dict[str, list[str]] = {
    "sentence": ["avg_cer", "avg_wer"],
    "markdown": ["avg_cer", "exact_match_rate", "normalized_match_rate"],
    "table": ["avg_teds", "avg_cell_accuracy", "avg_structure_f1"],
    "document": ["avg_overall_f1", "avg_layout_f1", "avg_reading_order", "avg_kv_f1"],
    "kie": ["avg_entity_f1", "entity_f1", "overall_f1", "line_item_f1", "avg_cer"],
}

LOWER_BETTER_METRICS = {"avg_cer", "avg_wer"}


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_metric_value(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is not None:
        return f"{numeric:.4f}"
    if value is None:
        return "-"
    return str(value)


def _parse_timestamp(candidate: str | None) -> datetime:
    if not candidate:
        return datetime.min
    normalized = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def _pick_metric(subset: str, rows: list[dict[str, Any]]) -> str | None:
    available = {
        metric_key
        for row in rows
        for metric_key, metric_value in row.get("metrics", {}).items()
        if _to_float(metric_value) is not None
    }

    for preferred in MAIN_METRIC_PREFERENCES.get(subset, []):
        if preferred in available:
            return preferred

    average_candidates = sorted(k for k in available if k.startswith("avg_"))
    if average_candidates:
        return average_candidates[0]

    if not available:
        return None
    return sorted(available)[0]


def _collect_latest_rows(base_dir: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}

    for report_path in sorted(base_dir.glob("*/**/report.json")):
        report = _load_json(report_path)
        config = report.get("config", {}) if isinstance(report.get("config"), dict) else {}
        summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}

        subset = str(config.get("subset") or report_path.parent.name)
        model_id = str(model_cfg.get("model_id") or report_path.parent.parent.name)

        protocol_path = report_path.parent / "protocol.json"
        protocol = _load_json(protocol_path) if protocol_path.exists() else {}
        timestamp = _parse_timestamp(str(protocol.get("timestamp")) if protocol else None)
        if timestamp == datetime.min:
            timestamp = datetime.fromtimestamp(report_path.stat().st_mtime)

        row = {
            "model_id": model_id,
            "backend": model_cfg.get("backend"),
            "subset": subset,
            "dataset": config.get("dataset_id"),
            "split": config.get("split"),
            "total_samples": summary.get("total_samples"),
            "successful": summary.get("successful"),
            "failed": summary.get("failed"),
            "avg_latency_ms": summary.get("avg_latency_ms"),
            "metrics": metrics,
            "timestamp": timestamp,
            "report_path": str(report_path),
        }

        key = (model_id, subset)
        previous = latest.get(key)
        if previous is None or row["timestamp"] >= previous["timestamp"]:
            latest[key] = row

    return list(latest.values())


def _sort_rows(rows: list[dict[str, Any]], metric_key: str | None) -> list[dict[str, Any]]:
    if not metric_key:
        return sorted(rows, key=lambda row: str(row.get("model_id") or ""))

    lower_is_better = metric_key in LOWER_BETTER_METRICS

    def key_fn(row: dict[str, Any]) -> tuple[int, float, str]:
        value = _to_float(row.get("metrics", {}).get(metric_key))
        if value is None:
            return (1, float("inf"), str(row.get("model_id") or ""))
        sort_value = value if lower_is_better else -value
        return (0, sort_value, str(row.get("model_id") or ""))

    return sorted(rows, key=key_fn)


def _write_subset_outputs(output_dir: Path, subset: str, rows: list[dict[str, Any]]) -> None:
    metric_key = _pick_metric(subset, rows)
    sorted_rows = _sort_rows(rows, metric_key)

    payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "subset": subset,
        "metric_key": metric_key,
        "metric_direction": "ascending" if metric_key in LOWER_BETTER_METRICS else "descending",
        "count": len(sorted_rows),
        "rows": [
            {
                **row,
                "timestamp": row["timestamp"].isoformat(),
            }
            for row in sorted_rows
        ],
    }

    json_path = output_dir / f"leaderboard_{subset}.json"
    md_path = output_dir / f"leaderboard_{subset}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Leaderboard ({subset})",
        "",
        f"- Sorted by: `{metric_key or 'N/A'}` ({'lower is better' if metric_key in LOWER_BETTER_METRICS else 'higher is better'})",
        f"- Entries: {len(sorted_rows)}",
        "",
        "| Rank | Model | Backend | Metric | Dataset | Split | Success/Total | Updated (UTC) |",
        "|---:|---|---|---:|---|---|---|---|",
    ]

    for rank, row in enumerate(sorted_rows, start=1):
        metric_value = None
        if metric_key:
            metric_value = row.get("metrics", {}).get(metric_key)
        success = row.get("successful")
        total = row.get("total_samples")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {}/{} | {} |".format(
                rank,
                row.get("model_id") or "-",
                row.get("backend") or "-",
                _format_metric_value(metric_value),
                row.get("dataset") or "-",
                row.get("split") or "-",
                success if isinstance(success, int) else "-",
                total if isinstance(total, int) else "-",
                row["timestamp"].isoformat(),
            )
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def update_leaderboards(base_dir: Path) -> None:
    if not base_dir.exists() or not base_dir.is_dir():
        raise NotADirectoryError(f"Invalid evaluation directory: {base_dir}")

    rows = _collect_latest_rows(base_dir)
    if not rows:
        raise FileNotFoundError(f"No report.json files found under: {base_dir}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subset") or "unknown")].append(row)

    for subset, subset_rows in grouped.items():
        _write_subset_outputs(base_dir, subset, subset_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-subset leaderboards from evaluation_result.")
    parser.add_argument(
        "--base-dir",
        default="evaluation_result",
        help="Base directory containing model/subset evaluation outputs",
    )
    args = parser.parse_args()
    update_leaderboards(Path(args.base_dir))


if __name__ == "__main__":
    main()
