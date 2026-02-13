#!/usr/bin/env python3
"""Summarize evaluation report.json files into JSON and Markdown tables.

Usage:
    uv run summary_test_results.py test_results/20260210_111146
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAIN_METRIC_PREFERENCES: dict[str, list[str]] = {
    "sentence": ["avg_cer", "avg_wer"],
    "markdown": ["normalized_match_rate", "exact_match_rate", "avg_cer"],
    "table": ["avg_teds", "avg_cell_accuracy", "avg_structure_f1"],
    "document": ["avg_text_table_formula_score", "avg_text_table_score", "avg_text_score", "avg_table_teds", "avg_overall_f1", "avg_layout_f1", "avg_reading_order", "avg_kv_f1"],
    "kie": ["avg_entity_f1", "avg_overall_f1", "overall_f1", "entity_f1", "line_item_f1"],
}


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_value(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is not None:
        return f"{numeric:.4f}"
    if value is None:
        return "-"
    return str(value)


def _filter_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop unsupported extrema metrics from summary outputs."""
    filtered: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("min_") or key.startswith("max_"):
            continue
        filtered[key] = value
    return filtered


def _metric_display_map(metrics: dict[str, Any]) -> dict[str, str]:
    """Build display-ready metrics with avg/std compact formatting."""
    display: dict[str, str] = {}

    for key in sorted(metrics):
        if key.startswith("std_"):
            continue

        if key.startswith("avg_"):
            base_key = key[4:]
            avg_value = _to_float(metrics.get(key))
            std_value = _to_float(metrics.get(f"std_{base_key}"))
            if avg_value is not None and std_value is not None:
                display[base_key] = f"{avg_value:.4f} ({std_value:.4f})"
            else:
                display[base_key] = _format_value(metrics.get(key))
            continue

        display[key] = _format_value(metrics.get(key))

    return display


def _pick_main_metric(subset: str, rows: list[dict[str, Any]]) -> str | None:
    available = {
        key
        for row in rows
        for key, value in row.get("metrics", {}).items()
        if _to_float(value) is not None
    }

    for preferred_key in MAIN_METRIC_PREFERENCES.get(subset, []):
        if preferred_key in available:
            return preferred_key

    avg_candidates = sorted(k for k in available if k.startswith("avg_"))
    if avg_candidates:
        return avg_candidates[0]

    if not available:
        return None
    return sorted(available)[0]


def _metric_sort_value(row: dict[str, Any], metric_key: str | None) -> float:
    if not metric_key:
        return float("inf")
    value = row.get("metrics", {}).get(metric_key)
    numeric = _to_float(value)
    return numeric if numeric is not None else float("inf")


def _load_row(report_path: Path) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    config = payload.get("config", {})
    summary = payload.get("summary", {})
    model_cfg = config.get("model", {})

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    filtered_metrics = _filter_metrics(metrics)

    return {
        "model": report_path.parent.name,
        "model_id": model_cfg.get("model_id", "N/A"),
        "backend": model_cfg.get("backend", "N/A"),
        "subset": config.get("subset", "N/A"),
        "format": config.get("format", "N/A"),
        "dataset": config.get("dataset_id", "N/A"),
        "split": config.get("split", "N/A"),
        "total_samples": summary.get("total_samples", 0),
        "successful": summary.get("successful", 0),
        "failed": summary.get("failed", 0),
        "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
        "metrics": filtered_metrics,
        "metrics_display": _metric_display_map(filtered_metrics),
        "report_path": str(report_path),
    }


def _collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = [_load_row(path) for path in sorted(results_dir.rglob("report.json"))]
    if not rows:
        raise FileNotFoundError(f"No report.json found under: {results_dir}")
    return rows


def _collect_metric_keys(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    keys_by_subset: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subset = str(row.get("subset", "N/A"))
        metrics = row.get("metrics_display", {})
        if isinstance(metrics, dict):
            for key in metrics:
                keys_by_subset[subset].add(key)
    return {subset: sorted(keys) for subset, keys in keys_by_subset.items()}


def _build_markdown(rows: list[dict[str, Any]], metric_keys_by_subset: dict[str, list[str]]) -> str:
    lines: list[str] = [
        "# Test Results Summary",
        "",
        f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Total reports: {len(rows)}",
        f"- Total models: {len({row['model'] for row in rows})}",
        "",
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subset", "N/A"))].append(row)

    for subset in sorted(grouped):
        main_metric = _pick_main_metric(subset, grouped[subset])
        subset_rows = sorted(
            grouped[subset],
            key=lambda r: (
                _metric_sort_value(r, main_metric),
                str(r["model"]),
            ),
        )
        metric_keys = metric_keys_by_subset.get(subset, [])

        headers = [
            "Model",
            "Model ID",
            "Format",
            "Dataset",
            "Split",
            "Total",
            "Success",
            "Failed",
            "Avg Latency (ms)",
            *metric_keys,
        ]

        lines.extend(
            [
                f"## Subset: {subset}",
                "",
                f"- Sorted by: `{main_metric or 'N/A'}` (ascending)",
                "",
                "| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] * len(headers)) + "|",
            ]
        )

        for row in subset_rows:
            values = [
                str(row.get("model", "-")),
                str(row.get("model_id", "-")),
                str(row.get("format", "-")),
                str(row.get("dataset", "-")),
                str(row.get("split", "-")),
                str(row.get("total_samples", 0)),
                str(row.get("successful", 0)),
                str(row.get("failed", 0)),
                _format_value(row.get("avg_latency_ms", 0.0)),
            ]

            metrics = row.get("metrics_display", {})
            for key in metric_keys:
                metric_value = metrics.get(key) if isinstance(metrics, dict) else None
                values.append(str(metric_value) if metric_value is not None else "-")

            lines.append("| " + " | ".join(values) + " |")

        lines.append("")

    return "\n".join(lines)


def summarize(results_dir: Path, output_json: Path, output_md: Path) -> tuple[Path, Path]:
    rows = _collect_rows(results_dir)
    metric_keys_by_subset = _collect_metric_keys(rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(results_dir),
        "total_reports": len(rows),
        "total_models": len({row["model"] for row in rows}),
        "metric_keys_by_subset": metric_keys_by_subset,
        "rows": rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_md.write_text(_build_markdown(rows, metric_keys_by_subset), encoding="utf-8")

    return output_json, output_md


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize test result report.json files by model and subset."
    )
    parser.add_argument("results_dir", help="Path to test results directory")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON path (default: <results_dir>/summary_metrics.json)",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Output Markdown path (default: <results_dir>/summary_metrics.md)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        raise NotADirectoryError(f"Invalid results directory: {results_dir}")

    output_json = (
        Path(args.output_json)
        if args.output_json
        else results_dir / "summary_metrics.json"
    )
    output_md = (
        Path(args.output_md) if args.output_md else results_dir / "summary_metrics.md"
    )

    json_path, md_path = summarize(results_dir, output_json, output_md)
    print(f"Saved JSON summary: {json_path}")
    print(f"Saved Markdown summary: {md_path}")


if __name__ == "__main__":
    main()
