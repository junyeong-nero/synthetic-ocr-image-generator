#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


MAIN_METRIC_PREFERENCES: dict[str, list[str]] = {
    "markdown": [
        "avg_markdown_overall_score",
        "avg_markdown_text_score",
        "avg_markdown_table_teds",
        "avg_markdown_formula_score",
        "avg_markdown_order_score",
    ],
}

LOWER_BETTER_METRICS = {"avg_cer", "avg_wer", "avg_formula_edit_distance"}


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


def _fallback_model_id_from_path(base_dir: Path, report_path: Path) -> str:
    try:
        relative = report_path.relative_to(base_dir)
        parts = list(relative.parts[:-1])
        if parts:
            return "/".join(parts)
    except ValueError:
        pass
    return report_path.parent.name


def _pick_metric(format_name: str, rows: list[dict[str, Any]]) -> str | None:
    available = {
        metric_key
        for row in rows
        for metric_key, metric_value in row.get("metrics", {}).items()
        if _to_float(metric_value) is not None
    }

    for preferred in MAIN_METRIC_PREFERENCES.get(format_name, []):
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

    for report_path in sorted(base_dir.rglob("report.json")):
        report = _load_json(report_path)
        config = report.get("config", {}) if isinstance(report.get("config"), dict) else {}
        summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}

        format_name = str(config.get("format") or "unknown")
        model_id = str(model_cfg.get("model_id") or "").strip()
        if not model_id:
            model_id = _fallback_model_id_from_path(base_dir, report_path)

        protocol_path = report_path.parent / "protocol.json"
        protocol = _load_json(protocol_path) if protocol_path.exists() else {}
        timestamp = _parse_timestamp(str(protocol.get("timestamp")) if protocol else None)
        if timestamp == datetime.min:
            timestamp = datetime.fromtimestamp(report_path.stat().st_mtime)

        row = {
            "model_id": model_id,
            "backend": model_cfg.get("backend"),
            "format": format_name,
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

        key = (model_id, format_name)
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


def _build_format_payload(format_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_key = _pick_metric(format_name, rows)
    sorted_rows = _sort_rows(rows, metric_key)

    return {
        "format": format_name,
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


def _write_leaderboard_outputs(output_dir: Path, formats: list[dict[str, Any]]) -> None:
    payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "count": len(formats),
        "formats": formats,
    }

    json_path = output_dir / "leaderboard.json"
    md_path = output_dir / "leaderboard.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Leaderboard", "", f"- Formats: {len(formats)}", ""]

    for format_payload in formats:
        format_name = str(format_payload.get("format") or "unknown")
        metric_key = str(format_payload.get("metric_key") or "") or None
        metric_note = "lower is better" if metric_key in LOWER_BETTER_METRICS else "higher is better"
        rows = format_payload.get("rows", [])
        include_markdown_columns = format_name == "markdown"

        lines.extend(
            [
                f"## {format_name}",
                "",
                f"- Sorted by: `{metric_key or 'N/A'}` ({metric_note})",
                f"- Entries: {len(rows)}",
                "",
                (
                    "| Rank | Model | Backend | Metric | Dataset | Split | Text | Table | Formula | Order | Success/Total | Updated (UTC) |"
                    if include_markdown_columns
                    else "| Rank | Model | Backend | Metric | Dataset | Split | Success/Total | Updated (UTC) |"
                ),
                (
                    "|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---|"
                    if include_markdown_columns
                    else "|---:|---|---|---:|---|---|---|---|"
                ),
            ]
        )

        for rank, row in enumerate(rows, start=1):
            metric_value = None
            if metric_key:
                metric_value = row.get("metrics", {}).get(metric_key)
            success = row.get("successful")
            total = row.get("total_samples")
            if include_markdown_columns:
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} |".format(
                        rank,
                        row.get("model_id") or "-",
                        row.get("backend") or "-",
                        _format_metric_value(metric_value),
                        row.get("dataset") or "-",
                        row.get("split") or "-",
                        _format_metric_value(row.get("metrics", {}).get("avg_markdown_text_score")),
                        _format_metric_value(row.get("metrics", {}).get("avg_markdown_table_teds")),
                        _format_metric_value(row.get("metrics", {}).get("avg_markdown_formula_score")),
                        _format_metric_value(row.get("metrics", {}).get("avg_markdown_order_score")),
                        success if isinstance(success, int) else "-",
                        total if isinstance(total, int) else "-",
                        row.get("timestamp") or "-",
                    )
                )
            else:
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
                        row.get("timestamp") or "-",
                    )
                )

        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def update_leaderboards(base_dir: Path) -> None:
    if not base_dir.exists() or not base_dir.is_dir():
        raise NotADirectoryError(f"Invalid evaluation directory: {base_dir}")

    rows = _collect_latest_rows(base_dir)
    if not rows:
        raise FileNotFoundError(f"No report.json files found under: {base_dir}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("format") or "unknown")].append(row)

    format_payloads = [
        _build_format_payload(format_name, format_rows)
        for format_name, format_rows in sorted(grouped.items(), key=lambda item: item[0])
    ]
    _write_leaderboard_outputs(base_dir, format_payloads)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated leaderboard files from evaluation_result.")
    parser.add_argument(
        "--base-dir",
        default="evaluation_result",
        help="Base directory containing model evaluation outputs",
    )
    args = parser.parse_args()
    update_leaderboards(Path(args.base_dir))


if __name__ == "__main__":
    main()
