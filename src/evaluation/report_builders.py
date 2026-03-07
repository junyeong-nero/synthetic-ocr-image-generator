from evaluation.types import EvaluationOutput


def build_json_report(output: EvaluationOutput) -> dict:
    return {
        "config": output.config,
        "metrics": output.metrics,
        "metric_views": output.metric_views,
        "summary": output.summary,
        "per_sample_results": output.per_sample_results,
    }


def build_markdown_report(output: EvaluationOutput) -> str:
    config = output.config
    metrics = output.metrics
    summary = output.summary
    metric_views = output.metric_views

    lines = [
        "# OCR Evaluation Report",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | {config.get('model', {}).get('model_id', 'N/A')} |",
        f"| Backend | {config.get('model', {}).get('backend', 'N/A')} |",
        f"| Dataset | {config.get('dataset_id', 'N/A')} |",
        "| Format | markdown |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {_render_metric_value(value)} |")

    normalized_metrics = metric_views.get("normalized", {})
    if normalized_metrics:
        lines.extend(
            [
                "",
                "## Metric Views",
                "",
                "### Normalized",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]
        )
        for key, value in normalized_metrics.items():
            lines.append(f"| {key} | {_render_metric_value(value)} |")
        lines.append("")

    lines.extend(
        [
            "",
            "## Execution Summary",
            "",
            f"- **Total Samples**: {summary.get('total_samples', 0)}",
            f"- **Successful**: {summary.get('successful', 0)}",
            f"- **Failed**: {summary.get('failed', 0)}",
            f"- **Empty Rate**: {summary.get('empty_rate', 0.0):.4f}",
            f"- **Parse Fail Rate**: {summary.get('parse_fail_rate', 0.0):.4f}",
            f"- **Average Latency**: {summary.get('avg_latency_ms', 0):.2f} ms",
            "",
            "## Markdown Block Scores",
            "",
            "| Component | Value |",
            "|-----------|-------|",
        ]
    )
    lines.extend(build_markdown_component_rows(metrics))
    lines.append("")
    return "\n".join(lines)


def build_html_report(output: EvaluationOutput) -> str:
    config = output.config
    metrics = output.metrics
    summary = output.summary
    metric_views = output.metric_views

    metrics_rows = ""
    for key, value in metrics.items():
        metrics_rows += f"<tr><td>{key}</td><td>{_render_metric_value(value)}</td></tr>\n"

    metric_view_sections = ""
    normalized_metrics = metric_views.get("normalized", {})
    if normalized_metrics:
        view_rows = ""
        for key, value in normalized_metrics.items():
            view_rows += f"<tr><td>{key}</td><td>{_render_metric_value(value)}</td></tr>\n"
        metric_view_sections = f"""
    <div class=\"card\">
        <h2>Metric View: Normalized</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {view_rows}
        </table>
    </div>
"""

    error_samples = [
        row for row in output.per_sample_results if row.get("error") is not None
    ][:10]
    error_rows = ""
    for sample in error_samples:
        error_rows += f"""
            <tr>
                <td>{sample.get('index', 'N/A')}</td>
                <td>{sample.get('error', 'N/A')}</td>
            </tr>
            """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .metric-value {{ font-family: monospace; font-size: 1.1em; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .summary-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .summary-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .summary-label {{ color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>OCR Evaluation Report</h1>

    <div class="card">
        <h2>Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Model</td><td>{config.get('model', {}).get('model_id', 'N/A')}</td></tr>
            <tr><td>Backend</td><td>{config.get('model', {}).get('backend', 'N/A')}</td></tr>
            <tr><td>Dataset</td><td>{config.get('dataset_id', 'N/A')}</td></tr>
            <tr><td>Format</td><td>markdown</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-value">{summary.get('total_samples', 0)}</div>
                <div class="summary-label">Total Samples</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{summary.get('successful', 0)}</div>
                <div class="summary-label">Successful</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{summary.get('failed', 0)}</div>
                <div class="summary-label">Failed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{summary.get('avg_latency_ms', 0):.1f}ms</div>
                <div class="summary-label">Avg Latency</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{summary.get('empty_rate', 0.0):.2%}</div>
                <div class="summary-label">Empty Rate</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{summary.get('parse_fail_rate', 0.0):.2%}</div>
                <div class="summary-label">Parse Fail Rate</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {metrics_rows}
        </table>
    </div>

{build_html_component_section(metrics)}

    {metric_view_sections}

    {"<div class='card'><h2>Errors (first 10)</h2><table><tr><th>Index</th><th>Error</th></tr>" + error_rows + "</table></div>" if error_samples else ""}

</body>
</html>
"""


def build_markdown_component_rows(metrics: dict) -> list[str]:
    rows: list[str] = []
    for label, key in markdown_component_keys():
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            rows.append(f"| {label} | {float(value):.4f} |")
        else:
            rows.append(f"| {label} | N/A |")
    return rows


def build_html_component_section(metrics: dict) -> str:
    rows = ""
    for label, key in markdown_component_keys():
        value = metrics.get(key)
        rendered = f"{float(value):.4f}" if isinstance(value, (int, float)) else "N/A"
        rows += f"<tr><td>{label}</td><td>{rendered}</td></tr>\n"

    return f"""
    <div class="card">
        <h2>Markdown Block Scores</h2>
        <table>
            <tr><th>Component</th><th>Value</th></tr>
            {rows}
        </table>
    </div>
"""


def markdown_component_keys() -> list[tuple[str, str]]:
    return [
        ("Text", "avg_markdown_text_score"),
        ("Table", "avg_markdown_table_teds"),
        ("Formula", "avg_markdown_formula_score"),
        ("Order", "avg_markdown_order_score"),
        ("Overall", "avg_markdown_overall_score"),
    ]


def _render_metric_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
