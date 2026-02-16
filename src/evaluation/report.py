"""Report generation for evaluation results."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.types import EvaluationOutput


class ReportGenerator:
    """
    Generate evaluation reports in multiple formats.

    Supports JSON, Markdown, and HTML output.
    """

    def __init__(self, output: EvaluationOutput):
        self.output = output

    def to_json(self, path: Path) -> Path:
        """
        Generate JSON report.

        Args:
            path: Output file path.

        Returns:
            Path to generated file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "config": self.output.config,
            "metrics": self.output.metrics,
            "metric_views": self.output.metric_views,
            "summary": self.output.summary,
            "per_sample_results": self.output.per_sample_results,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return path

    def to_markdown(self, path: Path) -> Path:
        """
        Generate Markdown report.

        Args:
            path: Output file path.

        Returns:
            Path to generated file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = self.output.config
        metrics = self.output.metrics
        summary = self.output.summary
        metric_views = self.output.metric_views
        format_name = "markdown"

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
            f"| Format | {format_name} |",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")

        normalized_metrics = metric_views.get("normalized", {})
        if normalized_metrics:
            lines.extend([
                "",
                "## Metric Views",
                "",
            ])
            lines.extend([
                "### Normalized",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            for key, value in normalized_metrics.items():
                lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")
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
            ]
        )

        lines.extend(
            [
                "## Markdown Block Scores",
                "",
                "| Component | Value |",
                "|-----------|-------|",
            ]
        )
        component_keys = [
            ("Text", "avg_markdown_text_score"),
            ("Table", "avg_markdown_table_teds"),
            ("Formula", "avg_markdown_formula_score"),
            ("Order", "avg_markdown_order_score"),
            ("Overall", "avg_markdown_overall_score"),
        ]
        for label, key in component_keys:
            value = metrics.get(key)
            if isinstance(value, float):
                lines.append(f"| {label} | {value:.4f} |")
            elif isinstance(value, int):
                lines.append(f"| {label} | {float(value):.4f} |")
            else:
                lines.append(f"| {label} | N/A |")
        lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return path

    def to_html(self, path: Path) -> Path:
        """
        Generate HTML report.

        Args:
            path: Output file path.

        Returns:
            Path to generated file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = self.output.config
        metrics = self.output.metrics
        summary = self.output.summary
        metric_views = self.output.metric_views
        format_name = "markdown"

        # Build metrics table rows
        metrics_rows = ""
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_rows += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>\n"
            else:
                metrics_rows += f"<tr><td>{key}</td><td>{value}</td></tr>\n"

        metric_view_sections = ""
        normalized_metrics = metric_views.get("normalized", {})
        if normalized_metrics:
            view_rows = ""
            for key, value in normalized_metrics.items():
                if isinstance(value, float):
                    view_rows += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>\n"
                else:
                    view_rows += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
            metric_view_sections = f"""
    <div class=\"card\">
        <h2>Metric View: Normalized</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {view_rows}
        </table>
    </div>
"""

        # Get error samples
        error_samples = [
            r for r in self.output.per_sample_results if r.get("error") is not None
        ][:10]

        error_rows = ""
        for sample in error_samples:
            error_rows += f"""
            <tr>
                <td>{sample.get('index', 'N/A')}</td>
                <td>{sample.get('error', 'N/A')}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
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
            <tr><td>Format</td><td>{format_name}</td></tr>
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

{self._markdown_component_section(metrics)}

    {metric_view_sections}

    {"<div class='card'><h2>Errors (first 10)</h2><table><tr><th>Index</th><th>Error</th></tr>" + error_rows + "</table></div>" if error_samples else ""}

</body>
</html>
"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        return path

    def _markdown_component_section(self, metrics: dict) -> str:

        component_keys = [
            ("Text", "avg_markdown_text_score"),
            ("Table", "avg_markdown_table_teds"),
            ("Formula", "avg_markdown_formula_score"),
            ("Order", "avg_markdown_order_score"),
            ("Overall", "avg_markdown_overall_score"),
        ]
        rows = ""
        for label, key in component_keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                rendered = f"{float(value):.4f}"
            else:
                rendered = "N/A"
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

    def save_all(self, output_dir: Path, prefix: str = "report") -> Dict[str, Path]:
        """
        Save reports in all formats.

        Args:
            output_dir: Output directory.
            prefix: File name prefix.

        Returns:
            Dict mapping format to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "json": self.to_json(output_dir / f"{prefix}.json"),
            "markdown": self.to_markdown(output_dir / f"{prefix}.md"),
            "html": self.to_html(output_dir / f"{prefix}.html"),
        }


def generate_report(
    output: EvaluationOutput,
    output_dir: str,
    formats: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Generate reports from evaluation output.

    Args:
        output: Evaluation output.
        output_dir: Output directory.
        formats: List of formats to generate (default: all).

    Returns:
        Dict mapping format to file path.
    """
    generator = ReportGenerator(output)
    output_path = Path(output_dir)

    if formats is None:
        return generator.save_all(output_path)

    results = {}
    for fmt in formats:
        if fmt == "json":
            results["json"] = generator.to_json(output_path / "report.json")
        elif fmt == "markdown":
            results["markdown"] = generator.to_markdown(output_path / "report.md")
        elif fmt == "html":
            results["html"] = generator.to_html(output_path / "report.html")

    return results
