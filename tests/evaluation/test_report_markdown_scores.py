from pathlib import Path

from evaluation.report import ReportGenerator
from evaluation.types import EvaluationOutput


def test_report_includes_markdown_block_scores(tmp_path: Path) -> None:
    output = EvaluationOutput(
        config={
            "dataset_id": "dummy",
            "format": "markdown",
            "model": {"model_id": "m", "backend": "b"},
        },
        metrics={
            "avg_markdown_text_score": 0.7,
            "avg_markdown_table_teds": 0.8,
            "avg_markdown_formula_score": 0.9,
            "avg_markdown_order_score": 0.6,
            "avg_markdown_overall_score": 0.75,
        },
        metric_views={},
        per_sample_results=[],
        summary={
            "total_samples": 1,
            "successful": 1,
            "failed": 0,
            "empty_rate": 0.0,
            "parse_fail_rate": 0.0,
            "avg_latency_ms": 1.0,
        },
    )
    generator = ReportGenerator(output)

    md_path = generator.to_markdown(tmp_path / "report.md")
    html_path = generator.to_html(tmp_path / "report.html")

    markdown = md_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")

    assert "## Markdown Block Scores" in markdown
    assert "| Overall | 0.7500 |" in markdown
    assert "Markdown Block Scores" in html
    assert "<td>Overall</td><td>0.7500</td>" in html
