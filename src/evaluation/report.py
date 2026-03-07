import json
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.report_builders import (
    build_html_report,
    build_json_report,
    build_markdown_report,
)
from evaluation.types import EvaluationOutput


class ReportGenerator:
    def __init__(self, output: EvaluationOutput):
        self.output = output

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(build_json_report(self.output), file, indent=2, ensure_ascii=False)
        return path

    def to_markdown(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(build_markdown_report(self.output))
        return path

    def to_html(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(build_html_report(self.output))
        return path

    def save_all(self, output_dir: Path, prefix: str = "report") -> Dict[str, Path]:
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