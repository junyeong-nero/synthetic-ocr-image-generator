"""Multi-model comparison utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelComparator:
    """
    Compare evaluation results across multiple models.

    Supports loading results from JSON files and generating comparison tables.
    """

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results

    @staticmethod
    def _build_row(result: Dict[str, Any]) -> Dict[str, Any]:
        config = result.get("config", {})
        model_config = config.get("model", {})
        summary = result.get("summary", {})
        metrics = result.get("metrics", {})
        total_samples = summary.get("total_samples", 0)
        successful = summary.get("successful", 0)

        success_rate = successful / total_samples if total_samples > 0 else 0
        format_name = config.get("format", "unknown")
        return {
            "model": model_config.get("model_id", "unknown"),
            "backend": model_config.get("backend", "unknown"),
            "format": format_name,
            "dataset": config.get("dataset_id", "unknown"),
            "metrics": metrics,
            "samples": total_samples,
            "success_rate": success_rate,
            "avg_latency_ms": summary.get("avg_latency_ms", 0),
        }

    def _iter_rows(self) -> List[Dict[str, Any]]:
        return [self._build_row(result) for result in self.results]

    @classmethod
    def from_json_files(cls, paths: List[Path]) -> "ModelComparator":
        """
        Create comparator from JSON report files.

        Args:
            paths: List of paths to JSON report files.

        Returns:
            ModelComparator instance.
        """
        results = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        return cls(results)

    @classmethod
    def from_directory(cls, directory: Path, pattern: str = "*.json") -> "ModelComparator":
        """
        Create comparator from all JSON files in a directory.

        Args:
            directory: Directory containing JSON report files.
            pattern: Glob pattern for JSON files.

        Returns:
            ModelComparator instance.
        """
        directory = Path(directory)
        paths = list(directory.glob(pattern))
        return cls.from_json_files(paths)

    def to_dataframe(self):
        """
        Create comparison DataFrame.

        Returns:
            pandas DataFrame with comparison data.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for DataFrame export. "
                "Install with: pip install pandas"
            )

        rows = []
        for row in self._iter_rows():
            rows.append({
                "model": row["model"],
                "backend": row["backend"],
                "format": row["format"],
                "dataset": row["dataset"],
                **row["metrics"],
                "samples": row["samples"],
                "success_rate": row["success_rate"],
                "avg_latency_ms": row["avg_latency_ms"],
            })

        return pd.DataFrame(rows)

    def to_markdown_table(self) -> str:
        """
        Generate Markdown comparison table.

        Returns:
            Markdown table string.
        """
        df = self.to_dataframe()
        return str(df.to_markdown(index=False))

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Get comparison data as list of dicts.

        Returns:
            List of comparison dictionaries.
        """
        return self._iter_rows()

    def rank_by_metric(
        self,
        metric: str,
        ascending: bool = True,
    ) -> List[str]:
        """
        Rank models by a specific metric.

        Args:
            metric: Metric name to rank by.
            ascending: Sort ascending (lower is better) if True.

        Returns:
            List of model IDs sorted by metric.
        """
        df = self.to_dataframe()
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        sorted_df = df.sort_values(metric, ascending=ascending)
        return [str(model) for model in sorted_df["model"].tolist()]

    def get_best_model(
        self,
        metric: str,
        lower_is_better: bool = True,
    ) -> Dict[str, Any]:
        """
        Get the best performing model for a metric.

        Args:
            metric: Metric name.
            lower_is_better: If True, lower values are better.

        Returns:
            Dict with best model info.
        """
        df = self.to_dataframe()
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        if lower_is_better:
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()

        best_row = df.loc[best_idx]
        return {
            "model": best_row["model"],
            "backend": best_row["backend"],
            metric: best_row[metric],
        }

    def save_comparison(self, output_path: Path) -> Dict[str, Path]:
        """
        Save comparison results.

        Args:
            output_path: Output file path (without extension).

        Returns:
            Dict mapping format to file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {}

        # Save JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        results["json"] = json_path

        # Save Markdown
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Model Comparison\n\n")
            f.write(self.to_markdown_table())
        results["markdown"] = md_path

        return results

    def print_summary(self) -> None:
        """Print comparison summary to console."""
        print("\n" + "=" * 60)
        print(" MODEL COMPARISON ")
        print("=" * 60)

        for row in self.to_dict():
            print(f"\n{row['model']} ({row['backend']})")
            print(f"  Dataset: {row['dataset']}")
            print(f"  Samples: {row['samples']} (success rate: {row['success_rate']:.1%})")
            print(f"  Latency: {row['avg_latency_ms']:.1f}ms")
            print("  Metrics:")
            for metric, value in row["metrics"].items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")

        print("\n" + "=" * 60)


def compare_models(
    report_paths: List[str],
    output_path: Optional[str] = None,
) -> ModelComparator:
    """
    Compare models from report files.

    Args:
        report_paths: List of paths to JSON report files.
        output_path: Optional path to save comparison results.

    Returns:
        ModelComparator instance.
    """
    comparator = ModelComparator.from_json_files(
        [Path(p) for p in report_paths]
    )

    if output_path:
        comparator.save_comparison(Path(output_path))

    return comparator
