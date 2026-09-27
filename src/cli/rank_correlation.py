from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--synthetic",
        required=True,
        help=(
            "Synthetic score source: an evaluation_result/leaderboard.json, a single "
            "report.json, or a directory scanned recursively for report.json files"
        ),
    )
    parser.add_argument(
        "--real",
        required=True,
        help="Real benchmark scores: CSV (model,score columns) or JSON (object or list of objects)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Synthetic metric key to correlate (default: avg_markdown_overall_score)",
    )
    parser.add_argument(
        "--aliases",
        default=None,
        help="Optional JSON/YAML file mapping alternate model names to a shared canonical name",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language to select when the synthetic source covers multiple languages (e.g. ko, ja)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Also write the markdown report to this path",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Bootstrap resamples for the 95%% CI (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Bootstrap RNG seed, for reproducible CIs (default: 12345)",
    )
    return parser


def run_with_args(args: argparse.Namespace) -> int:
    from src.evaluation.rank_correlation import (
        DEFAULT_BOOTSTRAP_ITERATIONS,
        DEFAULT_METRIC,
        DEFAULT_SEED,
        compute_rank_correlation,
        load_aliases,
        load_real_scores,
        load_synthetic_scores,
        render_markdown_report,
    )

    synthetic_path = Path(args.synthetic)
    real_path = Path(args.real)
    metric = args.metric or DEFAULT_METRIC

    try:
        synthetic_scores = load_synthetic_scores(synthetic_path, metric=metric, language=args.language)
        real_scores = load_real_scores(real_path)
        aliases = load_aliases(Path(args.aliases)) if args.aliases else {}
        result = compute_rank_correlation(
            synthetic_scores,
            real_scores,
            aliases=aliases,
            metric=metric,
            n_bootstrap=(
                args.bootstrap_iterations
                if args.bootstrap_iterations is not None
                else DEFAULT_BOOTSTRAP_ITERATIONS
            ),
            seed=args.seed if args.seed is not None else DEFAULT_SEED,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    report = render_markdown_report(
        result, synthetic_source=str(synthetic_path), real_source=str(real_path)
    )
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\nReport written to: {output_path}", file=sys.stderr)

    return 0
