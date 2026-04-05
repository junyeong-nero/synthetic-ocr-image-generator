from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("report_files", nargs="+", help="JSON report files to compare")
    parser.add_argument("-o", "--output", default="comparison", help="Output file prefix")
    return parser


def run_with_args(args: argparse.Namespace) -> None:
    from src.evaluation.comparator import ModelComparator

    paths = [Path(p) for p in args.report_files]

    for path in paths:
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    comparator = ModelComparator.from_json_files(paths)
    comparator.save_comparison(Path(args.output))
    comparator.print_summary()
    print(f"\nComparison saved to: {args.output}.json, {args.output}.md")
