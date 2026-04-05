#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.leaderboard import update_leaderboards


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh evaluation leaderboards")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base evaluation directory containing report.json files",
    )
    args = parser.parse_args()
    update_leaderboards(args.base_dir)


if __name__ == "__main__":
    main()