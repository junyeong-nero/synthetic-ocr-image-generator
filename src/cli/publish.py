from __future__ import annotations

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--generated-path",
        required=True,
        help="Path to a generated dataset root containing run_manifest.json",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Override the repository ID stored in the run manifest",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Override the train split ratio used for dataset publishing",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Override the test split ratio used for dataset publishing",
    )
    parser.add_argument(
        "--license",
        default=None,
        help="Dataset card license id, e.g. cc-by-sa-3.0 when the corpus comes from Wikipedia",
    )
    parser.add_argument(
        "--text-source",
        default=None,
        help='Attribution for the corpus text, e.g. "Korean WikiText (https://github.com/lovit/kowikitext), CC BY-SA 3.0"',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write DATASET_CARD.md and report split sizes locally without uploading",
    )
    return parser


def run_with_args(args: argparse.Namespace) -> None:
    from src.pipeline import publish_pipeline

    publish_pipeline(
        generated_path=args.generated_path,
        repo_id=args.repo_id,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        dry_run=getattr(args, "dry_run", False),
        license=getattr(args, "license", None),
        text_source=getattr(args, "text_source", None),
    )
