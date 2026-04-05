from __future__ import annotations

import argparse
import asyncio
import sys

from src.cli import compare, corpus, evaluate, generate, listing, publish
from src.env_utils import load_env_file


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic OCR Image Generator & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    generate.add_arguments(generate_parser)
    generate_parser.set_defaults(handler=generate.run_with_args)

    publish_parser = subparsers.add_parser("publish", help="Publish a generated dataset")
    publish.add_arguments(publish_parser)
    publish_parser.set_defaults(handler=publish.run_with_args)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    evaluate.add_evaluate_arguments(evaluate_parser, output_dir_default="./evaluation_result")
    evaluate_parser.set_defaults(handler=evaluate.run_with_args)

    evaluate_run_parser = subparsers.add_parser(
        "evaluate-run",
        help="Resolve config and dependency groups, then run evaluation",
    )
    evaluate.add_evaluate_arguments(evaluate_run_parser, output_dir_default=None)
    evaluate_run_parser.set_defaults(handler=evaluate.run_wrapper_with_args)

    evaluate_all_parser = subparsers.add_parser(
        "evaluate-all",
        help="Run evaluation for all public model configs",
    )
    evaluate.add_evaluate_all_arguments(evaluate_all_parser)
    evaluate_all_parser.set_defaults(handler=evaluate.run_all_with_args)

    refresh_leaderboard_parser = subparsers.add_parser(
        "refresh-leaderboard",
        help="Refresh consolidated evaluation leaderboard outputs",
    )
    evaluate.add_refresh_leaderboard_arguments(refresh_leaderboard_parser)
    refresh_leaderboard_parser.set_defaults(handler=evaluate.refresh_leaderboard_with_args)

    compare_parser = subparsers.add_parser("compare", help="Compare evaluation results")
    compare.add_arguments(compare_parser)
    compare_parser.set_defaults(handler=compare.run_with_args)

    backends_parser = subparsers.add_parser("list-backends", help="List available backends")
    backends_parser.set_defaults(handler=listing.run_list_backends)

    configs_parser = subparsers.add_parser("list-configs", help="List available model configurations")
    configs_parser.set_defaults(handler=listing.run_list_configs)

    corpus_parser = subparsers.add_parser("corpus", help="Corpus generation commands")
    corpus.configure_parser(corpus_parser)
    return parser


def main() -> None:
    load_env_file()
    parser = create_parser()
    args = parser.parse_args()

    async_handler = getattr(args, "async_handler", None)
    if async_handler is not None:
        sys.exit(asyncio.run(async_handler(args)))

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    result = handler(args)
    if isinstance(result, int):
        sys.exit(result)
