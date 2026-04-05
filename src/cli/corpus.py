from __future__ import annotations

import argparse

from src.corpus_llm import cli as corpus_cli


def configure_parser(corpus_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_command")
    corpus_subparsers.required = True

    corpus_generate_parser = corpus_subparsers.add_parser(
        "generate",
        help="Generate corpus data using LLM",
    )
    corpus_cli.add_arguments(corpus_generate_parser)
    corpus_generate_parser.set_defaults(async_handler=run_with_args)
    return corpus_parser


async def run_with_args(args: argparse.Namespace) -> int:
    return await corpus_cli.run_with_args(args)
