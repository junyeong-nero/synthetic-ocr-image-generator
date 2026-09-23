from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.corpus_llm import cli as corpus_cli

logger = logging.getLogger(__name__)


def configure_parser(corpus_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_command")
    corpus_subparsers.required = True

    corpus_generate_parser = corpus_subparsers.add_parser(
        "generate",
        help="Generate corpus data using LLM",
    )
    corpus_cli.add_arguments(corpus_generate_parser)
    corpus_generate_parser.set_defaults(async_handler=run_with_args)

    import_parser = corpus_subparsers.add_parser(
        "import-wikitext",
        help="Import a WikiText dump (e.g. Korean WikiText) as paragraphs/titles corpus files",
    )
    source = import_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", action="append", help="Local WikiText file (repeatable)")
    source.add_argument(
        "--kowikitext-split",
        action="append",
        choices=["train", "dev", "test"],
        help="Download lovit/kowikitext split(s) from GitHub releases (CC BY-SA 3.0)",
    )
    import_parser.add_argument("--lang", default="ko")
    import_parser.add_argument("--output-dir", default="data/corpus")
    import_parser.add_argument("--cache-dir", default="data/corpus/_downloads")
    import_parser.add_argument("--max-paragraphs", type=int, default=None)
    import_parser.add_argument("--min-chars", type=int, default=60)
    import_parser.add_argument("--max-chars", type=int, default=700)
    import_parser.set_defaults(handler=run_import_wikitext)
    return corpus_parser


def run_import_wikitext(args: argparse.Namespace) -> int:
    from src.corpus_wikitext import (
        download_kowikitext,
        iter_wikitext_lines,
        parse_wikitext,
        write_corpus,
    )

    if args.input:
        paths = [Path(path) for path in args.input]
    else:
        paths = [download_kowikitext(split, Path(args.cache_dir)) for split in args.kowikitext_split]
    corpus = parse_wikitext(
        iter_wikitext_lines(paths),
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    paragraphs_path, titles_path = write_corpus(
        corpus,
        Path(args.output_dir),
        args.lang,
        max_paragraphs=args.max_paragraphs,
    )
    logger.info(
        "Wrote %d paragraphs -> %s and %d titles -> %s",
        len(corpus.paragraphs) if not args.max_paragraphs else min(args.max_paragraphs, len(corpus.paragraphs)),
        paragraphs_path,
        len(corpus.titles),
        titles_path,
    )
    return 0


async def run_with_args(args: argparse.Namespace) -> int:
    return await corpus_cli.run_with_args(args)
