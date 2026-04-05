import argparse
import logging
from pathlib import Path

from src.corpus_llm.constants import (
    CATEGORIES,
    CORPUS_DIR,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
)
from src.corpus_llm.pipeline import run_generation
from src.corpus_llm.providers import get_provider


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--lang",
        type=str,
        default="ko",
        help="Language code to generate data for (for example: ko, en, ja, hi, fr, de, es)",
    )
    parser.add_argument(
        "--lang-name",
        type=str,
        default=None,
        help="Optional language name hint used for unsupported or custom language codes",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=list(CATEGORIES.keys()),
        help="Specific category to generate (default: all)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of items to generate per category",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model to use "
            f"(default: {DEFAULT_OPENAI_MODEL} for OpenAI, {DEFAULT_ANTHROPIC_MODEL} for Anthropic)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CORPUS_DIR),
        help="Output directory for corpus files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for generation (items per API call)",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate corpus data using LLM")
    return add_arguments(parser)


async def run_with_args(args: argparse.Namespace) -> int:
    try:
        provider = get_provider(args.provider, args.model)
    except ImportError as exc:
        logging.getLogger(__name__).error(str(exc))
        return 1

    categories = [args.category] if args.category else list(CATEGORIES.keys())

    return await run_generation(
        provider=provider,
        categories=categories,
        lang=args.lang,
        count=args.count,
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
        lang_name=args.lang_name,
    )


async def run_cli() -> int:
    parser = create_parser()
    args = parser.parse_args()
    return await run_with_args(args)