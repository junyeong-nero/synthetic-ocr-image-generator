from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    subparsers = parser.add_subparsers(dest="distribution_command")
    subparsers.required = True

    measure = subparsers.add_parser(
        "measure",
        help="Measure visual statistics of real or generated document images",
    )
    source = measure.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=str, help="Directory of images (searched recursively)")
    source.add_argument("--metadata", type=str, help="Generated metadata.jsonl (uses file_name)")
    source.add_argument("--hf-dataset", type=str, help="Hugging Face dataset id (streamed)")
    measure.add_argument("--hf-config", type=str, default=None)
    measure.add_argument("--hf-split", type=str, default="train")
    measure.add_argument("--hf-image-column", type=str, default="image")
    measure.add_argument("--max-images", type=int, default=500)
    measure.add_argument("--output", type=str, required=True, help="Stats JSON output path")
    measure.add_argument(
        "--suggest-yaml",
        type=str,
        default=None,
        help="Also write directly measurable profile specs (dpi, skew, grayscale, ...) as YAML",
    )
    measure.set_defaults(handler=run_measure)

    compare = subparsers.add_parser(
        "compare",
        help="Compare a candidate stats JSON (synthetic) against a reference stats JSON (real)",
    )
    compare.add_argument("--reference", type=str, required=True)
    compare.add_argument("--candidate", type=str, required=True)
    compare.add_argument("--output", type=str, default=None, help="Optional markdown report path")
    compare.set_defaults(handler=run_compare)
    return parser


def run_measure(args: argparse.Namespace) -> int:
    from src.realism.distribution_stats import (
        iter_hf_images,
        iter_image_paths,
        iter_metadata_image_paths,
        measure_images,
        summarize_stats,
    )

    limit = args.max_images if args.max_images and args.max_images > 0 else None
    if args.images:
        source = args.images
        images = iter_image_paths(Path(args.images), limit=limit)
    elif args.metadata:
        source = args.metadata
        images = iter_metadata_image_paths(Path(args.metadata), limit=limit)
    else:
        source = f"hf://{args.hf_dataset}/{args.hf_split}"
        images = iter_hf_images(
            args.hf_dataset,
            split=args.hf_split,
            image_column=args.hf_image_column,
            config=args.hf_config,
            limit=limit,
        )

    rows = measure_images(images)
    if not rows:
        logger.error("No images could be measured from %s", source)
        return 1

    summary = summarize_stats(rows, source=source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Measured %d images from %s -> %s", len(rows), source, output)

    if args.suggest_yaml:
        suggest_path = Path(args.suggest_yaml)
        suggest_path.parent.mkdir(parents=True, exist_ok=True)
        suggest_path.write_text(
            "# Directly measured specs. Paste into a profile under capture_channels.<channel>\n"
            "# (dpi, skew_deg, grayscale, binarize) or typography (colored_background).\n"
            + yaml.safe_dump(summary["suggested_profile_specs"], sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("Suggested profile specs -> %s", suggest_path)
    return 0


def run_compare(args: argparse.Namespace) -> int:
    from src.realism.distribution_stats import compare_summaries, format_comparison_markdown

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    rows = compare_summaries(reference, candidate)
    report = format_comparison_markdown(
        rows,
        reference=reference.get("source") or args.reference,
        candidate=candidate.get("source") or args.candidate,
    )
    print(report)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    return 0
