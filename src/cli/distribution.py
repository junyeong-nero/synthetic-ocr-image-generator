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
    measure.add_argument(
        "--where",
        action="append",
        default=None,
        help="With --metadata: only rows where key=value (repeatable), e.g. capture_channel=scanned",
    )
    measure.add_argument("--max-images", type=int, default=500)
    measure.add_argument("--output", type=str, required=True, help="Stats JSON output path")
    measure.add_argument(
        "--suggest-yaml",
        type=str,
        default=None,
        help="Also write directly measurable profile specs (dpi, skew, grayscale, ...) as YAML",
    )
    measure.add_argument(
        "--save-rows",
        action="store_true",
        help="Also write per-image rows (all stats + image path) under a `rows` key, "
        "needed by `distribution discriminate`",
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

    text_stats = subparsers.add_parser(
        "text-stats",
        help="Measure per-page text statistics (character-class shares, symbols) of real or generated text",
    )
    text_source = text_stats.add_mutually_exclusive_group(required=True)
    text_source.add_argument(
        "--metadata", type=str, help="Generated metadata.jsonl (uses GT_markdown, markdown syntax stripped)"
    )
    text_source.add_argument("--texts", type=str, help="JSONL file with one text field per line")
    text_source.add_argument("--text-dir", type=str, help="Directory of .txt files (searched recursively)")
    text_stats.add_argument("--field", type=str, default="text", help="With --texts: JSON field holding the text")
    text_stats.add_argument("--max-texts", type=int, default=500)
    text_stats.add_argument(
        "--top-k", type=int, default=20, help="Number of most frequent non-alphanumeric symbols to record"
    )
    text_stats.add_argument("--output", type=str, required=True, help="Stats JSON output path")
    text_stats.set_defaults(handler=run_text_stats)

    discriminate = subparsers.add_parser(
        "discriminate",
        help="Train a real-vs-synthetic classifier two-sample test on measured per-image rows",
    )
    discriminate.add_argument(
        "--reference", type=str, required=True, help="Reference (real) stats JSON from `distribution measure --save-rows`"
    )
    discriminate.add_argument(
        "--candidate", type=str, required=True, help="Candidate (synthetic) stats JSON from `distribution measure --save-rows`"
    )
    discriminate.add_argument("--top", type=int, default=20, help="Most-confident-synthetic images to report (default: 20)")
    discriminate.add_argument(
        "--folds", type=int, default=5, help="Stratified k-fold count (default: 5, clamped to the smaller class size)"
    )
    discriminate.add_argument("--l2", type=float, default=1.0, help="L2 regularisation strength (default: 1.0)")
    discriminate.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling/fold shuffling (default: 0)")
    discriminate.add_argument("--output", type=str, default=None, help="Optional markdown report path")
    discriminate.set_defaults(handler=run_discriminate)
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
        where = dict(item.split("=", 1) for item in (args.where or []) if "=" in item)
        if where:
            source = f"{source} [{', '.join(f'{k}={v}' for k, v in where.items())}]"
        images = iter_metadata_image_paths(Path(args.metadata), limit=limit, where=where)
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
    if args.save_rows:
        summary["rows"] = rows
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
            "# margins_mm -> page.margins_mm.{top,bottom,left,right} (present only when\n"
            "# layout stats could be measured; no body_font_pt suggestion, see docs).\n"
            + yaml.safe_dump(summary["suggested_profile_specs"], sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("Suggested profile specs -> %s", suggest_path)
    return 0


def run_text_stats(args: argparse.Namespace) -> int:
    from src.realism.text_stats import (
        compute_text_stats,
        iter_jsonl_texts,
        iter_metadata_texts,
        iter_text_dir,
        summarize_text_stats,
        top_symbols,
    )

    limit = args.max_texts if args.max_texts and args.max_texts > 0 else None
    if args.metadata:
        source = args.metadata
        texts = list(iter_metadata_texts(Path(args.metadata), limit=limit))
    elif args.texts:
        source = args.texts
        texts = list(iter_jsonl_texts(Path(args.texts), field=args.field, limit=limit))
    else:
        source = args.text_dir
        texts = list(iter_text_dir(Path(args.text_dir), limit=limit))

    if not texts:
        logger.error("No texts could be read from %s", source)
        return 1

    rows = [compute_text_stats(text) for text in texts]
    summary = summarize_text_stats(rows, source=source, top_symbols=top_symbols(texts, top_k=args.top_k))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Measured text stats for %d pages from %s -> %s", len(rows), source, output)
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


def run_discriminate(args: argparse.Namespace) -> int:
    from src.realism.discriminator import format_discriminator_markdown, load_rows, run_discriminator

    try:
        reference_rows, reference_source = load_rows(args.reference)
        candidate_rows, candidate_source = load_rows(args.candidate)
        result = run_discriminator(
            reference_rows,
            candidate_rows,
            n_splits=args.folds,
            l2=args.l2,
            top_n=args.top,
            seed=args.seed,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    report = format_discriminator_markdown(result, reference_source, candidate_source)
    print(report)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    return 0
