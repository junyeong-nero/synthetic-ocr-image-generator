"""Input pairing and reports for distribution fit-capture."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def configure_parser(subparsers) -> None:
    parser = subparsers.add_parser('fit-capture', help='Fit capture parameters from clean/printed page pairs')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pairs', help='CSV with clean,captured columns; paths relative to CSV')
    source.add_argument('--clean-dir', help='Directory of clean pages matched by exact filename')
    parser.add_argument('--captured-dir', help='Captured pages (required with --clean-dir)')
    parser.add_argument('--output', required=True, help='Directory for per-pair JSON and profile.yaml')
    parser.add_argument('--channel', choices=['scanned', 'photographed'], default='scanned')
    parser.add_argument('--clean-dpi', type=float, help='Clean render DPI; default assumes A4 width')
    parser.add_argument('--seed', type=int, default=0, help='OpenCV RANSAC seed (default: 0)')
    parser.set_defaults(handler=run_fit_capture)


def collect_pairs(*, pairs=None, clean_dir=None, captured_dir=None) -> list[tuple[Path, Path]]:
    if pairs:
        if clean_dir or captured_dir:
            raise ValueError('Use --pairs or both directory arguments')
        manifest = Path(pairs)
        with manifest.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if not {'clean', 'captured'} <= set(reader.fieldnames or []):
                raise ValueError('Pairs CSV requires clean,captured columns')
            result = []
            for row in reader:
                if not row.get('clean', '').strip() or not row.get('captured', '').strip():
                    raise ValueError('Pairs CSV contains an empty path')
                result.append(tuple(manifest.parent / row[key].strip() for key in ('clean', 'captured')))
    else:
        if not clean_dir or not captured_dir:
            raise ValueError('Provide --pairs or both --clean-dir and --captured-dir')
        directories = [Path(clean_dir), Path(captured_dir)]
        names = [{p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _EXTENSIONS}
                 for directory in directories]
        if names[0] != names[1]:
            raise ValueError(f'Directory filenames do not match: {sorted(names[0] ^ names[1])}')
        result = [(directories[0] / name, directories[1] / name) for name in sorted(names[0])]
    if not result:
        raise ValueError('No capture pairs found')
    return result


def aggregate_fits(rows: list[dict], channel: str) -> dict:
    """Empirical marginal distributions; missing/unidentifiable values are omitted."""
    specs = {}
    for key in sorted({key for row in rows for key in row['params']}):
        values = [row['params'][key] for row in rows if row['params'].get(key) is not None]
        if values:
            specs[key] = {'choices': values}
    dpi = specs.pop('dpi', None)
    result = {'degradations': specs}
    if dpi is not None:
        result['dpi'] = dpi
    return {'capture_channels': {channel: result}}


def run_fit_capture(args: argparse.Namespace) -> int:
    from src.realism.capture_fit import fit_capture_files

    try:
        pairs = collect_pairs(pairs=args.pairs, clean_dir=args.clean_dir, captured_dir=args.captured_dir)
    except (OSError, ValueError) as exc:
        logger.error('%s', exc)
        return 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (clean, captured) in enumerate(pairs, 1):
        try:
            result = fit_capture_files(clean, captured, clean_dpi=args.clean_dpi, seed=args.seed)
            rows.append(result)
        except (OSError, ValueError) as exc:
            result = {'clean': str(clean), 'captured': str(captured), 'error': str(exc)}
            logger.error('Pair %d: %s', index, exc)
        (output / f'pair-{index:04d}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'profile.yaml').write_text(
        '# [measured] Capture-fit suggestions; review before merging into a profile.\n'
        '# Empirical marginal choices omit unknown values; channel weight is unchanged.\n'
        + yaml.safe_dump(aggregate_fits(rows, args.channel), sort_keys=False), encoding='utf-8')
    logger.info('Fitted %d/%d pairs -> %s', len(rows), len(pairs), output)
    return 0 if len(rows) == len(pairs) else 1
