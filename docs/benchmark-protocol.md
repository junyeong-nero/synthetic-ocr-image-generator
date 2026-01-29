# Benchmark Protocol

This document defines the standardized protocol used for OCR benchmarking in this project.

## Protocol Version

- Version: 1.0

## Dataset Rules

- Default split: `train`
- Default subsets: `sentence`, `table`, `document`, `markdown`, `kie`
- If `--subset` is omitted, all default subsets are evaluated.

## Prompt Resolution

Prompt selection order (implemented in `EvaluationPipeline._resolve_prompt`):

1. CLI override (`--prompt` if provided programmatically)
2. Subset prompt in model config (`subsets.<name>.prompts.<format>`)
3. Format prompt in model config (`prompts.<format>`)
4. Default prompt in `src/evaluation/config.py`

The resolved prompt and its source are recorded in `protocol.json` and `report.json`.

## Protocol Snapshot

Each evaluation run writes a protocol snapshot to `protocol.json`. It captures:

- protocol version
- dataset ID, split, subset, and dataset fingerprint
- prompt, system prompt, and prompt source
- evaluation parameters (batch size, max samples)
- model config path, model ID, backend, and sampling parameters
- seed value (if provided)
- environment metadata

## Reproducibility Requirements

- Set `--seed` for deterministic generation/evaluation.
- Record environment metadata (python/torch/transformers versions, device).
- Record dataset identifiers and fingerprints.
- If `--batch-api` is used, record batch settings and output metadata.

## Reporting Outputs

Each evaluation run produces:

- `report.json` / `report.md` / `report.html`
- `protocol.json`
- `model_summary.json` (append-only)
- `leaderboard.json` / `leaderboard.md`

## Representative Metrics

- `sentence` -> `avg_cer` (lower is better)
- `table` -> `avg_teds` (higher is better)
- `document` -> `avg_overall_f1` (higher is better)
- `markdown` -> `avg_cer` (lower is better)
- `kie` -> `avg_entity_f1` (higher is better)

Normalization for leaderboard:

- `avg_cer`, `avg_wer` => `1 - score`
- other metrics are used as-is
- normalized averages are weighted by subset `total_samples` when available
