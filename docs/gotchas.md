# Gotchas

- Fonts are required for generation and `--font-path` is required by the CLI.
- Generated artifacts live in `data/`, `evaluation_results/`, and `test_results/` and should not be committed.
- Keep `model_id:` and `dependency_group:` at column 0 in model YAML files. Scripts parse them with `grep`.
- If you add a new dependency group, update both `[dependency-groups]` and `[tool.uv.conflicts]` in `pyproject.toml`.
- Dependency groups can be combined; use `uv run --group evaluate --group <model>` for model-specific evaluation.
- `evaluate` defaults to running all subsets if `--subset` is omitted.
- `evaluate` defaults to `--split train` to match generated datasets.
- `EvaluationRunner` uses `checkpoint.json` to resume runs; delete it to start fresh.
- Uploading to Hugging Face requires `huggingface-cli login`.
- Evaluation writes `protocol.json` and updates `leaderboard.json`/`leaderboard.md` in the output directory.
- Generation writes `realism_stats.json` alongside `metadata.jsonl`.
- Batch API results may return out of order; outputs are mapped by `custom_id`.

## Dataset Column Expectations

- Sentence: `typo_text` is the target column (default `EvaluationConfig.target_column`).
- Table: evaluator expects `html` and `json` columns.
- Document: evaluator expects `ground_truth` with `elements` list.
- Markdown: evaluator expects `markdown` column.
- KIE: evaluator expects `entities` or `ground_truth.entities` and optional `line_items`.
