# CLI Reference

The project provides a unified CLI via `main.py`.

Recommended invocation:

```bash
uv run main.py <command> [options]
```

## Global Options
- `-h, --help`: Show help message and exit.

---

## `corpus generate`
Generate corpus text data using an LLM provider.

```bash
uv run main.py corpus generate [OPTIONS]
```

### Options
- `--lang`: Language code to generate for (default: `ko`).
- `--lang-name`: Optional language name hint for custom or unsupported codes.
- `--category`: Specific corpus category to generate (default: all categories).
- `--count`: Number of items to generate per category (default: `1000`).
- `--provider`: LLM provider (`openai`, `anthropic`) (default: `openai`).
- `--model`: Optional provider-specific model override.
- `--output-dir`: Output directory for saved corpus files.
- `--batch-size`: Number of items requested per API call (default: `100`).

---

## `corpus import-wikitext`
Import a WikiText-format dump as `paragraphs.txt` / `titles.txt` corpus files.

```bash
uv run main.py corpus import-wikitext (--kowikitext-split SPLIT ... | --input FILE ...) [OPTIONS]
```

### Options
- `--kowikitext-split`: `train`, `dev` or `test` of [Korean WikiText](https://github.com/lovit/kowikitext), downloaded from GitHub releases (repeatable). The text is CC BY-SA 3.0.
- `--input`: Local WikiText file (repeatable).
- `--lang`: Corpus language directory (default: `ko`).
- `--output-dir`: Corpus root (default: `data/corpus`).
- `--cache-dir`: Download cache (default: `data/corpus/_downloads`).
- `--max-paragraphs`: Keep at most N paragraphs (shuffled with a fixed seed).
- `--min-chars`, `--max-chars`: Paragraph length filter (default: `60` / `700`).

---

## `generate`
Generate synthetic OCR datasets.

```bash
uv run main.py generate [OPTIONS]
```

### Options
- `--repo-id`: Optional HF Hub repository ID. Required only when `--upload` is used or when you want it stored in `run_manifest.json` for later `publish`.
- `--output-dir`: Base directory for generated data (default: `./data`).
- `--lang`: Language code (default: `ko`).
- `--seed`: Random seed for reproducible generation.
- `--size`: Number of images to generate (default: `100`).
- `--shard-size`: Samples per shard directory.
- `--max-shards`: Limit generation to the first N planned shards.
- `--resume`: Resume a previous sharded generation run.
- `--upload`: Upload to Hugging Face Hub after generation completes.
- `--upload-each-shard`: Publish every shard to `--repo-id` right after it is generated (one commit of parquet files), verify it on the Hub, then delete it locally. For datasets larger than the free disk. Creates the repo as private if it is missing. Excludes `--upload`. See [generation.md](generation.md#large-runs-upload-each-shard).
- `--workers`: Shards generated in parallel with `--upload-each-shard` (default: `1`).
- `--keep-uploaded-shards`: With `--upload-each-shard`, keep each shard's images and parquet locally after upload.
- `--public`: With `--upload-each-shard`, create the dataset repo as public instead of private. An existing repo keeps its visibility.
- `--upload-dry-run`: With `--upload-each-shard`, write each shard's parquet locally and skip every Hub call and deletion.
- `--template`: Optional generation template name.
- `--template-family`: Optional template family filter.
- `--min-template-complexity`: Minimum template complexity filter (`1-5`).
- `--max-template-complexity`: Maximum template complexity filter (`1-5`).
- `--template-config-dir`: Template catalog directory path.
- `--markdown-renderer`: Markdown render backend (`pil`, `html2image`, `playwright`) (default: `playwright`).
- `--style-profile`: Style variation profile (`legacy`, `balanced`, `aggressive`).
- `--coverage-target`: Family target ratio (`family=ratio`), repeatable.
- `--novelty-window`: Recent-sample window size for novelty guard.
- `--novelty-threshold`: Similarity threshold for novelty guard.
- `--novelty-max-attempts`: Retry count before accepting low-novelty sample.
- `--similar-char-ratio`: Ratio of similar-character substitutions (default: the distribution profile's `content.similar_char_ratio`, else `0.08`).
- `--similarity-db-path`: Optional similarity DB JSON path.
- `--add-noise`, `--no-add-noise`: Enable/disable noise effect.
- `--add-blur`, `--no-add-blur`: Enable/disable blur effect.
- `--distribution-profile`: Real-world distribution profile name (`real_world_v2`, `real_world_v1`, `ko_admin_scan_v1`) or YAML path. See [distribution.md](distribution.md).
- `--license`: Dataset card license id (default: `unknown`), e.g. `cc-by-sa-3.0` when the corpus comes from Wikipedia.
- `--text-source`: Text attribution written into the dataset card.
- `--train-ratio`: Train split ratio for dataset publishing (default: `0.9`).
- `--test-ratio`: Test split ratio for dataset publishing (default: `0.1`).

Notes:

- `generate` is local-first. It writes local artifacts and shard manifests even when `--repo-id` is omitted.
- Generated dataset roots contain `run_manifest.json`, root `metadata.jsonl`, `realism_stats.json`, and per-shard `metadata.jsonl` plus `_SUCCESS` markers under `shards/`.
- Use `--upload` for inline upload, or run `publish` later against the generated path.

---

## `publish`
Publish a previously generated dataset.

```bash
uv run main.py publish --generated-path <path> [OPTIONS]
```

### Options
- `--generated-path`: (Required) Generated dataset root containing `run_manifest.json`.
- `--repo-id`: Override the repository ID stored in the manifest.
- `--train-ratio`: Override the train ratio used for publishing.
- `--test-ratio`: Override the test ratio used for publishing.
- `--license`: Dataset card license id (e.g. `cc-by-sa-3.0`).
- `--text-source`: Text attribution written into the dataset card.
- `--dry-run`: Write `DATASET_CARD.md` and report split sizes without uploading.

Notes:

- `publish` reuses generation context from `run_manifest.json` and only needs `--repo-id` when no repository ID was stored during generation.

---

## `evaluate`
Run model evaluation.

```bash
uv run main.py evaluate [OPTIONS]
```

### Options
- `--model-config`: (Required) Path to model config YAML.
- `-d, --dataset`: (Required) HF dataset ID or local path.
- `-b, --backend`: Override inference backend.
- `--split`: Dataset split (default: `train`).
- `--max-samples`: Limit evaluation samples.
- `--seed`: Random seed for reproducible evaluation.
- `--batch-api`: Use OpenAI Batch API.
- `--batch-poll-seconds`: Polling interval for batch status.
- `--batch-timeout-seconds`: Max wait time for batch completion.
- `--batch-completion-window`: Batch completion window (default: `24h`).
- `--output-dir`: Results directory (default: `./evaluation_result`).
- `--report-format`: Output format (`json`, `markdown`, `html`, `all`) (default: `all`).
- `--inference-only`: Run inference only and save `checkpoints.json`.
- `--evaluate-only`: Skip inference and evaluate from `checkpoints.json`.
- `--batch-size`: Override batch size from model config.
- `--temperature`: Override generation temperature.
- `--max-tokens`: Override max output tokens.
- `--api-base`: Override API base URL.
- `--tensor-parallel`: Override tensor parallel size.

---

## `compare`
Compare multiple evaluation reports.

```bash
uv run main.py compare [REPORT_FILES...] [OPTIONS]
```

### Options
- `-o, --output`: Output file prefix (default: `comparison`).

---

## `rank-correlation`
Correlate model rankings from this repo's synthetic evaluation outputs against an external "real" benchmark ranking (Spearman rho and Kendall tau, with a bootstrap 95% CI).

```bash
uv run main.py rank-correlation --synthetic evaluation_result/leaderboard.json --real real_scores.csv [OPTIONS]
```

### Options
- `--synthetic`: Synthetic score source — an `evaluation_result/leaderboard.json`, a single `report.json`, or a directory scanned recursively for `report.json` files (e.g. `evaluation_result/`).
- `--real`: Real benchmark scores — CSV with `model,score` columns (or any two columns, matched by name or position), or JSON (an object of `name: score`, or a list of `{"model": ..., "score": ...}` objects). YAML is also accepted for the equivalent structure.
- `--metric`: Synthetic metric key to correlate (default: `avg_markdown_overall_score`).
- `--aliases`: Optional JSON/YAML file mapping an alternate name (from either side) to the name it should be treated as equal to, for models whose synthetic and real names do not normalize to the same id.
- `--language`: Language to select when `--synthetic` covers multiple languages (e.g. `ko`, `ja`); required when the source has more than one language.
- `--output`: Also write the markdown report (summary + per-model rank table) to this path; the report is always printed to stdout.
- `--bootstrap-iterations`: Bootstrap resamples for the 95% CI (default: `2000`).
- `--seed`: Bootstrap RNG seed, for reproducible CIs (default: `12345`).

Models are matched by a normalized id: any org/path prefix is stripped (text before the final `/`), then the remainder is lowercased with punctuation removed, e.g. `lightonai/LightOnOCR-2-1B` and `LightOnOCR 2.1B` both normalize to `lightonocr21b`.

---

## `distribution measure`
Measure visual statistics of real or generated document images.

```bash
uv run main.py distribution measure (--images DIR | --metadata JSONL | --hf-dataset ID) --output STATS.json [OPTIONS]
```

### Options
- `--images` / `--metadata` / `--hf-dataset`: Image source (directory, generated `metadata.jsonl`, or streamed HF dataset with `--hf-config`, `--hf-split`, `--hf-image-column`).
- `--where key=value`: With `--metadata`, only rows matching the filter (repeatable), e.g. `capture_channel=scanned`.
- `--max-images`: Limit (default: `500`; `0` = all).
- `--suggest-yaml`: Also write directly measurable profile specs (DPI and skew histograms, grayscale, binary and coloured-background rates, plus a `margins_mm` histogram derived from layout stats).
- `--save-rows`: Also write per-image rows (every measured stat plus the image `path`) under a `rows` key in the output JSON. Required input for `distribution discriminate`.

Every row also carries the layout metrics from `src/realism/layout_stats.py` (`text_line_count`, `text_line_height_frac`/`_pt`, `margin_*_frac`, `column_count`, `text_area_frac`, `rule_count`) merged in automatically — see [distribution.md](distribution.md#layout-and-text-statistics).

## `distribution compare`
Compare a candidate stats JSON against a reference stats JSON.

```bash
uv run main.py distribution compare --reference REAL.json --candidate SYN.json [--output GAP.md]
```

Prints a table ranked by normalised Wasserstein distance: under 0.1 is close, 0.1 to 0.3 is noticeable, above 0.3 is different. Works on `distribution measure` output or on `distribution text-stats` output (any two summaries with the same metric keys).

## `distribution discriminate`
Train a real-vs-synthetic classifier two-sample test (numpy-only L2 logistic regression, stratified k-fold) on the per-image rows from two `distribution measure --save-rows` outputs.

```bash
uv run main.py distribution discriminate --reference REAL.json --candidate SYN.json [OPTIONS]
```

### Options
- `--reference` / `--candidate`: Stats JSON files produced by `distribution measure --save-rows` (must contain a non-empty `rows` list).
- `--top`: Number of most-confident-synthetic candidate images to report, with paths and probabilities (default: `20`).
- `--folds`: Stratified k-fold count (default: `5`, clamped down to the smaller class size).
- `--l2`: L2 regularisation strength (default: `1.0`).
- `--seed`: RNG seed for class-balancing subsampling and fold shuffling, for a reproducible report (default: `0`).
- `--output`: Optional markdown report path.

Fails with a clear error (exit code `1`) if either input JSON has no `rows` (i.e. was measured without `--save-rows`). See [distribution.md](distribution.md#real-vs-synthetic-discriminator) for how to read AUC and the top-N list.

## `distribution text-stats`
Measure per-page text statistics (character-class shares, symbol usage) of real or generated text.

```bash
uv run main.py distribution text-stats (--metadata JSONL | --texts JSONL | --text-dir DIR) --output STATS.json [OPTIONS]
```

### Options
- `--metadata`: Generated `metadata.jsonl`; reads `GT_markdown` with markdown syntax (`#`, `*`, `|`, list markers, links, HTML tags) stripped first.
- `--texts` / `--field`: JSONL file with one text per line, and the field name holding it (default `text`).
- `--text-dir`: Directory of `.txt` files (searched recursively, no stripping).
- `--max-texts`: Limit (default: `500`; `0` = all).
- `--top-k`: Number of most frequent non-alphanumeric symbols to record (default: `20`).

See [distribution.md](distribution.md#layout-and-text-statistics) for the metric list.

---

## `list-backends`
List all available inference backends.

---

## `list-configs`
List all available model configurations in `configs/models/`.

---

## Script Wrappers

For common workflows, these scripts are recommended:

- `scripts/synthesize/generate.sh`
- `scripts/evaluate/run.sh`
- `scripts/evaluate/run-all.sh`
- `scripts/evaluate/update-leaderboard.sh`

### `scripts/evaluate/run.sh`

Runs a single model config with dependency-group handling.

```bash
scripts/evaluate/run.sh <config_name>|--model-id <config_name_or_model_id> [evaluation options...]
```

Wrapper-specific options:

- `-m, --model-id <ref>`: Model config name or model ID.
- `-d, --dataset <repo>`: Dataset ID/path override.
- `-l, --language <code>`: Language code (default: `ko`).
- `-n, --max-samples <n>`: Limit evaluation samples.
- `--split <train|test>`: Dataset split override.

All other evaluation flags are forwarded to `uv run main.py evaluate`.

### `scripts/evaluate/run-all.sh`

Runs all configs under `configs/models/`.

```bash
scripts/evaluate/run-all.sh [--dataset <repo>] [--language <code>] [-n|--max-samples <n>] [--split <train|test>]
scripts/evaluate/run-all.sh [DATASET] [MAX_SAMPLES] [SPLIT] [LANGUAGE]
```

Notes:

- Prefer `-n, --max-samples` for sample limits.
- `-m` is still accepted for max samples as a deprecated alias.
- `-l, --language` defaults to `ko` and is forwarded to each run.

### `scripts/synthesize/generate_similarity_db.sh`

Builds language-specific character similarity DB files used by `generate`.

Key options:

- `--lang <code>`: Target language (repeatable).
- `--all`: Build for all language scripts in `scripts/synthesize/lang/`.
- `--font-path <path>`: Override font file.
- `--corpus-path <path>`: Override the final merged corpus file.
- `--generate-corpus`: Run `main.py corpus generate` first, merge the generated category files, then build the DB.
- `--corpus-provider <name>` / `--corpus-model <name>`: Control the LLM corpus generation backend.
- `--corpus-count <n>` / `--corpus-batch-size <n>`: Control how much corpus text is generated before merging.
- `--corpus-category <name>`: Limit LLM corpus generation to specific categories (repeatable).
- `--auto-generate-corpus`: Auto-generate `corpus_<lang>.txt` from Wikimedia when missing.
- `--corpus-sentences <n>`: Sentence count for auto-generated corpus (default: `100000`).
- `--db-path <path>`: Override output DB path (single language only).
- `--threshold <float>`: Similarity threshold (default: `0.6`).
- `--top-k <int>`: Max similar chars per character (default: `8`).

## `distribution fit-capture`

```bash
uv run main.py distribution fit-capture --pairs pairs.csv --output stats/capture-fit [OPTIONS]
uv run main.py distribution fit-capture --clean-dir clean --captured-dir scans --output stats/capture-fit
```

- `--pairs`: CSV with `clean,captured` headers. Relative paths resolve beside the CSV.
- `--clean-dir`, `--captured-dir`: alternative input, matching exact filenames in the two directories (nonrecursive). Supported extensions: PNG, JPEG, TIFF, BMP, WebP. Missing counterparts and empty inputs are errors; use CSV for differing extensions/names.
- `--output`: required output directory, containing numbered per-pair JSON files and `profile.yaml`. Reusing a directory overwrites matching report names; prefer a fresh directory per run.
- `--channel`: `scanned` (default) or `photographed`; names the output profile channel.
- `--clean-dpi`: positive clean-render DPI; omitted means A4 width (210 mm).
- `--seed`: integer seed for registration and noise calibration (default `0`).

Successful pairs contribute empirical choices to the YAML snippet. Failed pairs
are recorded in their JSON and excluded; any failure returns exit status 1.
The command only writes reports, never edits profiles or generates pages.
See [capture fitting](distribution.md#fit-a-capture-channel-from-printscan-pairs)
for the physical collection loop, units, model assumptions and review steps.
