# Data Generation Guide

The `generate` command creates synthetic markdown OCR samples and writes local artifacts. Upload is now an explicit follow-up step via `publish` or `generate --upload`.

In practice, generation is shard-aware by default even for single-shard runs: the output root keeps publish/resume state in `run_manifest.json`, writes per-shard artifacts under `shards/`, and rebuilds aggregate metadata files at the dataset root.

## Quick Start

```bash
uv run main.py generate \
  --lang "ko" \
  --size 100
```

Optional upload:

- `--repo-id`: Hugging Face dataset repository ID.
- `--upload`: publish to Hugging Face Hub immediately after generation.

Core defaults:

- `--lang ko`
- `--size 100`
- `--output-dir ./data`
- `--markdown-renderer playwright`
- `--style-profile balanced`
- `--novelty-window 80`
- `--novelty-threshold 0.95`
- `--novelty-max-attempts 4`
- `--similar-char-ratio 0.08`

## Mental Model

Generation now follows a three-layer model:

- Template catalog: YAML templates define document families, shapes, sampling weights, and block composition rules.
- Sections composition: `mode: sections` is the supported template mode. Rich block composition is enabled by blueprint keys such as `document_shape`, `section_count`, `blocks_per_section`, `allowed_blocks`, and `required_blocks`.
- Quality controls: coverage balancing + novelty guard + style variation produce diverse but controllable outputs.

Important distinction:

- `--template` / `--template-family` / complexity filters control which template is used.
- `--style-profile` controls rendering variability only; it does not select a document family or shape.

## Option Reference

### Template Selection

- `--template`: Exact template id or alias. If resolved, it takes precedence over family/complexity filters.
- `--template-family`: Family filter for random sampling (examples: `business`, `technical`, `academic`, `operations`, `forms`, `sections`).
- `--min-template-complexity`: Lower bound (`1-5`), default `None`.
- `--max-template-complexity`: Upper bound (`1-5`), default `None`.
- `--template-config-dir`: Catalog directory, default uses `configs/generator/templates`.

Selection behavior details:

- If `--template` is valid, exactly that template is used.
- If `--template` is unknown, generation logs a warning and falls back to filtered random selection.
- If both min/max complexity are provided in reverse order, values are internally swapped.
- If filters remove every candidate, generation logs a warning and falls back to the full catalog.

### Diversity and Quality

- `--coverage-target family=ratio` (repeatable): nudges family distribution toward target ratios.
- `--novelty-window`: number of recent signatures kept for duplicate-structure detection.
- `--novelty-threshold`: max allowed structural similarity before retry.
- `--novelty-max-attempts`: retries per sample before accepting current candidate.

Coverage parsing notes:

- Accepts `family=0.4` and `family:0.4` forms.
- Ratios are clamped to `[0.0, 1.0]`.
- Invalid tokens are ignored.

### Rendering and OCR Noise

- `--markdown-renderer`: `pil`, `html2image`, or `playwright`.
- `--style-profile`: `legacy`, `balanced`, `aggressive`.
- `--similar-char-ratio`: proportion of characters replaced with lookalikes.
- `--similarity-db-path`: explicit JSON path for similarity DB lookup.
- `--add-noise` / `--no-add-noise`: explicit noise override.
- `--add-blur` / `--no-add-blur`: explicit blur override.

Formula generation/rendering notes:

- The built-in hard-coded formula pool now contains 100+ normalized expressions spanning algebra, calculus, physics, probability, and ML/LLM training objectives.
- Formula rasterization uses a bounded in-process cache (256 entries) to avoid unbounded memory growth during long generation runs.

### Dataset Split / Upload

- `--train-ratio` and `--test-ratio`: must each be in `[0, 1]` and sum to `1.0`.
- `--seed`: global seed used for reproducibility and per-sample seed derivation.
- `--shard-size`: number of samples per shard directory.
- `--max-shards`: limit work to the first N shards.
- `--resume`: continue a previous sharded run using `run_manifest.json`.
- `--upload`: upload after generation completes.

Local-first behavior:

- `generate` does not require `--repo-id` unless you also pass `--upload`.
- If you do provide `--repo-id`, it is stored in `run_manifest.json` so `publish` can reuse it later.

### Publish Command

- `uv run main.py publish --generated-path <path>` uploads a previously generated dataset root.
- `publish` reads generation context from `run_manifest.json`, so the dataset card and split settings do not need to be re-entered.
- `--repo-id` is optional on `publish` when the manifest already contains one, but can still be used to override it.

## Pipeline Workflow (Detailed)

1. Configuration and seed setup

- CLI options are parsed and forwarded through `pipeline.py` into generator runtime config.
- If `--seed` is set, global and per-sample RNG paths become reproducible.

2. Template catalog load

- YAML templates from `--template-config-dir` (or default directory) are merged by `id`.
- If no catalog entries are found, a built-in `sections` fallback template is used.
- Aliases are normalized (`-` and spaces become `_`) for robust matching.

3. Candidate resolution

- `--template` exact match wins when resolvable.
- Otherwise candidate pool is filtered by family and complexity bounds.

4. Weighted template sampling

- Sampling weight starts from template `weight`.
- Runtime balancing factors down-weight overused templates/families.
- Coverage targets can boost underrepresented families.

5. Content generation

- Sections mode composes markdown from the template `blueprint`.
- Rich block composition is used when the blueprint defines `document_shape`, `shape`, `section_count`, `blocks_per_section`, `allowed_blocks`, or `required_blocks`.
- Compatibility section blueprints without those rich block keys combine `text`, `table`, and `formula` subconfigs.
- Supported block types: `paragraph`, `bullet_list`, `numbered_list`, `checklist`, `table`, `formula`, `quote`, `code`, `command`, `image`, `rule`.
- The `text` block alias is accepted and normalized to `paragraph`.

6. Novelty guard

- A structure signature is built from markdown line types (`h1`, `table`, `code`, `ul`, `ol`, etc.).
- Signature similarity is compared against recent window history.
- If similarity is too high, generation retries up to `--novelty-max-attempts`.

7. Rendering and effects

- Style is sampled from base presets and perturbed according to `--style-profile`.
- Noise/blur probabilities are then applied.
- Markdown is rendered by `pil`, `html2image`, or headless `playwright` backend.
- Formula rasterization uses an in-process bounded cache to avoid unbounded memory growth in long runs.
- The current markdown image path records final image dimensions, but no hard A4 rescale is applied by `_fit_image_to_a4()` at the moment.

8. Metadata, shards, and publish

- Each sample writes rich metadata and `GT_markdown` / `GT_json`.
- Each shard writes its own `metadata.jsonl` and `_SUCCESS` marker under `shards/shard-XXXXXX/`.
- A top-level `run_manifest.json` tracks shard progress and stores the publish context.
- After shard completion, root `metadata.jsonl` and `realism_stats.json` are rebuilt from shard outputs.
- `--resume` skips shards already marked completed in the manifest when the shard `_SUCCESS` marker is present.
- Upload happens only when you run `publish` or pass `--upload`.

## Template Catalog Format

Catalog files live under `configs/generator/templates/*.yaml` by default.

Minimal example:

```yaml
templates:
  - id: business_report
    family: business
    complexity: 2
    weight: 1.25
    mode: sections
    aliases: [report, business-report]
    version: "3"
    blueprint:
      document_shape: business_report
      section_count: [4, 7]
      blocks_per_section: [1, 2]
      allowed_blocks: [paragraph, bullet_list, table, quote, rule]
      required_blocks: [paragraph, table]
      table:
        rows: [2, 5]
        columns: [3, 5]
```

Template fields:

- `id`: required unique key.
- `family`: grouping label used by `--template-family` and `--coverage-target`.
- `complexity`: integer complexity (internally clamped to `1..5`).
- `weight`: base sampling weight (internally floored to `0.01`).
- `mode`: supported block-composition mode is `sections`; unsupported values are coerced to `sections` with a warning.
- `aliases`: optional alternate names for CLI selection.
- `version`: template-level version label stored in metadata.
- `blueprint`: section and block composition rules used by `mode: sections`.
- `blueprint.document_shape`: shape label stored in metadata.
- `blueprint.section_count`: section count or `[min, max]` range.
- `blueprint.blocks_per_section`: block count or `[min, max]` range for each section.
- `blueprint.allowed_blocks`: block types eligible for sampling.
- `blueprint.required_blocks`: block types that must appear when possible.
- `blueprint.table.rows` / `blueprint.table.columns`: table size ranges.
- Compatibility section blueprints without rich block keys can also use `blueprint.text.section_count`, `blueprint.text.max_line_chars`, `blueprint.table.section_count`, and `blueprint.formula.section_count`.

Component note:

- `formula` emits display-style markdown formula lines (for example `$$ E = mc^2 $$`).
- `image` emits markdown image placeholders (`![alt](placeholder://...)`) and renderers draw stable placeholder boxes without external image downloads.

## Practical Recipes

### Structure-diverse documents

The default template catalog now includes multiple document families such as `business`, `technical`, `academic`, `operations`, `forms`, and legacy-compatible `sections`.

Use coverage targets when you want family balance:

```bash
uv run main.py generate \
  --lang "ko" \
  --size 1000 \
  --style-profile aggressive \
  --coverage-target business=0.2 \
  --coverage-target technical=0.25 \
  --coverage-target academic=0.2 \
  --coverage-target operations=0.15 \
  --coverage-target forms=0.2 \
  --novelty-threshold 0.92 \
  --novelty-max-attempts 6
```

Each sample records `document_family`, `document_shape`, `block_types`, `block_type_counts`, and `section_count` in metadata so generated diversity can be audited from `metadata.jsonl`.

### 1) Shorter documents

```bash
uv run main.py generate \
  --lang "ko" \
  --size 500 \
  --template-family forms \
  --max-template-complexity 2 \
  --style-profile balanced
```

### 2) Family-balanced distribution with controls

```bash
uv run main.py generate \
  --lang "ko" \
  --size 2000 \
  --min-template-complexity 2 \
  --max-template-complexity 4 \
  --coverage-target operations=0.3 \
  --coverage-target technical=0.3 \
  --coverage-target business=0.2 \
  --coverage-target forms=0.2 \
  --novelty-window 120 \
  --novelty-threshold 0.92 \
  --novelty-max-attempts 5
```

### 3) Fixed template for ablation / debugging

```bash
uv run main.py generate \
  --lang "ko" \
  --size 100 \
  --template technical_manual \
  --seed 42
```

### 4) Sharded local generation

```bash
uv run main.py generate \
  --lang "ko" \
  --size 1000 \
  --shard-size 250
```

### 5) Resume a partial sharded run

```bash
uv run main.py generate \
  --lang "ko" \
  --size 1000 \
  --shard-size 250 \
  --resume
```

### 6) Publish a finished local run

```bash
uv run main.py publish \
  --generated-path "./data/ko/images_markdown" \
  --repo-id "username/my-ocr-dataset"
```

Publish detail:

- Generation is markdown-focused.
- Records are shuffled with a fixed seed before split, so split assignment is deterministic for identical metadata ordering.
- `publish` requires `--repo-id` only when the manifest does not already contain one.

## Output Artifacts

Local output root:

- `<output-dir>/<lang>/images_markdown`

Core files:

- `run_manifest.json`
- `metadata.jsonl`
- `realism_stats.json`
- `shards/shard-000000/metadata.jsonl`
- `shards/shard-000000/_SUCCESS`
- generated image files within each shard directory (`shards/shard-000000/markdown_00000.png`, ...)

Per-sample metadata includes:

- template tracing: `template`, `template_id`, `template_family`, `template_complexity`, `template_mode`, `template_version`, `template_source`, `template_weight`
- GT fields: `GT_markdown`, `GT_json`
- document structure: `document_family`, `document_shape`, `block_types`, `block_type_counts`, `section_count`
- diversity trace: `selection_attempt`, `structure_signature`, `novelty_score`, `family_ratio`
- render trace: `renderer`, `style_profile`, `similar_char_mutations`, `image_width`, `image_height`
- reproducibility trace: `sample_index`, `sample_seed`
- path: `file_name`

## Helper Script (`scripts/synthesize/generate.sh`)

Wrapper usage example:

```bash
bash scripts/synthesize/generate.sh \
  --repo-id "username/my-ocr-dataset" \
  --lang "ko" \
  --size 1000 \
  --style-profile balanced \
  --shard-size 250 \
  --resume
```

Notes:

- Wrapper default renderer is `playwright`.
- Some language shortcut scripts, including `scripts/synthesize/lang/ko.sh`, run fixed presets and do not forward additional user arguments.
- Advanced template, family, and coverage controls such as `--template-family` or `--coverage-target` should use `uv run main.py generate` directly.

Playwright notes:

- The Playwright renderer launches Chromium in headless mode.
- Install browser binaries once with `uv run playwright install chromium` after syncing dependencies.
- Wrapper now runs `generate --upload` explicitly, so upload remains opt-in at the CLI level.
- The generic wrapper forwards its supported shard and resume flags to `main.py generate`.

## Character Similarity Database

Build a language-specific DB:

```bash
./scripts/synthesize/generate_similarity_db.sh --lang ko
```

Generate corpus text first and then build the DB in one pass:

```bash
./scripts/synthesize/generate_similarity_db.sh \
  --lang ko \
  --generate-corpus \
  --corpus-provider openai \
  --corpus-count 1000
```

Useful options:

- `--font-path`: override font used for similarity extraction.
- `--corpus-path`: override the final merged corpus file.
- `--generate-corpus`: run `main.py corpus generate`, merge category files into one corpus text, then build the DB.
- `--corpus-provider`, `--corpus-model`: choose the LLM backend for corpus generation.
- `--corpus-count`, `--corpus-batch-size`: control how much corpus text is generated before merging.
- `--corpus-category`: restrict LLM corpus generation to specific categories.
- `--auto-generate-corpus`: generate `corpus_<lang>.txt` from Wikimedia if missing.
- `--corpus-sentences`: number of sentences when auto-generating corpus (default: `100000`).
- `--db-path`: output JSON path override.
- `--threshold`: similarity threshold (default: `0.6`).
- `--top-k`: max similar characters per character (default: `8`).

## Troubleshooting

- Templates look too long: reduce `--max-template-complexity` and/or select shorter families such as `forms`.
- Keep pages under A4 more aggressively: lower `section_count` and `blocks_per_section` in `sections` templates, and reduce table rows/columns or `text.max_line_chars` as needed.
- Coverage targets seem ignored: ensure family names match catalog `family` values.
- Novelty retry feels heavy: lower `--novelty-threshold` or reduce `--novelty-max-attempts`.
- `sample_seed` is `null`: expected when `--seed` is not provided.
