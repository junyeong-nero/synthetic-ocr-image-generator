# Data Generation Guide

The `generate` command creates synthetic markdown OCR samples, writes local artifacts, uploads splits to Hugging Face Hub, and publishes a dataset card with reproducible run metadata.

## Quick Start

```bash
uv run main.py generate \
  --repo-id "username/my-ocr-dataset" \
  --lang "ko" \
  --size 100
```

Required:

- `--repo-id`: Hugging Face dataset repository ID.

Core defaults:

- `--lang ko`
- `--size 100`
- `--output-dir ./data`
- `--markdown-renderer pil`
- `--style-profile balanced`
- `--novelty-window 80`
- `--novelty-threshold 0.95`
- `--novelty-max-attempts 4`
- `--similar-char-ratio 0.08`

## Mental Model (A/B/C)

Generation now follows a three-layer model:

- Phase A (compatibility): legacy templates still work (e.g., `readme`, `tutorial`) but are managed through the same catalog system.
- Phase B (dynamic templates): blueprint-driven templates are loaded from YAML and sampled at runtime.
- Phase C (quality controls): coverage balancing + novelty guard + style variation produce diverse but controllable outputs.

Important distinction:

- `--template` / `--template-family` / complexity filters control which template is used.
- `--style-profile` controls rendering variability only; it does not force legacy or dynamic templates.

## Option Reference

### Template Selection

- `--template`: Exact template id or alias. If resolved, it takes precedence over family/complexity filters.
- `--template-family`: Family filter for random sampling (examples: `legacy`, `operations`, `api`, `incident`, `compliance`, `release`, `procedural`).
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

- `--markdown-renderer`: `pil` or `html2image`.
- `--style-profile`: `legacy`, `balanced`, `aggressive`.
- `--similar-char-ratio`: proportion of characters replaced with lookalikes.
- `--similarity-db-path`: explicit JSON path for similarity DB lookup.
- `--add-noise` / `--no-add-noise`: explicit noise override.
- `--add-blur` / `--no-add-blur`: explicit blur override.

### Dataset Split / Upload

- `--mixed`: enables train/test upload flow.
- `--train-ratio` and `--test-ratio`: must each be in `[0, 1]` and sum to `1.0`.
- `--seed`: global seed used for reproducibility and per-sample seed derivation.

## Pipeline Workflow (Detailed)

1. Configuration and seed setup

- CLI options are parsed and forwarded through `pipeline.py` into generator runtime config.
- If `--seed` is set, global and per-sample RNG paths become reproducible.

2. Template catalog load

- Built-in legacy specs are loaded first.
- YAML templates from `--template-config-dir` (or default directory) are merged by `id`.
- Aliases are normalized (`-` and spaces become `_`) for robust matching.

3. Candidate resolution

- `--template` exact match wins when resolvable.
- Otherwise candidate pool is filtered by family and complexity bounds.

4. Weighted template sampling

- Sampling weight starts from template `weight`.
- Runtime balancing factors down-weight overused templates/families.
- Coverage targets can boost underrepresented families.

5. Content generation

- Legacy mode: calls procedural template methods.
- Blueprint mode: assembles sections/blocks from blueprint ranges and block rules.
- Supported blueprint block types: `title`, `subtitle`, `contents`, `bullet_points`, `numbered_list`, `checklist`, `table`, `formula`, `image`, `code`, `quote`, `rule`, `command`.
- Backward-compatible aliases are accepted (for example: `bullet_list -> bullet_points`, `toc -> contents`, `equation -> formula`, `figure -> image`).

6. Novelty guard

- A structure signature is built from markdown line types (`h1`, `table`, `code`, `ul`, `ol`, etc.).
- Signature similarity is compared against recent window history.
- If similarity is too high, generation retries up to `--novelty-max-attempts`.

7. Rendering and effects

- Style is sampled from base presets and perturbed according to `--style-profile`.
- Noise/blur probabilities are then applied.
- Markdown is rendered by `pil` or `html2image` backend.
- A final size guard rescales oversized renders to stay within A4 bounds (`2480x3508` max).

8. Metadata and upload

- Each sample writes rich metadata and `GT_markdown` / `GT_json`.
- Local `metadata.jsonl` is saved.
- Data uploads to Hugging Face splits.
- Dataset `README.md` is generated and uploaded with generation settings and reproducible command.

## Template Catalog Format

Catalog files live under `configs/generator/templates/*.yaml` by default.

Minimal example:

```yaml
version: 1
templates:
  - id: dynamic_ops_brief
    family: operations
    complexity: 2
    weight: 1.25
    mode: blueprint
    aliases: [ops-brief]
    blueprint:
      title_prefix: Ops Brief
      section_count: [2, 4]
      paragraphs_per_section: [1, 1]
      blocks_per_section: [1, 2]
      max_total_lines: 95
      max_paragraph_chars: 220
      allowed_blocks: [subtitle, contents, bullet_points, table, formula, image, quote]
      required_blocks: [contents, table, formula, image]
```

Template fields:

- `id`: required unique key.
- `family`: grouping label used by `--template-family` and `--coverage-target`.
- `complexity`: integer complexity (internally clamped to `1..5`).
- `weight`: base sampling weight (internally floored to `0.01`).
- `mode`: `legacy` or `blueprint` (`dynamic`/`procedural` aliases are accepted and normalized to `blueprint`).
- `legacy_method`: method name for legacy templates (defaults to `id` if omitted).
- `aliases`: optional alternate names for CLI selection.
- `version`: template-level version label stored in metadata.
- `blueprint`: generation rules used when `mode: blueprint`.
- `blueprint.max_total_lines`: hard cap for generated markdown line count.
- `blueprint.max_paragraph_chars`: clipping threshold for long generated paragraphs.

Component note:

- `formula` emits display-style markdown formula lines (for example `$$ E = mc^2 $$`).
- `image` emits markdown image placeholders (`![alt](placeholder://...)`) and renderers draw stable placeholder boxes without external image downloads.

## Practical Recipes

### 1) Shorter documents

```bash
uv run main.py generate \
  --repo-id "username/my-ocr-dataset" \
  --lang "en" \
  --size 500 \
  --template-family procedural \
  --max-template-complexity 1 \
  --style-profile balanced
```

### 2) Dynamic-heavy distribution with controls

```bash
uv run main.py generate \
  --repo-id "username/my-ocr-dataset" \
  --lang "ko" \
  --size 2000 \
  --template-family operations \
  --min-template-complexity 2 \
  --max-template-complexity 4 \
  --coverage-target operations=0.6 \
  --coverage-target legacy=0.2 \
  --coverage-target incident=0.2 \
  --novelty-window 120 \
  --novelty-threshold 0.92 \
  --novelty-max-attempts 5
```

### 3) Fixed template for ablation / debugging

```bash
uv run main.py generate \
  --repo-id "username/my-ocr-dataset" \
  --lang "en" \
  --size 100 \
  --template dynamic_general_notes \
  --seed 42
```

### 4) Mixed split upload

```bash
uv run main.py generate \
  --repo-id "username/my-ocr-dataset" \
  --lang "ko" \
  --size 1000 \
  --mixed \
  --train-ratio 0.9 \
  --test-ratio 0.1
```

Split detail:

- Mixed mode is markdown-focused.
- Records are shuffled with a fixed seed before split, so split assignment is deterministic for identical metadata ordering.

## Output Artifacts

Local output roots:

- Non-mixed: `<output-dir>/<lang>/images_markdown`
- Mixed: `<output-dir>/<lang>/images_mixed`

Core files:

- `metadata.jsonl`
- generated image files (`markdown_00000.png`, ...)

Per-sample metadata includes:

- template tracing: `template`, `template_id`, `template_family`, `template_complexity`, `template_mode`, `template_version`, `template_source`, `template_weight`
- GT fields: `GT_markdown`, `GT_json`
- diversity trace: `selection_attempt`, `structure_signature`, `novelty_score`, `family_ratio`
- render trace: `renderer`, `style_profile`, `similar_char_mutations`, `a4_scaled`, `image_width`, `image_height`
- reproducibility trace: `sample_index`, `sample_seed`
- path: `file_name`

## Helper Script (`scripts/synthesize/generate.sh`)

Wrapper usage example:

```bash
bash scripts/synthesize/lang/en.sh \
  --size 1000 \
  --template-family operations \
  --coverage-target operations=0.5 \
  --coverage-target legacy=0.5
```

Notes:

- Wrapper default renderer is `html2image`.
- Wrapper forwards all new template/diversity flags to `main.py generate`.

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

- Templates look too long: reduce `--max-template-complexity` and/or select short families such as `procedural`.
- Keep pages under A4 more aggressively: lower `max_total_lines` in blueprint templates and keep `max_paragraph_chars` around `160-220`.
- Coverage targets seem ignored: ensure family names match catalog `family` values.
- Novelty retry feels heavy: lower `--novelty-threshold` or reduce `--novelty-max-attempts`.
- `sample_seed` is `null`: expected when `--seed` is not provided.
