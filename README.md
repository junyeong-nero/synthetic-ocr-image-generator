# Synthetic OCR Image Generator and Benchmark

End-to-end toolkit for generating synthetic OCR datasets and benchmarking vision-language models with a markdown-first pipeline.

## What This Does

This repo lets you:

1. **Generate corpus text** — build reusable language-specific text assets with an LLM provider (OpenAI, Anthropic, or Wikimedia auto-crawl).
2. **Generate synthetic OCR images** — render markdown documents as images with controllable noise, blur, and visual style variation, written to a local dataset root.
3. **Publish to Hugging Face Hub** — upload a completed local generation run as a HF dataset.
4. **Evaluate OCR/VLM models** — run models against generated datasets, compute markdown-aware metrics, and produce reproducible reports and leaderboards.

## How It Works

```
corpus text  →  generate markdown documents  →  render as images  →  evaluate OCR models
(LLM/Wikimedia)   (templates + diversity controls)   (Playwright/PIL)   (metrics + reports)
```

### Generation Pipeline (A/B/C phases)

| Phase | What it does |
|---|---|
| **A — Legacy** | Classic template methods (`readme`, `tutorial`, …) managed via YAML catalog |
| **B — Blueprint** | Dynamic document structures defined in `configs/generator/templates/*.yaml` |
| **C — Quality** | Novelty guard + family coverage balancing + style profiles for diverse, high-quality outputs |

Each generated image is paired with `GT_markdown` and `GT_json` ground truth, making the dataset directly usable for OCR model training and evaluation.

### Evaluation Pipeline

Loads a model config YAML → runs inference via the selected backend → computes markdown block metrics → writes JSON/Markdown/HTML reports + leaderboard files.

---

## Installation

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/your-repo/synthetic-ocr-image-generator.git
cd synthetic-ocr-image-generator

uv sync
uv run playwright install chromium   # required for the default Playwright renderer
```

### Optional: Formula Rendering

LaTeX formula rendering requires XeLaTeX (part of [MacTeX](https://tug.org/mactex/) on macOS):

```bash
xelatex --help   # verify installation
```

---

## Quick Start

### Step 1 — Generate corpus text

Corpus text is fed into document templates to produce realistic content. Skip this step if you only want to use placeholder text.

```bash
uv run main.py corpus generate \
  --lang "ko" \
  --provider openai \
  --count 1000
```

### Step 2 — Generate a local dataset

```bash
uv run main.py generate \
  --lang "ko" \
  --size 1000 \
  --shard-size 250
```

Output is written to `./data/ko/images_markdown/` with:
- `run_manifest.json` — tracks shard progress and publish context
- `metadata.jsonl` — aggregate metadata for all samples
- `realism_stats.json` — image realism statistics
- `shards/shard-000000/` — shard directories with images and per-shard `metadata.jsonl`

### Step 3 — Publish to Hugging Face Hub

```bash
uv run main.py publish \
  --generated-path "./data/ko/images_markdown" \
  --repo-id "your-username/my-ocr-dataset"
```

`publish` reads generation context from `run_manifest.json`, so you only need `--repo-id` if it was not set during generation.

### Step 4 — Evaluate a model

```bash
uv run main.py evaluate \
  --model-config configs/models/gpt-5-mini.yaml \
  --dataset "your-username/my-ocr-dataset" \
  --split train
```

### Step 5 — Compare evaluation reports

```bash
uv run main.py compare \
  evaluation_result/model_a/report.json \
  evaluation_result/model_b/report.json \
  -o comparison_results
```

---

## Evaluation Metrics

All metrics are markdown-aware and computed per-block:

| Metric | What it measures |
|---|---|
| `avg_markdown_text_score` | Plain text accuracy |
| `avg_markdown_table_teds` | Table structure accuracy (Tree Edit Distance Score) |
| `avg_markdown_formula_score` | Formula rendering accuracy |
| `avg_markdown_order_score` | Block ordering accuracy |
| `avg_markdown_overall_score` | **Primary metric** used in leaderboards |

---

## Leaderboards

### Korean OCR (ko)

| Rank | Model | Backend | Overall | Text | Table | Formula | Success |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | lightonai/LightOnOCR-2-1B | transformers | 0.9737 | 0.9549 | 1.0000 | 0.9437 | 100/100 |
| 2 | ./weights/DotsOCR | transformers | 0.9464 | 0.9177 | 0.9874 | 0.9004 | 100/100 |
| 3 | deepseek-ai/DeepSeek-OCR-2 | transformers | 0.9461 | 0.9376 | 0.9991 | 0.8719 | 100/100 |
| 4 | nanonets/Nanonets-OCR2-3B | transformers | 0.9201 | 0.9025 | 0.9988 | 0.8341 | 100/100 |
| 5 | Qwen/Qwen3-VL-4B-Instruct | transformers | 0.8639 | 0.7141 | 1.0000 | 0.8860 | 100/100 |

### Japanese OCR (ja)

| Rank | Model | Backend | Overall | Text | Table | Formula | Success |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | lightonai/LightOnOCR-2-1B | transformers | 0.9777 | 0.9682 | 0.9995 | 0.9458 | 100/100 |
| 2 | nanonets/Nanonets-OCR2-3B | transformers | 0.9605 | 0.9700 | 1.0000 | 0.8871 | 100/100 |
| 3 | ./weights/DotsOCR | transformers | 0.9288 | 0.8884 | 0.9923 | 0.9084 | 100/100 |
| 4 | deepseek-ai/DeepSeek-OCR-2 | transformers | 0.9141 | 0.8252 | 0.9847 | 0.8794 | 100/100 |
| 5 | Qwen/Qwen3-VL-4B-Instruct | transformers | 0.8641 | 0.7234 | 0.9934 | 0.8876 | 100/100 |

Refresh leaderboard files:

```bash
bash scripts/evaluate/update-leaderboard.sh
```

---

## Adding a New Model

1. Copy `configs/models/_template.yaml` and fill in `model_id`, `backend`, and `prompt.prompt`.
2. Run a quick smoke test:
   ```bash
   uv run main.py evaluate \
     --model-config configs/models/your-model.yaml \
     --dataset your-username/my-ocr-dataset \
     --max-samples 5
   ```
3. Run the full evaluation via the script wrapper:
   ```bash
   scripts/evaluate/run.sh your-model -d your-username/my-ocr-dataset -n 100
   ```

Supported backends: `openai`, `anthropic`, `google`, `upstage`, `transformers`, `paddleocr`, `surya`.

---

## Script Wrappers

Recommended shell wrappers for common workflows:

| Script | Purpose |
|---|---|
| `scripts/synthesize/generate.sh` | Dataset generation with sensible defaults |
| `scripts/evaluate/run.sh` | Single model evaluation with dependency-group handling |
| `scripts/evaluate/run-all.sh` | Batch evaluation for all configs under `configs/models/` |
| `scripts/evaluate/update-leaderboard.sh` | Regenerate leaderboard files from existing results |

Example — evaluate all configs against a dataset:

```bash
scripts/evaluate/run-all.sh -d your-username/my-ocr-dataset --language ko -n 200
```

---

## Project Structure

```
main.py                        CLI entrypoint
src/
  pipeline.py                  Generation and publish orchestration
  cli/                         CLI command definitions
  corpus_generator.py          LLM-backed corpus generation
  generator/                   Image generation, rendering, noise/blur effects
  evaluation/                  Evaluation orchestration, runner, checkpointing
  metrics/                     Metric implementations
configs/
  models/                      Model config YAML files
  generator/templates/         Blueprint template YAML files
scripts/
  synthesize/                  Dataset generation helpers
  evaluate/                    Evaluation and leaderboard helpers
fonts/                         Language-specific font files
docs/                          Detailed documentation
```

---

## Documentation

- [`docs/overview.md`](docs/overview.md) — architecture and typical workflow
- [`docs/generation.md`](docs/generation.md) — generation pipeline, templates, options, recipes
- [`docs/evaluation.md`](docs/evaluation.md) — evaluation pipeline and script wrappers
- [`docs/model-configs.md`](docs/model-configs.md) — model YAML config reference
- [`docs/metrics.md`](docs/metrics.md) — metric definitions
- [`docs/cli.md`](docs/cli.md) — full CLI reference
- [`docs/benchmark-protocol.md`](docs/benchmark-protocol.md) — benchmark reproducibility protocol

---

## Contributing

See [`AGENTS.md`](AGENTS.md) for repository conventions.
