#!/bin/bash

set -e

usage() {
    echo "Usage: $0 --repo-id <repo> --lang <code> [options]"
    echo ""
    echo "Options:"
    echo "  --size <n>            Number of markdown samples (default: 1000)"
    echo "  --similar-char-ratio <r> Similar character replacement ratio (default: 0.08)"
    echo "  --formula-source-mode <m> Formula source mode: mixed|dataset|random|synthetic (default: mixed)"
    echo "  --formula-dataset-path <path> Formula dataset file path (.txt/.json/.jsonl/.csv/.tsv)"
    echo "  --formula-dataset-weight <w> Formula dataset weight when --formula-source-mode=mixed (default: 0.45)"
    echo "  --formula-random-weight <w> Formula random weight when --formula-source-mode=mixed (default: 0.30)"
    echo "  --formula-synthetic-weight <w> Formula synthetic weight when --formula-source-mode=mixed (default: 0.25)"
    echo "  --text-section-count <min,max> Text section count range (default: 3,5)"
    echo "  --table-section-count <min,max> Table section count range (default: 1,2)"
    echo "  --table-rows <min,max> Table row range (default: 2,4)"
    echo "  --table-columns <min,max> Table column range (default: 3,5)"
    echo "  --formula-section-count <min,max> Formula section count range (default: 1,2)"
    echo "  --template-config-dir <dir> Use an explicit template config directory (optional)"
    echo "  --style-profile <p>   Style profile: legacy|balanced|aggressive (default: balanced)"
    echo "  --novelty-window <n>  Novelty guard window size (default: 80)"
    echo "  --novelty-threshold <r> Novelty similarity threshold (default: 0.95)"
    echo "  --novelty-max-attempts <n> Novelty retry count (default: 4)"
    echo "  --markdown-renderer <name> Renderer: pil|html2image|playwright (default: playwright)"
    echo "  --train-ratio <r>     Train split ratio for dataset publishing (default: 0.9)"
    echo "  --test-ratio <r>      Test split ratio for dataset publishing (default: 0.1)"
    echo "  --shard-size <n>      Samples per shard directory (optional)"
    echo "  --max-shards <n>      Limit generation to the first N shards (optional)"
    echo "  --resume              Resume a previous sharded run"
    echo "  --label <label>       Display label for logs (optional)"
    echo "  --repo-id <repo>      Hugging Face dataset repo id (required)"
    echo "  --lang <code>         Language code (required)"
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ID=""
LANG=""
SIZE=1000
LABEL=""
SIMILAR_CHAR_RATIO=0.08
TRAIN_RATIO=0.9
TEST_RATIO=0.1
TEMPLATE_CONFIG_DIR=""
STYLE_PROFILE="balanced"
NOVELTY_WINDOW=80
NOVELTY_THRESHOLD=0.95
NOVELTY_MAX_ATTEMPTS=4
MARKDOWN_RENDERER="playwright"
TEXT_SECTION_COUNT="1,2"
TABLE_SECTION_COUNT="0,1"
TABLE_ROWS="2,4"
TABLE_COLUMNS="3,5"
FORMULA_SECTION_COUNT="0,1"
FORMULA_SOURCE_MODE="mixed"
FORMULA_DATASET_PATH=""
FORMULA_DATASET_WEIGHT=0.45
FORMULA_RANDOM_WEIGHT=0.30
FORMULA_SYNTHETIC_WEIGHT=0.25
SHARD_SIZE=""
MAX_SHARDS=""
RESUME=0

parse_range() {
    local raw="$1"
    local label="$2"

    if [[ ! "$raw" =~ ^[0-9]+,[0-9]+$ ]]; then
        echo "Invalid ${label}: '$raw' (expected: min,max)" >&2
        exit 1
    fi

    local left="${raw%,*}"
    local right="${raw#*,}"

    if (( left <= right )); then
        echo "$left, $right"
    else
        echo "$right, $left"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)
            REPO_ID="$2"
            shift 2
            ;;
        --lang)
            LANG="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --similar-char-ratio)
            SIMILAR_CHAR_RATIO="$2"
            shift 2
            ;;
        --formula-source-mode)
            FORMULA_SOURCE_MODE="$2"
            shift 2
            ;;
        --formula-dataset-path)
            FORMULA_DATASET_PATH="$2"
            shift 2
            ;;
        --formula-dataset-weight)
            FORMULA_DATASET_WEIGHT="$2"
            shift 2
            ;;
        --formula-random-weight)
            FORMULA_RANDOM_WEIGHT="$2"
            shift 2
            ;;
        --formula-synthetic-weight)
            FORMULA_SYNTHETIC_WEIGHT="$2"
            shift 2
            ;;
        --text-section-count)
            TEXT_SECTION_COUNT="$2"
            shift 2
            ;;
        --table-section-count)
            TABLE_SECTION_COUNT="$2"
            shift 2
            ;;
        --table-rows)
            TABLE_ROWS="$2"
            shift 2
            ;;
        --table-columns)
            TABLE_COLUMNS="$2"
            shift 2
            ;;
        --formula-section-count)
            FORMULA_SECTION_COUNT="$2"
            shift 2
            ;;
        --template-config-dir)
            TEMPLATE_CONFIG_DIR="$2"
            shift 2
            ;;
        --style-profile)
            STYLE_PROFILE="$2"
            shift 2
            ;;
        --novelty-window)
            NOVELTY_WINDOW="$2"
            shift 2
            ;;
        --novelty-threshold)
            NOVELTY_THRESHOLD="$2"
            shift 2
            ;;
        --novelty-max-attempts)
            NOVELTY_MAX_ATTEMPTS="$2"
            shift 2
            ;;
        --markdown-renderer)
            MARKDOWN_RENDERER="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --test-ratio)
            TEST_RATIO="$2"
            shift 2
            ;;
        --shard-size)
            SHARD_SIZE="$2"
            shift 2
            ;;
        --max-shards)
            MAX_SHARDS="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$REPO_ID" || -z "$LANG" ]]; then
    usage
fi

if [[ -z "$LABEL" ]]; then
    LABEL="$REPO_ID"
fi

TEXT_SECTION_RANGE="$(parse_range "$TEXT_SECTION_COUNT" "text-section-count")"
TABLE_SECTION_RANGE="$(parse_range "$TABLE_SECTION_COUNT" "table-section-count")"
TABLE_ROWS_RANGE="$(parse_range "$TABLE_ROWS" "table-rows")"
TABLE_COLUMNS_RANGE="$(parse_range "$TABLE_COLUMNS" "table-columns")"
FORMULA_SECTION_RANGE="$(parse_range "$FORMULA_SECTION_COUNT" "formula-section-count")"

GENERATED_TEMPLATE_DIR=""
if [[ -z "$TEMPLATE_CONFIG_DIR" ]]; then
    GENERATED_TEMPLATE_DIR="$(mktemp -d "/tmp/synthetic-ocr-sections-XXXXXX")"
    TEMPLATE_CONFIG_DIR="$GENERATED_TEMPLATE_DIR"
    trap 'rm -rf "$GENERATED_TEMPLATE_DIR"' EXIT
    cat > "$TEMPLATE_CONFIG_DIR/default.yaml" <<YAML
version: 2
id: default
mode: sections

text:
  section_count: [${TEXT_SECTION_RANGE}]

table:
  section_count: [${TABLE_SECTION_RANGE}]
  rows: [${TABLE_ROWS_RANGE}]
  columns: [${TABLE_COLUMNS_RANGE}]

formula:
  section_count: [${FORMULA_SECTION_RANGE}]
YAML
fi

echo "=========================================="
echo "Generating OCR images: $LABEL"
echo "=========================================="

echo ""
echo "[1/1] Generating and uploading split-aware dataset (train/test splits)..."

CMD=(
    uv run --no-sync --group generate main.py generate
    --repo-id "$REPO_ID"
    --size "$SIZE"
    --lang "$LANG"
    --similar-char-ratio "$SIMILAR_CHAR_RATIO"
    --train-ratio "$TRAIN_RATIO"
    --test-ratio "$TEST_RATIO"
    --upload
    --markdown-renderer "$MARKDOWN_RENDERER"
    --style-profile "$STYLE_PROFILE"
    --novelty-window "$NOVELTY_WINDOW"
    --novelty-threshold "$NOVELTY_THRESHOLD"
    --novelty-max-attempts "$NOVELTY_MAX_ATTEMPTS"
    --formula-source-mode "$FORMULA_SOURCE_MODE"
    --formula-dataset-weight "$FORMULA_DATASET_WEIGHT"
    --formula-random-weight "$FORMULA_RANDOM_WEIGHT"
    --formula-synthetic-weight "$FORMULA_SYNTHETIC_WEIGHT"
    --template-config-dir "$TEMPLATE_CONFIG_DIR"
)

if [[ -n "$FORMULA_DATASET_PATH" ]]; then
    CMD+=(--formula-dataset-path "$FORMULA_DATASET_PATH")
fi

if [[ -n "$SHARD_SIZE" ]]; then
    CMD+=(--shard-size "$SHARD_SIZE")
fi

if [[ -n "$MAX_SHARDS" ]]; then
    CMD+=(--max-shards "$MAX_SHARDS")
fi

if [[ "$RESUME" -eq 1 ]]; then
    CMD+=(--resume)
fi

"${CMD[@]}"

echo ""
echo "=========================================="
echo "Dataset generated!"
echo "Dataset: https://huggingface.co/datasets/$REPO_ID"
echo "=========================================="
