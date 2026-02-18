#!/bin/bash

set -e

usage() {
    echo "Usage: $0 --repo-id <repo> --lang <code> [options]"
    echo ""
    echo "Options:"
    echo "  --size <n>            Number of markdown samples (default: 1000)"
    echo "  --similar-char-ratio <r> Similar character replacement ratio (default: 0.08)"
    echo "  --template <id>       Fixed template id or alias"
    echo "  --template-family <f> Template family filter"
    echo "  --min-template-complexity <n> Minimum template complexity (1-5)"
    echo "  --max-template-complexity <n> Maximum template complexity (1-5)"
    echo "  --template-config-dir <dir> Template catalog directory"
    echo "  --style-profile <p>   Style profile: legacy|balanced|aggressive (default: balanced)"
    echo "  --coverage-target <family=ratio> Coverage target (repeatable)"
    echo "  --novelty-window <n>  Novelty guard window size (default: 80)"
    echo "  --novelty-threshold <r> Novelty similarity threshold (default: 0.95)"
    echo "  --novelty-max-attempts <n> Novelty retry count (default: 4)"
    echo "  --markdown-renderer <name> Renderer: pil|html2image (default: html2image)"
    echo "  --train-ratio <r>     Train split ratio for mixed mode (default: 0.9)"
    echo "  --test-ratio <r>      Test split ratio for mixed mode (default: 0.1)"
    echo "  --label <label>       Display label for logs (optional)"
    echo "  --repo-id <repo>      Hugging Face dataset repo id (required)"
    echo "  --lang <code>         Language code (required)"
    exit 1
}

REPO_ID=""
LANG=""
SIZE=1000
LABEL=""
SIMILAR_CHAR_RATIO=0.08
TRAIN_RATIO=0.9
TEST_RATIO=0.1
TEMPLATE=""
TEMPLATE_FAMILY=""
MIN_TEMPLATE_COMPLEXITY="1"
MAX_TEMPLATE_COMPLEXITY="3"
TEMPLATE_CONFIG_DIR=""
STYLE_PROFILE="balanced"
NOVELTY_WINDOW=80
NOVELTY_THRESHOLD=0.95
NOVELTY_MAX_ATTEMPTS=4
MARKDOWN_RENDERER="html2image"
COVERAGE_TARGETS=()

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
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --template-family)
            TEMPLATE_FAMILY="$2"
            shift 2
            ;;
        --min-template-complexity)
            MIN_TEMPLATE_COMPLEXITY="$2"
            shift 2
            ;;
        --max-template-complexity)
            MAX_TEMPLATE_COMPLEXITY="$2"
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
        --coverage-target)
            COVERAGE_TARGETS+=("$2")
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

echo "=========================================="
echo "Generating OCR images: $LABEL"
echo "=========================================="

echo ""
echo "[1/1] Generating mixed dataset (train/test splits)..."

CMD=(
    uv run --group generate main.py generate
    --repo-id "$REPO_ID"
    --size "$SIZE"
    --lang "$LANG"
    --similar-char-ratio "$SIMILAR_CHAR_RATIO"
    --mixed
    --train-ratio "$TRAIN_RATIO"
    --test-ratio "$TEST_RATIO"
    --markdown-renderer "$MARKDOWN_RENDERER"
    --style-profile "$STYLE_PROFILE"
    --novelty-window "$NOVELTY_WINDOW"
    --novelty-threshold "$NOVELTY_THRESHOLD"
    --novelty-max-attempts "$NOVELTY_MAX_ATTEMPTS"
)

if [[ -n "$TEMPLATE" ]]; then
    CMD+=(--template "$TEMPLATE")
fi

if [[ -n "$TEMPLATE_FAMILY" ]]; then
    CMD+=(--template-family "$TEMPLATE_FAMILY")
fi

if [[ -n "$MIN_TEMPLATE_COMPLEXITY" ]]; then
    CMD+=(--min-template-complexity "$MIN_TEMPLATE_COMPLEXITY")
fi

if [[ -n "$MAX_TEMPLATE_COMPLEXITY" ]]; then
    CMD+=(--max-template-complexity "$MAX_TEMPLATE_COMPLEXITY")
fi

if [[ -n "$TEMPLATE_CONFIG_DIR" ]]; then
    CMD+=(--template-config-dir "$TEMPLATE_CONFIG_DIR")
fi

if (( ${#COVERAGE_TARGETS[@]} > 0 )); then
    for target in "${COVERAGE_TARGETS[@]}"; do
        CMD+=(--coverage-target "$target")
    done
fi

"${CMD[@]}"

echo ""
echo "=========================================="
echo "Dataset generated!"
echo "Dataset: https://huggingface.co/datasets/$REPO_ID"
echo "=========================================="
