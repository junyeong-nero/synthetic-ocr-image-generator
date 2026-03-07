#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 --lang <code> [--lang <code> ...] [options]"
    echo ""
    echo "Options:"
    echo "  --lang <code>             Language code (repeatable, required)"
    echo "  --all                     Generate corpus for all scripts/synthesize/lang/*.sh"
    echo "  --provider <name>         LLM provider: openai|anthropic (default: openai)"
    echo "  --model <name>            Provider-specific model override"
    echo "  --lang-name <name>        Optional language name hint for custom codes"
    echo "  --count <n>               Items to generate per category (default: 1000)"
    echo "  --batch-size <n>          Items requested per API call (default: 100)"
    echo "  --output-dir <path>       Corpus output directory (default: data/corpus)"
    echo "  --category <name>         Corpus category to generate (repeatable, default: all)"
    echo "  --dry-run                 Print the resolved commands without running them"
    echo "  -h, --help                Show this help message"
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LANG_DIR="$SCRIPT_DIR/lang"

LANGS=()
USE_ALL=false
PROVIDER="openai"
MODEL=""
LANG_NAME=""
COUNT="1000"
BATCH_SIZE="100"
OUTPUT_DIR="$PROJECT_DIR/data/corpus"
CATEGORIES=()
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lang)
            LANGS+=("$2")
            shift 2
            ;;
        --all)
            USE_ALL=true
            shift
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --lang-name)
            LANG_NAME="$2"
            shift 2
            ;;
        --count)
            COUNT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --category)
            CATEGORIES+=("$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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

if [[ "$USE_ALL" == true ]]; then
    if [[ ! -d "$LANG_DIR" ]]; then
        echo "Error: language script directory not found: $LANG_DIR"
        exit 1
    fi
    while IFS= read -r path; do
        LANGS+=("$(basename "$path" .sh)")
    done < <(ls "$LANG_DIR"/*.sh 2>/dev/null | sort)
fi

if [[ ${#LANGS[@]} -eq 0 ]]; then
    echo "Error: at least one --lang or --all is required."
    usage
fi

mkdir -p "$OUTPUT_DIR"

for lang in "${LANGS[@]}"; do
    lang="$(printf '%s' "$lang" | tr '[:upper:]' '[:lower:]')"

    cmd=(
        uv run --no-sync main.py corpus generate
        --lang "$lang"
        --provider "$PROVIDER"
        --count "$COUNT"
        --batch-size "$BATCH_SIZE"
        --output-dir "$OUTPUT_DIR"
    )

    if [[ -n "$MODEL" ]]; then
        cmd+=(--model "$MODEL")
    fi

    if [[ -n "$LANG_NAME" ]]; then
        cmd+=(--lang-name "$LANG_NAME")
    fi

    if [[ ${#CATEGORIES[@]} -gt 0 ]]; then
        for category in "${CATEGORIES[@]}"; do
            cmd+=(--category "$category")
        done
    fi

    echo "=========================================="
    echo "Generating corpus: $lang"
    echo "Output dir: $OUTPUT_DIR/$lang"
    echo "Provider: $PROVIDER"
    echo "Count/category: $COUNT"
    echo "=========================================="

    if [[ "$DRY_RUN" == true ]]; then
        printf '[dry-run] Would run:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi

    (
        cd "$PROJECT_DIR"
        "${cmd[@]}"
    )

done

echo "Done."
