#!/bin/bash

set -euo pipefail

usage() {
    echo "Usage: $0 --lang <code> [--lang <code> ...] [options]"
    echo ""
    echo "Options:"
    echo "  --lang <code>         Language code (repeatable, required)"
    echo "  --all                 Generate DBs for all scripts/dataset/lang/*.sh"
    echo "  --font-path <path>    Font file path override for all languages"
    echo "  --corpus-path <path>  Corpus file override for all languages"
    echo "  --db-path <path>      DB output path override (single language only)"
    echo "  --auto-generate-corpus Auto-generate corpus when missing"
    echo "  --corpus-sentences <n> Number of wiki sentences for auto corpus (default: 100000)"
    echo "  --threshold <float>   Similarity threshold (default: 0.6)"
    echo "  --top-k <int>         Max similar chars per character (default: 8)"
    echo "  --dry-run             Print resolved inputs only"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Default font selection priority (when --font-path is omitted):"
    echo "  1) fonts/<lang>/NotoSans*VariableFont*.ttf"
    echo "  2) fonts/<lang>/NotoSans*.ttf"
    echo "  3) fonts/NotoSans*VariableFont*.ttf"
    echo "  4) fonts/<lang>/*.ttf"
    echo "  5) fonts/*.ttf"
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LANG_DIR="$SCRIPT_DIR/lang"

LANGS=()
USE_ALL=false
FONT_PATH=""
CORPUS_PATH=""
DB_PATH=""
AUTO_GENERATE_CORPUS=false
CORPUS_SENTENCES="100000"
THRESHOLD="0.6"
TOP_K="8"
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
        --font-path)
            FONT_PATH="$2"
            shift 2
            ;;
        --corpus-path)
            CORPUS_PATH="$2"
            shift 2
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --auto-generate-corpus)
            AUTO_GENERATE_CORPUS=true
            shift
            ;;
        --corpus-sentences)
            CORPUS_SENTENCES="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
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
        echo "Error: Language script directory not found: $LANG_DIR"
        exit 1
    fi
    while IFS= read -r path; do
        code="$(basename "$path" .sh)"
        LANGS+=("$code")
    done < <(ls "$LANG_DIR"/*.sh 2>/dev/null | sort)
fi

if [[ ${#LANGS[@]} -eq 0 ]]; then
    echo "Error: at least one --lang or --all is required."
    usage
fi

if [[ -n "$DB_PATH" && ${#LANGS[@]} -ne 1 ]]; then
    echo "Error: --db-path can only be used with exactly one language."
    exit 1
fi

find_font_path() {
    local lang="$1"

    if [[ -n "$FONT_PATH" ]]; then
        echo "$FONT_PATH"
        return 0
    fi

    local candidates=(
        "$PROJECT_DIR/fonts/$lang/NotoSans"*"VariableFont"*.ttf
        "$PROJECT_DIR/fonts/$lang/NotoSans"*.ttf
        "$PROJECT_DIR/fonts/NotoSans"*"VariableFont"*.ttf
        "$PROJECT_DIR/fonts/$lang/"*.ttf
        "$PROJECT_DIR/fonts/"*.ttf
    )

    local c
    for c in "${candidates[@]}"; do
        if [[ -f "$c" ]]; then
            echo "$c"
            return 0
        fi
    done

    return 1
}

find_corpus_path() {
    local lang="$1"

    if [[ -n "$CORPUS_PATH" ]]; then
        echo "$CORPUS_PATH"
        return 0
    fi

    local candidates=(
        "$PROJECT_DIR/data/$lang/corpus_$lang.txt"
        "$PROJECT_DIR/data/$lang/corpus.txt"
        "$PROJECT_DIR/data/corpus_$lang.txt"
        "$PROJECT_DIR/data/corpus.txt"
    )

    local c
    for c in "${candidates[@]}"; do
        if [[ -f "$c" ]]; then
            echo "$c"
            return 0
        fi
    done

    return 1
}

preferred_corpus_path() {
    local lang="$1"
    if [[ -n "$CORPUS_PATH" ]]; then
        echo "$CORPUS_PATH"
    else
        echo "$PROJECT_DIR/data/$lang/corpus_$lang.txt"
    fi
}

auto_generate_corpus() {
    local lang="$1"
    local output_path="$2"

    mkdir -p "$(dirname "$output_path")"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[dry-run] Would auto-generate corpus: $output_path"
        return 0
    fi

    echo "Auto-generating corpus for '$lang'..."
    PROJECT_DIR_ENV="$PROJECT_DIR" \
    LANG_ENV="$lang" \
    OUTPUT_PATH_ENV="$output_path" \
    CORPUS_SENTENCES_ENV="$CORPUS_SENTENCES" \
    uv run --group generate python3 - <<'PY'
import logging
import os
import pathlib
import sys

project_dir = pathlib.Path(os.environ["PROJECT_DIR_ENV"])
sys.path.insert(0, str(project_dir / "src"))

from corpus_generator import create_corpus_from_wiki

logging.basicConfig(level=logging.INFO)

create_corpus_from_wiki(
    output_path=os.environ["OUTPUT_PATH_ENV"],
    lang=os.environ["LANG_ENV"],
    num_sentences=int(os.environ["CORPUS_SENTENCES_ENV"]),
)
PY
}

for lang in "${LANGS[@]}"; do
    lang="$(printf '%s' "$lang" | tr '[:upper:]' '[:lower:]')"

    if ! resolved_font="$(find_font_path "$lang")"; then
        echo "Error: no font found for '$lang'. Use --font-path to set one explicitly."
        exit 1
    fi

    if [[ ! -f "$resolved_font" ]]; then
        echo "Error: font file not found: $resolved_font"
        exit 1
    fi

    if ! resolved_corpus="$(find_corpus_path "$lang")"; then
        resolved_corpus="$(preferred_corpus_path "$lang")"
    fi

    if [[ ! -f "$resolved_corpus" ]]; then
        if [[ "$AUTO_GENERATE_CORPUS" == true ]]; then
            auto_generate_corpus "$lang" "$resolved_corpus"
        else
            echo "Error: no corpus found for '$lang'. Expected one of:"
            echo "  $PROJECT_DIR/data/$lang/corpus_$lang.txt"
            echo "  $PROJECT_DIR/data/$lang/corpus.txt"
            echo "  $PROJECT_DIR/data/corpus_$lang.txt"
            echo "  $PROJECT_DIR/data/corpus.txt"
            echo "Or pass --corpus-path explicitly, or use --auto-generate-corpus."
            exit 1
        fi
    fi

    if [[ ! -f "$resolved_corpus" ]]; then
        if [[ "$DRY_RUN" == true && "$AUTO_GENERATE_CORPUS" == true ]]; then
            echo "[dry-run] Corpus would be auto-generated before DB build: $resolved_corpus"
        else
            echo "Error: corpus file not found after auto-generation: $resolved_corpus"
            exit 1
        fi
    fi

    if [[ -n "$DB_PATH" ]]; then
        resolved_db="$DB_PATH"
    else
        resolved_db="$PROJECT_DIR/data/$lang/char_similarity_db_$lang.json"
    fi

    mkdir -p "$(dirname "$resolved_db")"

    echo "=========================================="
    echo "Generating similarity DB: $lang"
    echo "Corpus: $resolved_corpus"
    echo "Font:   $resolved_font"
    echo "Output: $resolved_db"
    echo "Threshold: $THRESHOLD, Top-K: $TOP_K"
    echo "=========================================="

    if [[ "$DRY_RUN" == true ]]; then
        continue
    fi

    PROJECT_DIR_ENV="$PROJECT_DIR" \
    CORPUS_PATH_ENV="$resolved_corpus" \
    DB_PATH_ENV="$resolved_db" \
    FONT_PATH_ENV="$resolved_font" \
    THRESHOLD_ENV="$THRESHOLD" \
    TOP_K_ENV="$TOP_K" \
    uv run --group generate python3 - <<'PY'
import logging
import os
import pathlib
import sys

project_dir = pathlib.Path(os.environ["PROJECT_DIR_ENV"])
sys.path.insert(0, str(project_dir / "src"))

from character_similarity import generate_similar_chars_db

logging.basicConfig(level=logging.INFO)

generate_similar_chars_db(
    corpus_path=os.environ["CORPUS_PATH_ENV"],
    db_path=os.environ["DB_PATH_ENV"],
    font_path=os.environ["FONT_PATH_ENV"],
    threshold=float(os.environ["THRESHOLD_ENV"]),
    top_k=int(os.environ["TOP_K_ENV"]),
)
PY
done

echo "Done."
