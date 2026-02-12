#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_DIR="$PROJECT_DIR/configs/models"
DEFAULT_DATASET="junyeong-nero/synthetic-ocr-images-ko"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_name>|--model-id <config_name_or_model_id> [evaluation options...]"
    echo ""
    echo "Available configs:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | grep -v "^_" | sort
    exit 1
fi

extract_yaml_key() {
    local yaml_file="$1"
    local key="$2"
    grep "^${key}:" "$yaml_file" 2>/dev/null | sed "s/${key}:[[:space:]]*//" | tr -d "\"'" || true
}

resolve_config_file() {
    local ref="$1"
    local by_name="$CONFIG_DIR/${ref}.yaml"
    if [[ -f "$by_name" ]]; then
        echo "$by_name"
        return 0
    fi

    for file in "$CONFIG_DIR"/*.yaml; do
        local base
        base="$(basename "$file" .yaml)"
        if [[ "$base" == _* ]]; then
            continue
        fi

        local model_id
        model_id="$(extract_yaml_key "$file" "model_id")"
        if [[ "$model_id" == "$ref" ]]; then
            echo "$file"
            return 0
        fi
    done

    return 1
}

MODEL_REF=""
if [[ "${1:-}" != -* ]]; then
    MODEL_REF="$1"
    shift
fi

SUBSETS=()
PASSTHROUGH_ARGS=()
HAS_DATASET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id|-m)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value"
                exit 1
            fi
            MODEL_REF="$2"
            shift 2
            ;;
        --subset|-s)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value"
                exit 1
            fi
            IFS=',' read -r -a parsed_subsets <<< "$2"
            for subset in "${parsed_subsets[@]}"; do
                trimmed="$(echo "$subset" | xargs)"
                if [[ -n "$trimmed" ]]; then
                    SUBSETS+=("$trimmed")
                fi
            done
            shift 2
            ;;
        --subset=*)
            IFS=',' read -r -a parsed_subsets <<< "${1#*=}"
            for subset in "${parsed_subsets[@]}"; do
                trimmed="$(echo "$subset" | xargs)"
                if [[ -n "$trimmed" ]]; then
                    SUBSETS+=("$trimmed")
                fi
            done
            shift
            ;;
        -d|--dataset)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value"
                exit 1
            fi
            HAS_DATASET=true
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --dataset=*)
            HAS_DATASET=true
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$MODEL_REF" ]]; then
    echo "Error: model reference is required."
    exit 1
fi

CONFIG_FILE="$(resolve_config_file "$MODEL_REF" || true)"
if [[ -z "$CONFIG_FILE" ]] || [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config not found for model reference: $MODEL_REF"
    echo ""
    echo "Available configs:"
    ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | grep -v "^_" | sort
    exit 1
fi

CONFIG_NAME="$(basename "$CONFIG_FILE" .yaml)"
MODEL_ID="$(extract_yaml_key "$CONFIG_FILE" "model_id")"
DEPENDENCY_GROUP="$(extract_yaml_key "$CONFIG_FILE" "dependency_group")"

if [[ ${#SUBSETS[@]} -eq 0 ]]; then
    SUBSETS=("sentence" "table" "document" "markdown" "kie")
    echo "No --subset specified. Running all default subsets."
fi

for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [[ "$arg" == "--help" ]] || [[ "$arg" == "-h" ]]; then
        SUBSETS=("sentence")
        break
    fi
done

echo "Config: $CONFIG_NAME"
echo "Model: $MODEL_ID"
echo "Dependency Group: ${DEPENDENCY_GROUP:-none}"
echo ""

run_subset() {
    local subset="$1"
    local subset_dir="$PROJECT_DIR/evaluation_result/$MODEL_ID/$subset"
    mkdir -p "$subset_dir"

    local cmd=(uv run --group evaluate)
    if [[ -n "$DEPENDENCY_GROUP" ]]; then
        cmd+=(--group "$DEPENDENCY_GROUP")
    fi
    cmd+=(main.py evaluate --model-config "$CONFIG_FILE" --subset "$subset" --output-dir "$subset_dir")
    if [[ "$HAS_DATASET" == "false" ]]; then
        cmd+=(-d "$DEFAULT_DATASET")
    fi

    echo "Running subset: $subset"
    echo "Output: $subset_dir"
    "${cmd[@]}" "${PASSTHROUGH_ARGS[@]}"
}

for subset in "${SUBSETS[@]}"; do
    run_subset "$subset"
done
