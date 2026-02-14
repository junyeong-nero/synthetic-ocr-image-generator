#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_DIR="$PROJECT_DIR/configs/models"
DEFAULT_DATASET="junyeong-nero/synthetic-ocr-images-ko"
DEFAULT_SPLIT="test"

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

PASSTHROUGH_ARGS=()
HAS_DATASET=false
HAS_SPLIT=false

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
        --split)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value"
                exit 1
            fi
            if [[ "$2" != "train" && "$2" != "test" ]]; then
                echo "Error: --split must be one of: train, test"
                exit 1
            fi
            HAS_SPLIT=true
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --split=*)
            split_value="${1#*=}"
            if [[ "$split_value" != "train" && "$split_value" != "test" ]]; then
                echo "Error: --split must be one of: train, test"
                exit 1
            fi
            HAS_SPLIT=true
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
MODEL_DIR_NAME="${MODEL_ID##*/}"
DEPENDENCY_GROUP="$(extract_yaml_key "$CONFIG_FILE" "dependency_group")"

echo "Config: $CONFIG_NAME"
echo "Model: $MODEL_ID"
echo "Dependency Group: ${DEPENDENCY_GROUP:-none}"
echo ""

run_eval() {
    local output_dir="$PROJECT_DIR/evaluation_result/$MODEL_DIR_NAME"
    mkdir -p "$output_dir"

    local cmd=(uv run --group evaluate)
    if [[ -n "$DEPENDENCY_GROUP" ]]; then
        cmd+=(--group "$DEPENDENCY_GROUP")
    fi
    cmd+=(main.py evaluate --model-config "$CONFIG_FILE" --output-dir "$output_dir")
    if [[ "$HAS_DATASET" == "false" ]]; then
        cmd+=(-d "$DEFAULT_DATASET")
    fi
    if [[ "$HAS_SPLIT" == "false" ]]; then
        cmd+=(--split "$DEFAULT_SPLIT")
    fi

    echo "Running evaluation"
    echo "Output: $output_dir"
    "${cmd[@]}" "${PASSTHROUGH_ARGS[@]}"
}

run_eval
