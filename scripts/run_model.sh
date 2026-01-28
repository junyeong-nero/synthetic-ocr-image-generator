#!/bin/bash
# Run evaluation for a single model with automatic dependency group resolution
# Usage: ./scripts/run_model.sh <config_name> [options...]
#
# Examples:
#   ./scripts/run_model.sh qwen3-vl-2b -d dataset --subset sentence --max-samples 10
#   ./scripts/run_model.sh deepseek-ocr-2 -d dataset --subset markdown

set -e

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_name> [evaluation options...]"
    echo ""
    echo "Available configs:"
    ls -1 configs/models/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | grep -v "^_" | sort
    exit 1
fi

CONFIG_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/configs/models/${CONFIG_NAME}.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available configs:"
    ls -1 "$PROJECT_DIR/configs/models"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | grep -v "^_" | sort
    exit 1
fi

# Extract model_id and dependency_group from config
MODEL_ID=$(grep "^model_id:" "$CONFIG_FILE" | sed 's/model_id:[[:space:]]*//' | tr -d '"'"'")
DEPENDENCY_GROUP=$(grep "^dependency_group:" "$CONFIG_FILE" 2>/dev/null | sed 's/dependency_group:[[:space:]]*//' | tr -d '"'"'" || echo "")

echo "Config: $CONFIG_NAME"
echo "Model: $MODEL_ID"
echo "Dependency Group: ${DEPENDENCY_GROUP:-none}"
echo ""

# Check if --subset is passed in arguments
HAS_SUBSET=false
for arg in "$@"; do
    if [[ "$arg" == "--subset" ]] || [[ "$arg" == "-s" ]]; then
        HAS_SUBSET=true
        break
    fi
done

if [[ "$HAS_SUBSET" == "true" ]]; then
    # Run once with provided arguments
    if [[ -z "$DEPENDENCY_GROUP" ]]; then
        echo "Warning: No dependency_group defined, running without --group"
        uv run main.py evaluate --model-config "$CONFIG_FILE" "$@"
    else
        echo "Running: uv run --group $DEPENDENCY_GROUP main.py evaluate ..."
        uv run --group "$DEPENDENCY_GROUP" main.py evaluate --model-config "$CONFIG_FILE" "$@"
    fi
else
    # Iterate over default subsets
    echo "No --subset specified. Running for all default subsets."
    SUBSETS=("sentence" "table" "document" "markdown" "kie")
    
    for subset in "${SUBSETS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Running subset: $subset"
        echo "=========================================="
        
        if [[ -z "$DEPENDENCY_GROUP" ]]; then
            echo "Cmd: uv run main.py evaluate --model-config $CONFIG_FILE --subset $subset $@"
            uv run main.py evaluate --model-config "$CONFIG_FILE" --subset "$subset" "$@"
        else
            echo "Cmd: uv run --group $DEPENDENCY_GROUP main.py evaluate --model-config $CONFIG_FILE --subset $subset $@"
            uv run --group "$DEPENDENCY_GROUP" main.py evaluate --model-config "$CONFIG_FILE" --subset "$subset" "$@"
        fi
    done
fi
