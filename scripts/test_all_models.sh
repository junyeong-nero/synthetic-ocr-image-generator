#!/bin/bash
# Test all OCR models with 10 samples each
# Usage: ./scripts/test_all_models.sh [DATASET] [FORMAT] [MAX_SAMPLES]
#
# Examples:
#   ./scripts/test_all_models.sh
#   ./scripts/test_all_models.sh junyeong-nero/synthetic-ocr-images-korean sentence 10
#   ./scripts/test_all_models.sh my-dataset table 5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/configs/models"

# Default settings
DEFAULT_DATASET="junyeong-nero/synthetic-ocr-images-korean"
DEFAULT_FORMAT="sentence"
DEFAULT_MAX_SAMPLES=10

DATASET="${1:-$DEFAULT_DATASET}"
FORMAT="${2:-$DEFAULT_FORMAT}"
MAX_SAMPLES="${3:-$DEFAULT_MAX_SAMPLES}"

# Output directory
OUTPUT_BASE="$PROJECT_DIR/test_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_BASE"

# Results summary file
SUMMARY_FILE="$OUTPUT_BASE/summary.txt"

echo "==========================================" | tee "$SUMMARY_FILE"
echo "Testing All OCR Models" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Dataset: $DATASET" | tee -a "$SUMMARY_FILE"
echo "Format: $FORMAT" | tee -a "$SUMMARY_FILE"
echo "Max Samples: $MAX_SAMPLES" | tee -a "$SUMMARY_FILE"
echo "Output: $OUTPUT_BASE" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Function to extract dependency_group from YAML
get_dependency_group() {
    local yaml_file="$1"
    grep "^dependency_group:" "$yaml_file" 2>/dev/null | sed 's/dependency_group:[[:space:]]*//' | tr -d '"'"'" || echo ""
}

# Function to extract model_id from YAML
get_model_id() {
    local yaml_file="$1"
    grep "^model_id:" "$yaml_file" 2>/dev/null | sed 's/model_id:[[:space:]]*//' | tr -d '"'"'" || echo ""
}

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Process each model config
for config_file in "$CONFIG_DIR"/*.yaml; do
    # Skip template
    if [[ "$(basename "$config_file")" == "_template.yaml" ]]; then
        continue
    fi

    config_name=$(basename "$config_file" .yaml)
    model_id=$(get_model_id "$config_file")
    dependency_group=$(get_dependency_group "$config_file")

    echo "------------------------------------------" | tee -a "$SUMMARY_FILE"
    echo "Model: $config_name" | tee -a "$SUMMARY_FILE"
    echo "Model ID: $model_id" | tee -a "$SUMMARY_FILE"
    echo "Dependency Group: ${dependency_group:-none}" | tee -a "$SUMMARY_FILE"
    echo "------------------------------------------" | tee -a "$SUMMARY_FILE"

    if [[ -z "$dependency_group" ]]; then
        echo "SKIPPED: No dependency_group defined" | tee -a "$SUMMARY_FILE"
        SKIPPED=$((SKIPPED + 1))
        echo "" | tee -a "$SUMMARY_FILE"
        continue
    fi

    # Output directory for this model
    MODEL_OUTPUT="$OUTPUT_BASE/$config_name"
    mkdir -p "$MODEL_OUTPUT"

    # Build the evaluation command
    # Note: For synthetic-ocr-images datasets, subset should match format
    CMD="uv run --group $dependency_group python -m evaluation.cli evaluate \
        -m \"$model_id\" \
        --model-config \"$config_file\" \
        -d \"$DATASET\" \
        --subset \"$FORMAT\" \
        --split train \
        -f \"$FORMAT\" \
        --max-samples $MAX_SAMPLES \
        --output-dir \"$MODEL_OUTPUT\""

    echo "Command: $CMD" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"

    # Run the evaluation
    LOG_FILE="$MODEL_OUTPUT/test.log"
    if eval "$CMD" > "$LOG_FILE" 2>&1; then
        echo "PASSED" | tee -a "$SUMMARY_FILE"
        PASSED=$((PASSED + 1))
    else
        echo "FAILED (see $LOG_FILE)" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
        # Print last 20 lines of error
        echo "Last 20 lines of error:" | tee -a "$SUMMARY_FILE"
        tail -20 "$LOG_FILE" | tee -a "$SUMMARY_FILE"
    fi
    echo "" | tee -a "$SUMMARY_FILE"
done

echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Summary" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Passed: $PASSED" | tee -a "$SUMMARY_FILE"
echo "Failed: $FAILED" | tee -a "$SUMMARY_FILE"
echo "Skipped: $SKIPPED" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

# Exit with error if any failed
if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
