#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_DIR="$PROJECT_DIR/configs/models"
RUN_SCRIPT="$SCRIPT_DIR/run.sh"

DEFAULT_DATASET="junyeong-nero/synthetic-ocr-images-ko"
DEFAULT_MAX_SAMPLES=200

DATASET="${1:-$DEFAULT_DATASET}"
SUBSET="${2:-}"
MAX_SAMPLES="${3:-$DEFAULT_MAX_SAMPLES}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_DIR="$PROJECT_DIR/evaluation_result/_runs"
SUMMARY_FILE="$SUMMARY_DIR/run-all-${TIMESTAMP}.log"
mkdir -p "$SUMMARY_DIR"

PASSED=0
FAILED=0

echo "==========================================" | tee "$SUMMARY_FILE"
echo "Running evaluations for all model configs" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Dataset: $DATASET" | tee -a "$SUMMARY_FILE"
if [[ -n "$SUBSET" ]]; then
    echo "Subset: $SUBSET" | tee -a "$SUMMARY_FILE"
else
    echo "Subset: default(all)" | tee -a "$SUMMARY_FILE"
fi
echo "Max Samples: $MAX_SAMPLES" | tee -a "$SUMMARY_FILE"
echo "Log: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

for config_file in "$CONFIG_DIR"/*.yaml; do
    config_name="$(basename "$config_file" .yaml)"
    if [[ "$config_name" == _* ]]; then
        continue
    fi

    echo "------------------------------------------" | tee -a "$SUMMARY_FILE"
    echo "Config: $config_name" | tee -a "$SUMMARY_FILE"
    echo "------------------------------------------" | tee -a "$SUMMARY_FILE"

    run_args=(--model-id "$config_name" -d "$DATASET" --max-samples "$MAX_SAMPLES")
    if [[ -n "$SUBSET" ]]; then
        run_args+=(--subset "$SUBSET")
    fi

    if "$RUN_SCRIPT" "${run_args[@]}" 2>&1 | tee -a "$SUMMARY_FILE"; then
        PASSED=$((PASSED + 1))
        echo "Result: PASSED" | tee -a "$SUMMARY_FILE"
    else
        FAILED=$((FAILED + 1))
        echo "Result: FAILED" | tee -a "$SUMMARY_FILE"
    fi
    echo "" | tee -a "$SUMMARY_FILE"
done

echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Passed: $PASSED" | tee -a "$SUMMARY_FILE"
echo "Failed: $FAILED" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
