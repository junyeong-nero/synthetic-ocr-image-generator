#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_DIR="$PROJECT_DIR/configs/models"
RUN_SCRIPT="$PROJECT_DIR/scripts/evaluate/run.sh"

DEFAULT_DATASET="junyeong-nero/synthetic-ocr-images-ko"
DEFAULT_MAX_SAMPLES=200

usage() {
    echo "Usage: $0 [--dataset <repo>] [--subset <name>] [--max-samples <n>]"
    echo "       $0 [DATASET] [SUBSET] [MAX_SAMPLES]"
}

DATASET="$DEFAULT_DATASET"
SUBSET=""
MAX_SAMPLES="$DEFAULT_MAX_SAMPLES"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dataset)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value"
                usage
                exit 1
            fi
            DATASET="$2"
            shift 2
            ;;
        --dataset=*)
            DATASET="${1#*=}"
            shift
            ;;
        -s|--subset)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value"
                usage
                exit 1
            fi
            SUBSET="$2"
            shift 2
            ;;
        --subset=*)
            SUBSET="${1#*=}"
            shift
            ;;
        --max-samples|-m)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a value"
                usage
                exit 1
            fi
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --max-samples=*)
            MAX_SAMPLES="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            POSITIONAL_ARGS+=("$@")
            break
            ;;
        -*)
            echo "Error: unknown option: $1"
            usage
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 3 ]]; then
    echo "Error: too many positional arguments"
    usage
    exit 1
fi

if [[ ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
    DATASET="${POSITIONAL_ARGS[0]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
    SUBSET="${POSITIONAL_ARGS[1]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 3 ]]; then
    MAX_SAMPLES="${POSITIONAL_ARGS[2]}"
fi

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

    run_args=("$config_name" -d "$DATASET" --max-samples "$MAX_SAMPLES")
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
