#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_name_or_model_id> [DATASET] [SUBSET] [MAX_SAMPLES] [extra options...]"
    exit 1
fi

MODEL_REF="$1"
shift

DATASET=""
SUBSET=""
MAX_SAMPLES=""

if [[ $# -gt 0 ]] && [[ "${1:-}" != -* ]]; then
    DATASET="$1"
    shift
fi

if [[ $# -gt 0 ]] && [[ "${1:-}" != -* ]]; then
    SUBSET="$1"
    shift
fi

if [[ $# -gt 0 ]] && [[ "${1:-}" != -* ]]; then
    MAX_SAMPLES="$1"
    shift
fi

ARGS=(--model-id "$MODEL_REF")

if [[ -n "$DATASET" ]]; then
    ARGS+=(-d "$DATASET")
fi

if [[ -n "$SUBSET" ]]; then
    ARGS+=(--subset "$SUBSET")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    ARGS+=(--max-samples "$MAX_SAMPLES")
fi

ARGS+=("$@")

exec "$SCRIPT_DIR/../evaluate/run.sh" "${ARGS[@]}"
