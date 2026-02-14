#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

BASE_DIR="$PROJECT_DIR/evaluation_result"
if [[ $# -gt 0 ]] && [[ "$1" != -* ]]; then
    BASE_DIR="$1"
    shift
fi

uv run --group evaluate python "$SCRIPT_DIR/update_leaderboard.py" --base-dir "$BASE_DIR" "$@"
