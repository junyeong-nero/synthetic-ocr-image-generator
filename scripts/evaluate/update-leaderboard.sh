#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

BASE_DIR="${1:-$PROJECT_DIR/evaluation_result}"

uv run --group evaluate python "$SCRIPT_DIR/update_leaderboard.py" --base-dir "$BASE_DIR"
