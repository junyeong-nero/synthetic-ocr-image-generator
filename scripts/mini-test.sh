#!/bin/bash
# Mini dependency-group smoke test before full evaluation runs.
#
# What it tests per group:
# 1) `uv sync --group evaluate --group <group>`
# 2) Optional import smoke test for each dependency in the group
#
# Usage:
#   ./scripts/mini-test.sh
#   ./scripts/mini-test.sh --source configs
#   ./scripts/mini-test.sh --groups qwen3-vl,paddle-ocr
#   ./scripts/mini-test.sh --sync-only

set -u

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --source <pyproject|configs>  Group source (default: pyproject)"
    echo "  --groups <a,b,c>              Comma-separated groups to test"
    echo "  --include-core                Include generate/evaluate groups"
    echo "  --sync-only                   Skip python import smoke test"
    echo "  --output-dir <dir>            Output directory for logs/summary"
    echo "  -h, --help                    Show help"
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SOURCE="pyproject"
GROUPS_OVERRIDE=""
INCLUDE_CORE="false"
SYNC_ONLY="false"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --groups)
            GROUPS_OVERRIDE="$2"
            shift 2
            ;;
        --include-core)
            INCLUDE_CORE="true"
            shift
            ;;
        --sync-only)
            SYNC_ONLY="true"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
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

if [[ "$SOURCE" != "pyproject" && "$SOURCE" != "configs" ]]; then
    echo "Error: --source must be one of: pyproject, configs"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROJECT_DIR/test_results/mini_dep_test_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

echo "==========================================" | tee "$SUMMARY_FILE"
echo "Mini Dependency Group Test" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Project: $PROJECT_DIR" | tee -a "$SUMMARY_FILE"
echo "Source: $SOURCE" | tee -a "$SUMMARY_FILE"
echo "Sync only: $SYNC_ONLY" | tee -a "$SUMMARY_FILE"
echo "Include core groups: $INCLUDE_CORE" | tee -a "$SUMMARY_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv command not found" | tee -a "$SUMMARY_FILE"
    exit 1
fi

get_groups_from_pyproject() {
    python3 - "$PROJECT_DIR/pyproject.toml" "$INCLUDE_CORE" <<'PY'
import sys
import tomllib

pyproject_path = sys.argv[1]
include_core = sys.argv[2].lower() == "true"

with open(pyproject_path, "rb") as f:
    data = tomllib.load(f)

groups = sorted((data.get("dependency-groups") or {}).keys())

for group in groups:
    if not include_core and group in {"generate", "evaluate"}:
        continue
    print(group)
PY
}

get_groups_from_configs() {
    grep "^dependency_group:" "$PROJECT_DIR"/configs/models/*.yaml 2>/dev/null \
        | sed 's/.*dependency_group:[[:space:]]*//' \
        | tr -d '"' \
        | sort -u
}

GROUPS=()

if [[ -n "$GROUPS_OVERRIDE" ]]; then
    IFS=',' read -r -a GROUPS <<< "$GROUPS_OVERRIDE"
else
    if [[ "$SOURCE" == "pyproject" ]]; then
        while IFS= read -r group; do
            [[ -z "$group" ]] && continue
            GROUPS+=("$group")
        done < <(get_groups_from_pyproject)
    else
        while IFS= read -r group; do
            [[ -z "$group" ]] && continue
            GROUPS+=("$group")
        done < <(get_groups_from_configs)
    fi
fi

if [[ ${#GROUPS[@]} -eq 0 ]]; then
    echo "No groups found to test." | tee -a "$SUMMARY_FILE"
    exit 1
fi

echo "Groups to test (${#GROUPS[@]}): ${GROUPS[*]}" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

PASSED=0
FAILED=0

run_import_smoke() {
    local group="$1"
    local log_file="$2"

    uv run --group evaluate --group "$group" python - "$PROJECT_DIR/pyproject.toml" "$group" >>"$log_file" 2>&1 <<'PY'
import importlib
import re
import sys
import tomllib

pyproject_path = sys.argv[1]
group_name = sys.argv[2]

with open(pyproject_path, "rb") as f:
    data = tomllib.load(f)

groups = data.get("dependency-groups") or {}
deps = groups.get(group_name, [])

SPECIAL_IMPORT_MAP = {
    "opencv-python": "cv2",
    "pillow": "PIL",
    "scikit-image": "skimage",
    "pyyaml": "yaml",
    "huggingface-hub": "huggingface_hub",
    "google-generativeai": "google.generativeai",
    "qwen-vl-utils": "qwen_vl_utils",
    "paddlepaddle": "paddle",
}

def normalize_req_name(req: str) -> str:
    left = req.split("@", 1)[0].strip()
    left = re.split(r"[<>=!~;\[]", left, maxsplit=1)[0].strip()
    return left.lower()

failures = []

print(f"[import-smoke] group={group_name}")
for dep in deps:
    pkg_name = normalize_req_name(dep)
    if not pkg_name:
        continue
    module_name = SPECIAL_IMPORT_MAP.get(pkg_name, pkg_name.replace("-", "_"))
    try:
        importlib.import_module(module_name)
        print(f"  OK   {dep} -> import {module_name}")
    except Exception as exc:
        failures.append((dep, module_name, repr(exc)))
        print(f"  FAIL {dep} -> import {module_name}: {exc}")

if failures:
    sys.exit(1)
PY
}

for group in "${GROUPS[@]}"; do
    echo "------------------------------------------" | tee -a "$SUMMARY_FILE"
    echo "Group: $group" | tee -a "$SUMMARY_FILE"
    echo "------------------------------------------" | tee -a "$SUMMARY_FILE"

    group_safe_name=$(echo "$group" | tr '/' '_')
    log_file="$OUTPUT_DIR/${group_safe_name}.log"

    echo "[1/2] Sync: uv sync --group evaluate --group $group" | tee -a "$SUMMARY_FILE"
    if ! uv sync --group evaluate --group "$group" >"$log_file" 2>&1; then
        echo "FAILED: sync error (see $log_file)" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
        echo "" | tee -a "$SUMMARY_FILE"
        continue
    fi

    if [[ "$SYNC_ONLY" == "true" ]]; then
        echo "PASSED: sync ok (import smoke skipped)" | tee -a "$SUMMARY_FILE"
        PASSED=$((PASSED + 1))
        echo "" | tee -a "$SUMMARY_FILE"
        continue
    fi

    echo "[2/2] Import smoke: uv run --group evaluate --group $group python ..." | tee -a "$SUMMARY_FILE"
    if run_import_smoke "$group" "$log_file"; then
        echo "PASSED" | tee -a "$SUMMARY_FILE"
        PASSED=$((PASSED + 1))
    else
        echo "FAILED: import smoke error (see $log_file)" | tee -a "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
    fi
    echo "" | tee -a "$SUMMARY_FILE"
done

echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Summary" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo "Passed: $PASSED" | tee -a "$SUMMARY_FILE"
echo "Failed: $FAILED" | tee -a "$SUMMARY_FILE"
echo "Summary file: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
