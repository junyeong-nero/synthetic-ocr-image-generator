#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Evaluating OCR images for all languages"
echo "=========================================="

LANGUAGES=("korean" "japanese" "hindi")

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating: $lang"
    echo "=========================================="
    
    if [ -f "$SCRIPT_DIR/$lang/evaluate.sh" ]; then
        bash "$SCRIPT_DIR/$lang/evaluate.sh"
    else
        echo "Warning: $SCRIPT_DIR/$lang/evaluate.sh not found"
    fi
done

echo ""
echo "=========================================="
echo "All languages evaluated!"
echo "=========================================="

# Run analysis after evaluation
bash "$SCRIPT_DIR/common/analyze.sh"
