#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Generating OCR images for all languages"
echo "=========================================="

LANGUAGES=("korean" "japanese" "hindi")

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing: $lang"
    echo "=========================================="
    
    if [ -f "$SCRIPT_DIR/$lang/generate.sh" ]; then
        bash "$SCRIPT_DIR/$lang/generate.sh"
    else
        echo "Warning: $SCRIPT_DIR/$lang/generate.sh not found"
    fi
done

echo ""
echo "=========================================="
echo "All languages processed!"
echo "=========================================="
