#!/bin/bash

set -e

echo "=========================================="
echo "Analyzing OCR evaluation results"
echo "=========================================="

uv run src/analyze.py

echo ""
echo "=========================================="
echo "Analysis completed!"
echo "=========================================="
