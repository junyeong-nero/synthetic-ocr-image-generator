# Benchmark Protocol

This document defines the standard protocol for benchmarking models to ensure consistency and comparability of results.

## 1. Environment Standardization

- **Python Version**: 3.11 or higher.
- **Dependency Management**: Use `uv` with the exact versions specified in `uv.lock`.
- **Hardware**: For local models (Transformers), specify the GPU model and available VRAM in the report.

## 2. Evaluation Procedure

### Subset Selection
A complete benchmark should ideally include all core subsets:
- `sentence`
- `table`
- `document`
- `markdown`
- `kie`

### Sample Size
- **Minimum**: 100 samples per subset for a preliminary score.
- **Standard**: 1,000+ samples per subset for official leaderboard submission.

### Reproducibility
- Always set a fixed `--seed` (default recommendation: `42`).
- Include the `protocol.json` artifact with any submitted results.

## 3. Metric Reporting

- **Primary Metrics**:
    - Sentence: CER
    - Table: TEDS
    - KIE: Entity F1
    - Document: Overall F1
- **Latency**: Report `avg_latency_ms` as provided in the summary.

## 4. Prompting Guidelines

The benchmark uses a standardized prompt for each format. If a model requires a custom prompt format (e.g., specific chat templates), it should be handled within the model's backend implementation in `src/models/`, not by changing the evaluation script, to maintain task consistency.

## 5. Versioning

- **Protocol Version**: 1.0
- Results are tagged with a UTC timestamp and the protocol version.
- When the generation logic or metric calculation changes, the protocol version will be incremented, and previous results may need to be re-evaluated.
