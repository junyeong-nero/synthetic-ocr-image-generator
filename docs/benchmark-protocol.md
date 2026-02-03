# Benchmark Protocol

To ensure reproducibility and fair comparison, the evaluation pipeline generates a **Protocol Snapshot** for every run.

## Protocol Versioning

The current protocol version is **1.0**.

## Snapshot Content (`protocol.json`)

Every evaluation run produces a `protocol.json` file containing:

-   **`protocol_version`**: Version of the protocol used.
-   **`timestamp`**: UTC timestamp of the run.
-   **`command`**: The exact command (or equivalent) executed.
-   **`config`**: The full configuration state, including resolved defaults.
-   **`summary`**: High-level results (accuracy, latency).
-   **`prompt`**: The exact prompt template used.

## Leaderboard

When running multiple subsets or models, a `leaderboard.json` and `leaderboard.md` are generated.

### Ranking Logic

1.  **Normalization**: All metrics are normalized to a 0-1 scale where 1 is best.
    -   CER/WER are inverted: $1 - Metric$.
    -   TEDS/F1 are used as-is.
2.  **Aggregation**: Scores are averaged across all evaluated subsets.
3.  **Sorting**: Models are ranked by their normalized average score.

## Reproducibility

To reproduce a result from a `protocol.json`:
1.  Check the `config` section for the exact model parameters.
2.  Use the same `dataset` and `split`.
3.  Ensure the `seed` matches if stochastic elements (like temperature) were non-zero.
