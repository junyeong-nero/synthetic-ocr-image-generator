# Evaluation Metrics

The pipeline calculates specific metrics depending on the format of the generated data.

## Text Recognition Metrics

Used for `sentence`, `document` (text parts), and `markdown` formats.

### Character Error Rate (CER)
The Levenshtein distance between the predicted text and ground truth, normalized by the length of the ground truth.

$$ CER = \frac{S + D + I}{N} $$

Where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is the total number of characters in the reference. Lower is better.

### Word Error Rate (WER)
Similar to CER but calculated at the word level. Lower is better.

## Table Metrics

Used for the `table` format.

### Tree Edit Distance based Similarity (TEDS)
Measures the similarity between the tree structure of the predicted HTML table and the ground truth HTML table. It accounts for both structure (rows, cols) and cell content. Scores range from 0 to 1. Higher is better.

## Key Information Extraction (KIE) Metrics

Used for the `kie` format.

### Entity F1 Score
Measures the precision and recall of extracted key-value pairs.

-   **Precision**: Correctly extracted pairs / Total extracted pairs.
-   **Recall**: Correctly extracted pairs / Total reference pairs.
-   **F1**: Harmonic mean of Precision and Recall.

$$ F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall} $$

## Aggregation

Metrics are aggregated across the dataset:
-   **Average**: Mean score across all samples.
-   **Normalized**: Some metrics are inverted (e.g., 1 - CER) to ensure "higher is better" for leaderboard ranking.
