# Metrics

The pipeline uses several standardized metrics to evaluate OCR and VLM performance across different formats.

## Text-Based Metrics

Used primarily for `sentence` and `markdown` formats.

### 1. CER (Character Error Rate)
Measures the edit distance at the character level.
- **Formula**: `(S + D + I) / N`
    - `S`: Substitutions
    - `D`: Deletions
    - `I`: Insertions
    - `N`: Total characters in reference
- **Lower is better**.

### 2. WER (Word Error Rate)
Measures the edit distance at the word level.
- **Lower is better**.

## Table Metrics

Used for the `table` format.

### TEDS (Tree Edit Distance-based Similarity)
Evaluates both the content and the structure of the predicted table by comparing the HTML tree structures.
- **Range**: [0, 1]
- **Higher is better**.

## KIE (Key Information Extraction) Metrics

Used for the `kie` format.

### Entity-Level F1-Score
Measures the model's ability to correctly extract key-value pairs (entities).
- **Precision**: Ratio of correctly predicted entities to total predicted entities.
- **Recall**: Ratio of correctly predicted entities to total ground truth entities.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **Higher is better**.

## Document Metrics

Used for the `document` format.

### Overall F1-Score
Aggregated performance across all elements in a document layout.

## Normalization and Leaderboard

To create a unified leaderboard across different formats and metrics, we use **Normalized Scores**:
- For metrics where "lower is better" (e.g., CER, WER), the normalized score is `1.0 - value`.
- For metrics where "higher is better" (e.g., TEDS, F1), the normalized score is the raw value.

The **Average Score** on the leaderboard is a weighted average of these normalized scores across all evaluated subsets, weighted by the number of samples in each subset.
