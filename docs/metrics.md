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

### Document Component Scores
- `avg_layout_f1`: Layout detection quality.
- `avg_reading_order`: Reading order correctness.
- `avg_kv_f1`: Key-value extraction quality.
- `avg_text_score`: Text element matching quality.
- `avg_table_teds`: Table structure/content quality in document context.

### Formula-Aware Composite Score
- `avg_formula_edit_distance`: Formula text/LaTeX edit distance (lower is better).
- `avg_text_table_formula_score`: Composite score `(text_score + table_teds + (1 - formula_edit_distance)) / 3`.

This composite is the recommended representative metric for `document` because it balances text, table, and formula quality.

### Legacy Aggregate
- `avg_text_table_score` / `avg_overall_f1` are still emitted for backward compatibility.

## Normalization and Leaderboard

To create a unified leaderboard across different formats and metrics, we use **Normalized Scores**:
- For metrics where "lower is better" (e.g., CER, WER, formula edit distance), the normalized score is `1.0 - value`.
- For metrics where "higher is better" (e.g., TEDS, F1), the normalized score is the raw value.

The **Average Score** on the leaderboard is a weighted average of these normalized scores across all evaluated subsets, weighted by the number of samples in each subset.

## Representative Metrics by Subset

- `sentence`: `avg_cer`
- `table`: `avg_teds`
- `document`: `avg_text_table_formula_score`
- `markdown`: `normalized_match_rate` (text-fidelity first), with `avg_cer` as supporting metric
- `kie`: `avg_entity_f1` (or `avg_overall_f1` as fallback)
