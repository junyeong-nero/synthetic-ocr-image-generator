"""OCR evaluation metrics."""

from src.metrics.edit_distance import (
    accuracy,
    cer,
    character_accuracy,
    levenshtein_distance,
    normalized_cer,
    normalized_wer,
    wer,
    word_accuracy,
)

try:
    from src.metrics.table_edit_distance import TEDS
except ImportError:
    TEDS = None

__all__ = [
    "levenshtein_distance",
    "cer",
    "wer",
    "normalized_cer",
    "normalized_wer",
    "accuracy",
    "word_accuracy",
    "character_accuracy",
    "TEDS",
]