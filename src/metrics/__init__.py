"""OCR evaluation metrics."""

from metrics.edit_distance import (
    accuracy,
    cer,
    character_accuracy,
    levenshtein_distance,
    normalized_cer,
    normalized_wer,
    wer,
    word_accuracy,
)

# Optional imports for table metrics
try:
    from metrics.table_edit_distance import TEDS
except ImportError:
    TEDS = None

__all__ = [
    # Edit distance metrics
    "levenshtein_distance",
    "cer",
    "wer",
    "normalized_cer",
    "normalized_wer",
    "accuracy",
    "word_accuracy",
    "character_accuracy",
    # Table metrics
    "TEDS",
]
