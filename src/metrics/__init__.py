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
from metrics.normalization import (
    create_normalizer,
    normalize_for_language,
    normalize_for_ocr,
    normalize_text,
    strip_markdown,
)

# Optional imports for table/document metrics
try:
    from metrics.table_edit_distance import TEDS
except ImportError:
    TEDS = None

try:
    from metrics.table_document_metrics import (
        calculate_iou,
        cell_level_text_accuracy,
        evaluate_document,
        evaluate_table,
        key_value_extraction_f1,
        layout_detection_map,
        reading_order_accuracy,
        row_column_detection_metrics,
    )
except ImportError:
    # Provide stub functions if dependencies are missing
    def evaluate_table(*args, **kwargs):
        raise ImportError("table_document_metrics requires lxml and apted packages")

    def evaluate_document(*args, **kwargs):
        raise ImportError("table_document_metrics requires lxml and apted packages")

    calculate_iou = None
    cell_level_text_accuracy = None
    key_value_extraction_f1 = None
    layout_detection_map = None
    reading_order_accuracy = None
    row_column_detection_metrics = None

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
    # Normalization
    "normalize_text",
    "normalize_for_language",
    "normalize_for_ocr",
    "create_normalizer",
    "strip_markdown",
    # Table metrics
    "TEDS",
    "evaluate_table",
    "cell_level_text_accuracy",
    "row_column_detection_metrics",
    # Document metrics
    "evaluate_document",
    "calculate_iou",
    "layout_detection_map",
    "reading_order_accuracy",
    "key_value_extraction_f1",
]
