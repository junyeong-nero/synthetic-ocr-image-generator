"""Nanonets-OCR model wrapper."""

from src.models.transformers.nanonets_ocr2 import NanonetsOCR2


class NanonetsOCR(NanonetsOCR2):
    """Wrapper for the original Nanonets OCR model."""

    DEFAULT_MODEL_ID = "nanonets/Nanonets-OCR-s"
