"""Kiri-OCR model wrapper."""

from typing import List, Union

from PIL import Image

from models.transformers.base import BaseTransformersOCR
from models.config import ModelConfig


class KiriOCR(BaseTransformersOCR):
    """Wrapper for the Kiri-OCR model.

    Note: This model uses the kiri_ocr library which provides its own
    model loading and inference logic.
    """

    DEFAULT_MODEL_ID = "kiri-ocr"

    def _load_model(self, model_id: str) -> None:
        from kiri_ocr import OCR

        self.ocr = OCR()

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts (ignored for this model).
            images: List of PIL Images.

        Returns:
            List of extracted text strings.
        """
        results = []

        for image in images:
            text, _ = self.ocr.extract_text(image)
            results.append(text)

        return results
