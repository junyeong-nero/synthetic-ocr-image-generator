"""PaddleOCR engine wrapper."""

from typing import List, Union

from PIL import Image
import numpy as np

from src.evaluation.config import ModelConfig
from src.models.base import VLMModel


class PaddleOCREngine(VLMModel):
    """
    Wrapper for the standard PaddleOCR engine (non-VLM).
    Uses the 'paddleocr' python package.
    """

    DEFAULT_LANG = "en"

    def __init__(self, config: Union[ModelConfig, str, None] = None):
        """
        Initialize the model.

        Args:
            config: ModelConfig object, model_id string, or None for default.
        """
        model_id, self.config = self._parse_config(config)
        
        # Parse language from model_id if possible (e.g., "paddleocr/korean")
        self.lang = self.DEFAULT_LANG
        if "/" in model_id:
            parts = model_id.split("/")
            if len(parts) > 1:
                # Simple mapping or usage of the second part as lang
                lang_candidate = parts[-1]
                # Map common names to paddle lang codes if needed
                lang_map = {
                    "korean": "korean",
                    "ko": "korean",
                    "english": "en",
                    "en": "en",
                    "chinese": "ch",
                    "ch": "ch",
                    "japan": "japan",
                    "ja": "japan",
                }
                self.lang = lang_map.get(lang_candidate, self.DEFAULT_LANG)

        self._load_model()

    def _parse_config(
        self, config: Union[ModelConfig, str, None]
    ) -> tuple[str, Union[ModelConfig, None]]:
        if config is None:
            return "paddleocr", None
        if isinstance(config, str):
            return config, None
        return config.model_id, config

    def _load_model(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. Please install it with: "
                "pip install paddlepaddle paddleocr"
            )
        
        # Initialize PaddleOCR
        # use_angle_cls=True is generally good
        # show_log=False to reduce noise
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.
        PaddleOCR engine typically ignores the text prompt and just extracts text.

        Args:
            prompts: List of text prompts (ignored).
            images: List of PIL Images.

        Returns:
            List of extracted text strings.
        """
        results = []
        for image in images:
            # Convert PIL Image to numpy array (RGB)
            img_np = np.array(image)
            
            # Run OCR
            # cls=True to use angle classifier
            ocr_result = self.ocr.ocr(img_np, cls=True)
            
            # Parse result
            # Result structure: list of lists (one per page/image)
            # Each item: [[[x1,y1],[x2,y2]...], ("text", score)]
            
            extracted_text = []
            if ocr_result and ocr_result[0]:
                # Sort by Y coordinate first (top to bottom), then X (left to right) if needed
                # But PaddleOCR usually returns in reading order.
                # Just join the text.
                for line in ocr_result[0]:
                    if line:
                        text_part = line[1][0]
                        extracted_text.append(text_part)
            
            # Join with spaces or newlines? 
            # For sentence evaluation, space is usually safer.
            # For document, newline might be better.
            # Let's use space for now as a safe default for general purpose.
            results.append(" ".join(extracted_text))
            
        return results
