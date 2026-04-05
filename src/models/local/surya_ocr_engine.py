from typing import List, Union

from PIL import Image

from src.evaluation.config import ModelConfig
from src.models.base import VLMModel


class SuryaOCREngine(VLMModel):
    def __init__(self, config: Union[ModelConfig, str, None] = None):
        self.model_id, self.config = self._parse_config(config)
        self._load_model()

    def _parse_config(
        self, config: Union[ModelConfig, str, None]
    ) -> tuple[str, Union[ModelConfig, None]]:
        if config is None:
            return "surya/surya-ocr", None
        if isinstance(config, str):
            return config, None
        return config.model_id, config

    def _load_model(self) -> None:
        try:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise ImportError(
                "Surya OCR is not installed. Please install it with: pip install surya-ocr"
            ) from exc

        foundation_predictor = FoundationPredictor()
        self.recognition_predictor = RecognitionPredictor(foundation_predictor)
        self.detection_predictor = DetectionPredictor()

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        _ = prompts

        ocr_results = self.recognition_predictor(
            images,
            det_predictor=self.detection_predictor,
            sort_lines=True,
        )

        results: List[str] = []
        for page_result in ocr_results:
            lines = [line.text for line in page_result.text_lines if line.text]
            results.append(" ".join(lines))

        return results
