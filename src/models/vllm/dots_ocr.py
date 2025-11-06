from ..base import vLLMModel


class DotsOCR(vLLMModel):
    """Wrapper for the Dots.OCR model using vLLM."""

    def __init__(self, **kwargs) -> None:
        super().__init__("rednote-hilab/dots.ocr", temperature=0.0, max_tokens=1024)
