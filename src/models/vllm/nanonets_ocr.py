from ..base import vLLMModel


class NanonetsOCR(vLLMModel):
    """Wrapper for the Nanonets-OCR2-3B model using vLLM."""

    def __init__(self, **kwargs) -> None:
        super().__init__("nanonets/Nanonets-OCR2-3B", temperature=0.0, max_tokens=1024)
