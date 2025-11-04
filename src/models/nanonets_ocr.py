from models.base import vLLMModel


class NanonetsOCR(vLLMModel):

    def __init__(self, **kwargs) -> None:
        super().__init__("nanonets/Nanonets-OCR2-3B", temperature=0.0, max_tokens=8192)
