from models.base import vLLMModel


class OlmOCR(vLLMModel):

    def __init__(self, **kwargs) -> None:
        super().__init__("allenai/olmOCR-2-7B-1025", temperature=0.0, max_tokens=8192)
