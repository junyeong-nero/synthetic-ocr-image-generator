from ..base import vLLMModel


class OlmOCR(vLLMModel):
    """Wrapper for the olmOCR-2-7B-1025 model using vLLM."""

    def __init__(self, **kwargs) -> None:
        super().__init__("allenai/olmOCR-2-7B-1025", temperature=0.0, max_tokens=1024)
