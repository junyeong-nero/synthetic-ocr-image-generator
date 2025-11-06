from ..base import vLLMModel


class LightOnOCR(vLLMModel):
    """Wrapper for the LightOnOCR-1B-1025 model using vLLM."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            "lightonai/LightOnOCR-1B-1025", temperature=0.2, max_tokens=1024, top_p=0.9
        )
