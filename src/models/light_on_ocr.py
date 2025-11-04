from models.base import vLLMModel


class LightOnOCR(vLLMModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            "lightonai/LightOnOCR-1B-1025", temperature=0.2, max_tokens=4096, top_p=0.9
        )
