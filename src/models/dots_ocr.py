from models.base import vLLMModel


class DotsOCR(vLLMModel):

    def __init__(self, **kwargs) -> None:
        super().__init__("rednote-hilab/dots.ocr", temperature=0.0, max_tokens=1024)
