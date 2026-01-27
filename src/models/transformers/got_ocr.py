"""GOT-OCR model wrapper."""

from typing import List, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.config import ModelConfig
from models.base import VLMModel


class GotOCR(VLMModel):
    """Wrapper for the GOT-OCR-2.0-hf model."""

    DEFAULT_MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"

    def __init__(self, config: Union[ModelConfig, str, None] = None):
        """
        Initialize GotOCR model.

        Args:
            config: ModelConfig object, model_id string, or None for default.
        """
        if config is None:
            model_id = self.DEFAULT_MODEL_ID
            self.config = None
        elif isinstance(config, str):
            model_id = config
            self.config = None
        else:
            model_id = config.model_id
            self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        max_tokens = 1024
        if self.config is not None:
            max_tokens = self.config.max_tokens

        results = []
        for image in images:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=max_tokens,
            )

            decoded = self.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            results.append(decoded)
        return results
