"""PaddleOCR-VL model wrapper."""

from typing import List, Union

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from evaluation.config import ModelConfig
from models.base import VLMModel


class PaddleOCR(VLMModel):
    """Wrapper for the PaddleOCR-VL model."""

    DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL"

    def __init__(self, config: Union[ModelConfig, str, None] = None):
        """
        Initialize PaddleOCR model.

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
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

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
        for prompt, image in zip(prompts, images):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            decoded = decoded.split("Assistant:")[-1].strip()
            results.append(decoded)
        return results
