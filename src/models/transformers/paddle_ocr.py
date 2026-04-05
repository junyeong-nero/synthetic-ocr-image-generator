"""PaddleOCR-VL model wrapper."""

from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from src.models.transformers.base import BaseTransformersOCR


class PaddleOCR(BaseTransformersOCR):
    """Wrapper for the PaddleOCR-VL model."""

    DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL"

    def _load_model(self, model_id: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        max_tokens = self._get_max_tokens()

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
