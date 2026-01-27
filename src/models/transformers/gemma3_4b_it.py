"""Gemma3-4B-IT model wrapper."""

from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from models.transformers.base import BaseTransformersOCR


class Gemma3_4B_IT(BaseTransformersOCR):
    """Wrapper for the Gemma-3-4B-IT model."""

    DEFAULT_MODEL_ID = "google/gemma-3-4b-it"

    def _load_model(self, model_id: str) -> None:
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
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
        max_tokens = self._get_max_tokens()

        results = []
        for prompt, image in zip(prompts, images):
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False
                )
                generation = generation[0][input_len:]

            decoded = self.processor.decode(generation, skip_special_tokens=True)
            results.append(decoded)
        return results
