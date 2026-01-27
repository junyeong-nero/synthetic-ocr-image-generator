"""LightOnOCR-2 model wrapper."""

from typing import List

import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from models.transformers.base import BaseTransformersOCR


class LightOnOCR2(BaseTransformersOCR):
    """Wrapper for the LightOnOCR-2-1B model."""

    DEFAULT_MODEL_ID = "lightonai/LightOnOCR-2-1B"

    def _load_model(self, model_id: str) -> None:
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
        else:
            self.device = "cpu"
            self.dtype = torch.bfloat16

        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.processor = LightOnOcrProcessor.from_pretrained(model_id)

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
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(device=self.device, dtype=self.dtype)
                if v.is_floating_point()
                else v.to(self.device)
                for k, v in inputs.items()
            }

            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            results.append(output_text)

        return results
