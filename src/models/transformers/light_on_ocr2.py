"""LightOnOCR-2 model wrapper."""

from typing import List, Union

import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from evaluation.config import ModelConfig
from models.base import VLMModel


class LightOnOCR2(VLMModel):
    """Wrapper for the LightOnOCR-2-1B model."""

    DEFAULT_MODEL_ID = "lightonai/LightOnOCR-2-1B"

    def __init__(self, config: Union[ModelConfig, str, None] = None):
        """
        Initialize LightOnOCR2 model.

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

        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float32 if self.device == "mps" else torch.bfloat16

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
        max_tokens = 1024
        if self.config is not None:
            max_tokens = self.config.max_tokens

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
