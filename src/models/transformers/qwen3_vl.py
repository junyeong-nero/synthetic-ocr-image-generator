"""Qwen3-VL model wrapper."""

from typing import List, Union

from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluation.config import ModelConfig
from models.base import VLMModel


class Qwen3VL(VLMModel):
    """Wrapper for the Qwen3-VL-2B-Instruct model."""

    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(self, config: Union[ModelConfig, str, None] = None):
        """
        Initialize Qwen3VL model.

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

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
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
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.extend(output_text)
        return results
