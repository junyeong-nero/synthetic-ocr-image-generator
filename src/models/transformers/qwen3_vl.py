"""Qwen3-VL model wrapper."""

from typing import List

from PIL import Image
from transformers import Qwen3VLForConditionalGeneration

from src.models.transformers.base import StandardTransformersOCR


class Qwen3VL(StandardTransformersOCR):
    """Wrapper for the Qwen3-VL-2B-Instruct model."""

    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    MODEL_CLASS = Qwen3VLForConditionalGeneration

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
