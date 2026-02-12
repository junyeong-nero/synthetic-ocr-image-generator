"""HunyuanOCR model wrapper."""

from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

from models.transformers.base import BaseTransformersOCR


class HunYuanOCR(BaseTransformersOCR):
    """Wrapper for the HunyuanOCR model."""

    DEFAULT_MODEL_ID = "tencent/HunyuanOCR"
    DEFAULT_MAX_TOKENS = 16384

    def _load_model(self, model_id: str) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
            model_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            model_dtype = torch.float16
        else:
            self.device = "cpu"
            model_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        try:
            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="eager",
                dtype=model_dtype,
                device_map="auto",
            ).eval()
        except TypeError:
            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="eager",
                torch_dtype=model_dtype,
                device_map="auto",
            ).eval()

    @staticmethod
    def _clean_repeated_substrings(text: str) -> str:
        """Remove long trailing repeated substrings the model sometimes emits."""
        n = len(text)
        if n < 8000:
            return text

        for length in range(2, n // 10 + 1):
            candidate = text[-length:]
            count = 0
            i = n - length

            while i >= 0 and text[i : i + length] == candidate:
                count += 1
                i -= length

            if count >= 10:
                return text[: n - length * (count - 1)]

        return text

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
                    "content": "",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False
                )

            input_ids = inputs.get("input_ids") or inputs.get("inputs")
            generate_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generate_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            cleaned = self._clean_repeated_substrings(decoded)
            results.append(cleaned)

        return results
