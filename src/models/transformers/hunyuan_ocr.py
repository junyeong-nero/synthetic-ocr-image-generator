from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

from ..base import Model


class HunYuanOCR(Model):
    """Wrapper for the HunyuanOCR model."""

    def __init__(self, model_id="tencent/HunyuanOCR"):
        if torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            torch_dtype = torch.float16
        else:
            self.device = "cpu"
            torch_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.model = HunYuanVLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
            device_map={"": self.device},
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

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            model_dtype = next(self.model.parameters()).dtype
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=16384, do_sample=False
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
