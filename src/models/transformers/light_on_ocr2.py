from typing import List

import torch
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

try:
    from ..base import Model
except ImportError:
    # Fallback for direct file import (avoids circular import in evaluate.py)
    class Model:
        """Base class for OCR models."""
        def __init__(self) -> None:
            pass

        def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
            assert len(prompts) == len(images)
            return ["empty"] * len(prompts)


class LightOnOCR2(Model):
    """Wrapper for the LightOnOCR-2-1B model."""

    def __init__(self, model_id="lightonai/LightOnOCR-2-1B"):
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

            output_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            results.append(output_text)

        return results
