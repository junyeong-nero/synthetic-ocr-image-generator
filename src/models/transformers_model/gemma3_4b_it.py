from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from ..base import Model


class Gemma3_4B_IT(Model):
    def __init__(self, model_id="google/gemma-3-4b-it"):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
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
                    **inputs, max_new_tokens=1024, do_sample=False
                )
                generation = generation[0][input_len:]

            decoded = self.processor.decode(generation, skip_special_tokens=True)
            results.append(decoded)
        return results
