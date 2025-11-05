from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from ..base import Model


class PaddleOCR(Model):
    def __init__(self, model_id="PaddlePaddle/PaddleOCR-VL"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

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
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model.generate(**inputs, max_new_tokens=1024)
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            results.append(decoded)
        return results
