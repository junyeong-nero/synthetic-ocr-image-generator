from typing import List

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..base import Model


class GotOCR(Model):
    def __init__(self, model_id="stepfun-ai/GOT-OCR-2.0-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        results = []
        for image in images:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )

            decoded = self.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            results.append(decoded)
        return results
