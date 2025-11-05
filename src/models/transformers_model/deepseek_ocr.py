import os
import tempfile
from typing import List

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ..base import Model


class DeepSeekOCR(Model):
    def __init__(self, model_name="deepseek-ai/DeepSeek-OCR"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            # _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        results = []
        for prompt, image in zip(prompts, images):
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as temp_image_file:
                image.save(temp_image_file, format="PNG")
                temp_image_path = temp_image_file.name

            full_prompt = f"<image>\n{prompt}"

            res = self.model.infer(
                self.tokenizer,
                prompt=full_prompt,
                image_file=temp_image_path,
                output_path="output/",
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=True,
            )
            results.append(res)
            os.remove(temp_image_path)
        return results
