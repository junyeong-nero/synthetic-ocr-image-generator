import os
import tempfile
from typing import List

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ..base import Model


class DeepSeekOCR2(Model):
    """Wrapper for the DeepSeek-OCR-2 model.
    
    DeepSeek-OCR-2 is an improved version with enhanced grounding capabilities
    for document-to-markdown conversion.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-OCR-2",
        use_flash_attention: bool = True,
        base_size: int = 1024,
        image_size: int = 768,
    ):
        self.base_size = base_size
        self.image_size = image_size

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        attn_impl = "flash_attention_2" if use_flash_attention else None
        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }
        if attn_impl:
            model_kwargs["_attn_implementation"] = attn_impl

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model = self.model.eval().cuda().to(torch.bfloat16)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        results = []
        for prompt, image in zip(prompts, images):
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as temp_image_file:
                image.save(temp_image_file, format="PNG")
                temp_image_path = temp_image_file.name

            full_prompt = f"<image>\n<|grounding|>{prompt}"

            try:
                res = self.model.infer(
                    self.tokenizer,
                    prompt=full_prompt,
                    image_file=temp_image_path,
                    output_path="output/",
                    base_size=self.base_size,
                    image_size=self.image_size,
                    crop_mode=True,
                    save_results=False,
                )
                results.append(res)
            finally:
                os.remove(temp_image_path)

        return results
