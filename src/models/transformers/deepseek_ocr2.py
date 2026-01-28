"""DeepSeek-OCR-2 model wrapper."""

import os
import tempfile
from typing import List

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from models.transformers.base import BaseTransformersOCR, get_attn_implementation


class DeepSeekOCR2(BaseTransformersOCR):
    """
    Wrapper for the DeepSeek-OCR-2 model.

    DeepSeek-OCR-2 is an improved version with enhanced grounding capabilities
    for document-to-markdown conversion.
    """

    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"

    def _load_model(self, model_id: str) -> None:
        self.base_size = 1024
        self.image_size = 768

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation=get_attn_implementation(),
        )
        self.model = self.model.eval()

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
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
