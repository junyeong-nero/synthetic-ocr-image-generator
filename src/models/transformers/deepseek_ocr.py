"""DeepSeek-OCR model wrapper."""

import os
import tempfile
from typing import List

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from models.transformers.base import BaseTransformersOCR


class DeepSeekOCR(BaseTransformersOCR):
    """Wrapper for the DeepSeek-OCR model."""

    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"

    def _load_model(self, model_id: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
        )
        if self.device == "cuda":
            self.model = self.model.eval().cuda().to(torch.bfloat16)
        else:
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

            full_prompt = f"<image>\n{prompt}"

            try:
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
            finally:
                os.remove(temp_image_path)
        return results
