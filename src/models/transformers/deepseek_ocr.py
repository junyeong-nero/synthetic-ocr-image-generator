"""DeepSeek-OCR model wrapper."""

import os
import tempfile
from typing import List, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from evaluation.config import ModelConfig
from models.base import VLMModel


class DeepSeekOCR(VLMModel):
    """Wrapper for the DeepSeek-OCR model."""

    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"

    def __init__(self, config: Union[ModelConfig, str, None] = None):
        """
        Initialize DeepSeekOCR model.

        Args:
            config: ModelConfig object, model_id string, or None for default.
        """
        if config is None:
            model_id = self.DEFAULT_MODEL_ID
            self.config = None
        elif isinstance(config, str):
            model_id = config
            self.config = None
        else:
            model_id = config.model_id
            self.config = config

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)

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
