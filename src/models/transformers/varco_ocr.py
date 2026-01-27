"""VARCO-VISION OCR model wrapper."""

from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from models.transformers.base import BaseTransformersOCR
from utils import extract_tag


class VarcoOCR(BaseTransformersOCR):
    """Wrapper for the VARCO-VISION-2.0-1.7B-OCR model."""

    DEFAULT_MODEL_ID = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
    UPSCALE_TARGET_SIZE = 2304

    def _load_model(self, model_id: str) -> None:
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _upscale_image(self, image: Image.Image) -> Image.Image:
        """Upscales an image if its largest dimension is smaller than the target size."""
        w, h = image.size
        if max(w, h) >= self.UPSCALE_TARGET_SIZE:
            return image

        scaling_factor = self.UPSCALE_TARGET_SIZE / max(w, h)
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        return image.resize((new_w, new_h))

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        max_tokens = self._get_max_tokens()

        results = []
        for prompt, image in zip(prompts, images):
            image = self._upscale_image(image)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, torch.float16)

            generate_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generate_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
            ]
            output = self.processor.decode(
                generate_ids_trimmed[0], skip_special_tokens=False
            )
            output = " ".join([""] + extract_tag(output, tag="char"))
            results.append(output)
        return results
