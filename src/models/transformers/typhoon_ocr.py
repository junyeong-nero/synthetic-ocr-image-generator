"""Typhoon-OCR model wrapper."""

from typing import List

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from models.transformers.base import BaseTransformersOCR


class TyphoonOCR(BaseTransformersOCR):
    """Wrapper for the Typhoon-OCR model."""

    DEFAULT_MODEL_ID = "scb10x/typhoon-ocr1.5-2b"
    DEFAULT_MAX_TOKENS = 10000
    MAX_IMAGE_SIZE = 1800

    def _load_model(self, model_id: str) -> None:
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        """Resize image if dimensions exceed threshold.

        The model is trained with a fixed image dimension of 1800px.

        Args:
            img: PIL Image to resize.

        Returns:
            Resized PIL Image if needed, otherwise original.
        """
        width, height = img.size
        max_size = self.MAX_IMAGE_SIZE

        if width > 300 or height > 300:
            if width >= height:
                scale = max_size / float(width)
                new_size = (max_size, int(height * scale))
            else:
                scale = max_size / float(height)
                new_size = (int(width * scale), max_size)

            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img

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
        max_tokens = self._get_max_tokens()

        for prompt, image in zip(prompts, images):
            resized_image = self._resize_if_needed(image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resized_image},
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
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.append(output_text[0])

        return results
