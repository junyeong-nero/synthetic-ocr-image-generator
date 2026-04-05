"""Nanonets-OCR2 model wrapper."""

from typing import List

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from src.models.transformers.base import BaseTransformersOCR, get_attn_implementation


class NanonetsOCR2(BaseTransformersOCR):
    """Wrapper for the Nanonets-OCR2 model."""

    DEFAULT_MODEL_ID = "nanonets/Nanonets-OCR2-3B"
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_PROMPT = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

    def _load_model(self, model_id: str) -> None:
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=get_attn_implementation(),
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)

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
            effective_prompt = prompt if prompt else self.DEFAULT_PROMPT

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": effective_prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            results.append(output_text[0])

        return results
