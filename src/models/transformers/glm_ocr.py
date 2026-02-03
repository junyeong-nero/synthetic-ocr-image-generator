"""GLM-OCR model wrapper."""

from typing import List

from PIL import Image

from models.transformers.base import StandardTransformersOCR


class GlmOCR(StandardTransformersOCR):
    """Wrapper for the GLM-OCR model."""

    DEFAULT_MODEL_ID = "zai-org/GLM-OCR"
    DEFAULT_MAX_TOKENS = 8192

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
            ).to(self.model.device)
            inputs.pop("token_type_ids", None)

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            prompt_len = inputs["input_ids"].shape[1]
            decoded = self.processor.decode(
                generated_ids[0][prompt_len:], skip_special_tokens=False
            )
            results.append(decoded)
        return results
