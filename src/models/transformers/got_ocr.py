"""GOT-OCR model wrapper."""

from typing import List

from PIL import Image

from src.models.transformers.base import StandardTransformersOCR


class GotOCR(StandardTransformersOCR):
    """Wrapper for the GOT-OCR-2.0-hf model."""

    DEFAULT_MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"

    # StandardTransformersOCR handles _load_model using AutoModelForImageTextToText

    @staticmethod
    def _should_enable_formatted_ocr(prompt: str) -> bool:
        if not prompt:
            return False

        normalized_prompt = prompt.lower()
        formatting_keywords = (
            "markdown",
            "latex",
            "formatted",
            "table",
            "formula",
            "math",
            "html",
        )
        return any(keyword in normalized_prompt for keyword in formatting_keywords)

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

        if len(prompts) != len(images):
            raise ValueError(
                "GotOCR requires prompts and images to have the same length "
                f"(got {len(prompts)} prompts, {len(images)} images)."
            )

        results = []
        for prompt, image in zip(prompts, images):
            processor_kwargs = {"return_tensors": "pt"}
            if self._should_enable_formatted_ocr(prompt):
                processor_kwargs["format"] = True

            inputs = self.processor(image, **processor_kwargs).to(self.model.device)

            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=max_tokens,
            )

            decoded = self.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.append(decoded)
        return results
