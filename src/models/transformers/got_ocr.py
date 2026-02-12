"""GOT-OCR model wrapper."""

from typing import List

from PIL import Image

from models.transformers.base import StandardTransformersOCR


class GotOCR(StandardTransformersOCR):
    """Wrapper for the GOT-OCR-2.0-hf model."""

    DEFAULT_MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"

    # StandardTransformersOCR handles _load_model using AutoModelForImageTextToText

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
        for image in images:
            inputs = self.processor(image, return_tensors="pt").to(self.model.device)

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
            )
            results.append(decoded)
        return results
