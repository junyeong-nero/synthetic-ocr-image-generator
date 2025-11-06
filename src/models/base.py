import io
import base64

from PIL import Image
from typing import List, Dict


def generate_message(
    image: Image.Image,
    prompt: str = "OCR this image",
) -> List[Dict]:
    """
    Generates a message payload for a multimodal model.

    Args:
        image: The PIL Image to be included.
        prompt: The text prompt to accompany the image.

    Returns:
        A list of dictionaries representing the message structure.
    """
    buf = io.BytesIO()
    image = image.convert("RGB")
    image.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


class Model:
    """Base class for OCR models."""

    def __init__(self) -> None:
        pass

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Runs the OCR model on a batch of images and prompts.

        Args:
            prompts: A list of text prompts.
            images: A list of PIL Images.

        Returns:
            A list of OCR results as strings.
        """
        assert len(prompts) == len(images)
        return ["empty"] * len(prompts)


class vLLMModel(Model):
    """A wrapper for vLLM models."""

    def __init__(
        self, model_id, temperature=0, max_model_len=2048, max_tokens=1024, top_p=1.0
    ) -> None:
        super().__init__()

        from vllm import SamplingParams, LLM

        self.model = LLM(model_id, trust_remote_code=True, max_model_len=max_model_len)
        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p
        )

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        messages = [
            generate_message(image, prompt) for image, prompt in zip(images, prompts)
        ]
        outputs = self.model.chat(messages, self.sampling_params)
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            results.append(text)

        return results
