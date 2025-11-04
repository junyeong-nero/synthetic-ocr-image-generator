import io
import base64

from PIL import Image
from typing import List, Dict
from vllm import SamplingParams, LLM


def generate_message(
    image: Image.Image,
    prompt: str = "OCR this image",
) -> List[Dict]:

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

    def __init__(self) -> None:
        pass

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        assert len(prompts) == len(images)
        return ["empty" * len(prompts)]


class vLLMModel(Model):

    def __init__(self, model_id, temperature=0, max_token=4096, top_p=1.0) -> None:
        super().__init__()
        self.model = LLM(model_id, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=temperature, max_token=max_token, top_p=top_p
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
