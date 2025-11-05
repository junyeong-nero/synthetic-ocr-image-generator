from typing import List

from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..base import Model


class Qwen3VL(Model):
    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct"):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
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
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.extend(output_text)
        return results
