"""Qwen2.5-VL model wrapper."""

from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from models.transformers.base import BaseTransformersOCR, get_attn_implementation


class Qwen25VL(BaseTransformersOCR):
    """Generic wrapper for Qwen2.5-VL-based OCR models."""

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    def _load_model(self, model_id: str) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation=get_attn_implementation(),
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation="eager",
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation="eager",
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        from qwen_vl_utils import process_vision_info

        results = []
        max_tokens = self._get_max_tokens()

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

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(device=self.device, dtype=self.dtype)
                if value.is_floating_point()
                else value.to(self.device)
                for key, value in inputs.items()
            }

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.append(output_text[0])

        return results
