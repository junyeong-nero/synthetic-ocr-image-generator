"""DotsOCR model wrapper."""

from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from models.transformers.base import BaseTransformersOCR, get_attn_implementation


class DotsOCR(BaseTransformersOCR):
    """Wrapper for the DotsOCR model."""

    DEFAULT_MODEL_ID = "./weights/DotsOCR"
    DEFAULT_MAX_TOKENS = 24000
    DEFAULT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

    def _load_model(self, model_id: str) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
            attn_implementation = get_attn_implementation()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation=attn_implementation,
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
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        from qwen_vl_utils import process_vision_info

        results = []
        max_tokens = self._get_max_tokens()

        for prompt, image in zip(prompts, images):
            effective_prompt = prompt if prompt else self.DEFAULT_PROMPT

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": effective_prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vision_info = process_vision_info(messages)
            image_inputs = vision_info[0]
            video_inputs = vision_info[1]
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
