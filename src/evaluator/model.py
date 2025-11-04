import os
import torch
import tempfile
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import List


# --- Helper Function ---
def convert_img_to_path(image: Image.Image) -> str:
    """
    Pillow Image 객체를 임시 파일로 저장하고 해당 파일의 경로를 반환합니다.

    Args:
        image (Image.Image): 저장할 Pillow Image 객체.

    Returns:
        str: 이미지가 저장된 임시 파일의 전체 경로.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file, format="PNG")
        return temp_file.name


class DotsOCR:
    def __init__(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            "rednote-hilab/dots.ocr",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            "rednote-hilab/dots.ocr", trust_remote_code=True
        )

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        여러 개의 프롬프트와 이미지를 단일 배치로 받아 OCR을 수행합니다.
        배치 크기는 입력 리스트의 길이에 따라 결정됩니다.

        Args:
            prompts (List[str]): 각 이미지에 해당하는 프롬프트의 리스트.
            images (List[Image.Image]): OCR을 수행할 Pillow Image 객체의 리스트.

        Returns:
            List[str]: 각 이미지에서 추출된 텍스트의 리스트.
        """
        if len(prompts) != len(images):
            raise ValueError("프롬프트의 수와 이미지의 수가 일치해야 합니다.")

        if not prompts:
            return []

        print(f"--- 총 {len(prompts)}개의 데이터를 단일 배치로 처리합니다 ---")

        temp_image_paths = []
        messages = []

        for prompt, image in zip(prompts, images):
            image_path = convert_img_to_path(image)
            temp_image_paths.append(image_path)

            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages.append(message)

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]

        all_image_inputs = []
        all_video_inputs = []
        for msg in messages:
            image_inputs, video_inputs = process_vision_info(msg)
            all_image_inputs.extend(image_inputs)
            all_video_inputs.extend(video_inputs)

        inputs = self.processor(
            text=texts,
            images=all_image_inputs,
            videos=all_video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=24000)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # 모든 임시 파일 정리
        for path in temp_image_paths:
            try:
                os.remove(path)
            except OSError as e:
                print(f"임시 파일 삭제 오류: {e.filename} - {e.strerror}")

        return output_texts
