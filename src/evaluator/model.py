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
    def __init__(self, model_path: str) -> None:
        """
        모델과 프로세서를 초기화하고 로드합니다.

        Args:
            model_path (str): Hugging Face에 있는 모델의 경로.
        """
        print("모델을 로딩합니다...")
        # flash_attention_2는 Ampere 아키텍처 이상의 GPU에서만 지원됩니다.
        # 호환되지 않는 환경에서는 이 옵션을 제거하거나 'sdpa'로 변경해야 할 수 있습니다.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        print("모델 로딩 완료.")

    def run(
        self, prompts: List[str], images: List[Image.Image], batch_size: int = 8
    ) -> List[str]:
        """
        여러 개의 프롬프트와 이미지를 지정된 배치 크기로 나누어 OCR을 수행합니다.

        Args:
            prompts (List[str]): 각 이미지에 해당하는 프롬프트의 리스트.
            images (List[Image.Image]): OCR을 수행할 Pillow Image 객체의 리스트.
            batch_size (int, optional): 한 번에 처리할 배치의 크기. 기본값은 8.

        Returns:
            List[str]: 각 이미지에서 추출된 텍스트의 리스트.
        """
        if len(prompts) != len(images):
            raise ValueError("프롬프트의 수와 이미지의 수가 일치해야 합니다.")

        if not prompts:
            return []

        all_output_texts = []
        temp_image_paths = []

        # 전체 데이터를 지정된 배치 크기로 나눕니다.
        for i in range(0, len(prompts), batch_size):
            # 현재 배치의 데이터 슬라이싱
            batch_prompts = prompts[i : i + batch_size]
            batch_images = images[i : i + batch_size]

            print(
                f"--- 배치 {i // batch_size + 1} 처리 중 (크기: {len(batch_prompts)}) ---"
            )

            # --- (이하 로직은 기존과 거의 동일, 변수명만 batch_ 로 변경) ---

            batch_messages = []
            for prompt, image in zip(batch_prompts, batch_images):
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
                batch_messages.append(message)

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batch_messages
            ]

            all_image_inputs = []
            all_video_inputs = []
            for msg in batch_messages:
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

            # 현재 배치의 결과를 전체 결과 리스트에 추가
            all_output_texts.extend(output_texts)

        # 모든 임시 파일 정리
        for path in temp_image_paths:
            try:
                os.remove(path)
            except OSError as e:
                print(f"임시 파일 삭제 오류: {e.filename} - {e.strerror}")

        return all_output_texts
