import re
from pathlib import Path
import random
from datasets import load_dataset, Dataset, Image as HFImage
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)
import numpy as np
import json
from huggingface_hub import HfApi, HfFolder


# --------------------------------------------------------------------------
# 1. 텍스트 정제 함수 (기존과 동일)
# --------------------------------------------------------------------------
def clean_wiki_text(text):
    """
    위키피디아 텍스트에서 마크업, 불필요한 공백, 그리고 한글 외 텍스트를 제거하는 함수
    """
    text = re.sub(r"\[\[[^\]\|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?:\/\/[^\]]+\]", "", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"==+\s*(.*?)\s*==+", r"\1.", text)
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.?!]", "", text)
    text = " ".join(text.split())
    return text


# --------------------------------------------------------------------------
# 2. 위키피디아에서 코퍼스 파일을 생성하는 함수 (기존과 동일)
# --------------------------------------------------------------------------
def create_corpus_from_wiki(output_path, num_sentences=5000):
    """
    Hugging Face Wikipedia 데이터셋에서 텍스트를 가져와 정제한 후,
    지정된 경로에 텍스트 파일(코퍼스)로 저장합니다.
    """
    print(f"'{output_path}' 생성을 시작합니다. 목표 문장 수: {num_sentences}")
    dataset = load_dataset(
        "wikimedia/wikipedia", "20231101.ko", split="train", streaming=True
    )
    shuffled_dataset = dataset.shuffle(buffer_size=10000)
    collected_sentences = []
    for data in shuffled_dataset:
        if len(collected_sentences) >= num_sentences:
            break
        cleaned_text = clean_wiki_text(data["text"])
        sentences = re.split(r"(?<=[.?!])\s+", cleaned_text)
        for sentence in sentences:
            s = sentence.strip()
            if 10 < len(s) < 100:
                collected_sentences.append(s)
                if len(collected_sentences) % 100 == 0:
                    print(
                        f"... {len(collected_sentences)} / {num_sentences} 문장 수집 완료"
                    )
                if len(collected_sentences) >= num_sentences:
                    break
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in collected_sentences:
            f.write(sentence + "\n")
    print(
        f"'{output_path}' 파일에 총 {len(collected_sentences)}개의 문장을 저장했습니다."
    )


# --------------------------------------------------------------------------
# 3. 코퍼스 파일을 사용해 이미지를 생성하는 함수 (수정됨)
# --------------------------------------------------------------------------
def generate(corpus_path, num_images=1000):
    """
    주어진 코퍼스 파일에서 텍스트를 읽어 이미지를 생성하고,
    이미지-텍스트 쌍 메타데이터를 저장합니다.
    """
    print(
        f"\n'{corpus_path}' 파일을 사용하여 이미지 생성을 시작합니다. 목표 이미지 수: {num_images}"
    )

    font_dir = Path("fonts")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]
    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            korean_texts = [line.strip() for line in f if line.strip()]
        if not korean_texts:
            raise ValueError("코퍼스 파일이 비어있습니다.")
    except FileNotFoundError:
        print(
            f"오류: '{corpus_path}' 파일을 찾을 수 없습니다. 먼저 코퍼스를 생성해야 합니다."
        )
        return

    background_colors = [
        (255, 255, 255),
        (240, 240, 240),
        (255, 250, 240),
        (240, 255, 240),
        (240, 248, 255),
        (255, 240, 245),
        (245, 245, 220),
        (250, 250, 210),
        (230, 230, 250),
    ]

    image_text_pairs = []  # 이미지 파일명과 텍스트를 저장할 리스트

    for idx in range(num_images):
        font_path = random.choice(font_paths)
        text = random.choice(korean_texts)
        bg_color = random.choice(background_colors)
        bold = random.choice([True, False])
        italic = random.choice([True, False])
        tilt = random.randint(-45, 45)
        shadow = random.choice([True, False])

        img = generate_text_image(
            text=text,
            font_path=font_path,
            background_color=bg_color,
            bold=bold,
            italic=italic,
            tilt=tilt,
            shadow=shadow,
        )

        image_filename = f"image_{idx:04d}.png"
        image_path = output_dir / image_filename
        img.save(image_path)

        # 이미지 경로와 텍스트를 저장
        image_text_pairs.append({"file_name": str(image_path), "text": text})

        if (idx + 1) % 100 == 0:
            print(f"... {idx + 1} / {num_images} 이미지 생성 완료")

    # 메타데이터 파일 저장
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in image_text_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"총 {num_images}개의 이미지 및 메타데이터 생성을 완료했습니다.")
    return str(output_dir)


# --------------------------------------------------------------------------
# 4. 이미지 생성 함수 (기존과 동일)
# --------------------------------------------------------------------------
def generate_text_image(
    text, font_path, background_color, bold=False, italic=False, tilt=0, shadow=False
):
    font_size = 80
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"폰트 파일을 찾을 수 없습니다: {font_path}. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding = 40
    img_width = text_width + padding * 2
    img_height = text_height + padding * 2

    if tilt != 0:
        img_width = int(img_width * 1.5)
        img_height = int(img_height * 1.5)

    img = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    x = (img_width - text_width) // 2
    y = (img_height - text_height) // 2

    if shadow:
        shadow_offset = 5
        shadow_color = (50, 50, 50)
        draw.text(
            (x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color
        )

    text_color = (0, 0, 0) if sum(background_color[:3]) > 384 else (255, 255, 255)
    draw.text((x, y), text, font=font, fill=text_color)

    if bold:
        for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            draw.text((x + offset[0], y + offset[1]), text, font=font, fill=text_color)

    if italic:
        img = img.transform(
            img.size, Image.AFFINE, (1, -0.3, 0, 0, 1, 0), resample=Image.BICUBIC
        )

    if tilt != 0:
        img = img.rotate(tilt, expand=True, fillcolor=background_color)
        img = img.crop(img.getbbox())

    return img


# --------------------------------------------------------------------------
# 5. Hugging Face Hub에 데이터셋을 업로드하는 함수 (새로 추가)
# --------------------------------------------------------------------------
def upload_dataset_to_hub(dataset_dir, repo_id):
    """
    생성된 이미지와 텍스트 쌍을 Hugging Face Hub에 업로드합니다.

    Args:
        dataset_dir (str): 이미지와 metadata.jsonl 파일이 있는 디렉토리 경로
        repo_id (str): 업로드할 Hugging Face 저장소 ID (e.g., "your-username/your-dataset-name")
    """
    print(f"\n'{repo_id}' 저장소에 데이터셋 업로드를 시작합니다.")

    try:
        # Hugging Face 로그인 확인
        if HfFolder.get_token() is None:
            raise ConnectionError(
                "Hugging Face 로그인이 필요합니다. 'huggingface-cli login'을 실행해주세요."
            )

        # 데이터셋 생성
        dataset = load_dataset("imagefolder", data_dir=dataset_dir)

        # 데이터셋을 Hub에 푸시
        dataset.push_to_hub(repo_id, private=False)

        print(f"'{repo_id}' 저장소에 데이터셋 업로드를 완료했습니다.")
        print(f"Hub에서 확인: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"데이터셋 업로드 중 오류가 발생했습니다: {e}")


# --------------------------------------------------------------------------
# 메인 실행 블록 (수정됨)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    CORPUS_FILE_PATH = "corpus.txt"
    NUM_SENTENCES = 5000
    NUM_IMAGES = 1000

    # Hugging Face 저장소 ID 설정 (자신의 ID로 변경해야 합니다)
    # 예: "my-username/my-korean-ocr-dataset"
    HF_REPO_ID = "junyeong-nero/synthetic-ocr-bench"

    # 1. 코퍼스 파일 생성
    # create_corpus_from_wiki(output_path=CORPUS_FILE_PATH, num_sentences=NUM_SENTENCES)

    # 2. 생성된 코퍼스 파일을 사용하여 이미지 생성
    # image_directory = generate(corpus_path=CORPUS_FILE_PATH, num_images=NUM_IMAGES)

    # 3. 생성된 이미지와 텍스트 쌍을 Hugging Face Hub에 업로드
    # 위 1, 2번 단계 실행 후 생성된 'images' 폴더를 사용합니다.
    # image_directory 변수 대신 직접 경로를 지정해도 됩니다.
    upload_dataset_to_hub(dataset_dir="images", repo_id=HF_REPO_ID)
