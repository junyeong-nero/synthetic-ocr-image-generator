import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from PIL import Image, ImageDraw, ImageFont


def generate_single_text_image(
    text: str,
    font_path: str,
    background_color: Tuple[int, int, int],
    bold: bool = False,
    italic: bool = False,
    tilt: int = 0,
    shadow: bool = False,
) -> Image.Image:
    """
    단일 문장을 다양한 스타일로 렌더링한 이미지를 생성합니다.
    """
    font_size = 80
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # print(f"폰트 파일을 찾을 수 없습니다: {font_path}. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    # 텍스트 크기 측정
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 이미지 크기 계산
    padding = 40
    img_width, img_height = text_width + padding * 2, text_height + padding * 2
    if tilt != 0:
        # 회전 시 이미지 잘림 방지를 위해 크기 확장
        img_width, img_height = int(img_width * 1.5), int(img_height * 1.5)

    # 이미지 생성 및 그리기 준비
    img = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    x, y = (img_width - text_width) // 2, (img_height - text_height) // 2

    # 텍스트 색상 결정 (배경 밝기에 따라)
    text_color = (0, 0, 0) if sum(background_color[:3]) > 384 else (255, 255, 255)

    # 그림자 효과
    if shadow:
        shadow_offset = 5
        shadow_color = (50, 50, 50)
        draw.text(
            (x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color
        )

    # 기본 텍스트 렌더링
    draw.text((x, y), text, font=font, fill=text_color)

    # 볼드 효과
    if bold:
        for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            draw.text((x + offset[0], y + offset[1]), text, font=font, fill=text_color)

    # 이탤릭 효과
    if italic:
        # 기울임 변환
        img = img.transform(
            img.size, Image.AFFINE, (1, -0.3, 0, 0, 1, 0), resample=Image.BICUBIC
        )

    # 회전 효과
    if tilt != 0:
        img = img.rotate(tilt, expand=True, fillcolor=background_color)
        # 이미지 내용이 있는 영역으로 크롭
        img = img.crop(img.getbbox())

    return img


# --------------------------------------------------------------------------
# 1.4 단일 문장 이미지 생성 메인 함수
# --------------------------------------------------------------------------
def generate_single_line_images(
    corpus_path: str, num_images: int = 1000, output_dir: str = "images"
) -> Optional[str]:
    """
    주어진 코퍼스 파일에서 텍스트를 읽어 단일 문장 이미지를 생성하고,
    이미지-텍스트 쌍 메타데이터를 저장합니다.

    :param corpus_path: 텍스트 코퍼스 파일 경로.
    :param num_images: 생성할 이미지 개수.
    :param output_dir: 이미지를 저장할 디렉토리.
    :return: 생성된 이미지 디렉토리 경로.
    """
    print(
        f"\n'{corpus_path}' 파일을 사용하여 [단일 문장] 이미지 생성을 시작합니다. 목표 이미지 수: {num_images:,}"
    )

    # 폰트 경로 준비
    font_dir = Path("fonts")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]
    if not font_paths:
        print("오류: 'fonts' 디렉토리에 .ttf 폰트 파일이 없습니다. 작업을 중단합니다.")
        return None

    # 출력 디렉토리 준비
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 코퍼스 텍스트 로드
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            korean_texts = [line.strip() for line in f if line.strip()]
        if not korean_texts:
            raise ValueError("코퍼스 파일이 비어있습니다.")
    except FileNotFoundError:
        print(
            f"오류: '{corpus_path}' 파일을 찾을 수 없습니다. 먼저 코퍼스를 생성해야 합니다. 작업을 중단합니다."
        )
        return None
    except ValueError as e:
        print(f"오류: {e} 작업을 중단합니다.")
        return None

    # 배경색상 리스트
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

    image_text_pairs: List[Dict[str, str]] = []

    for idx in range(num_images):
        # 랜덤 스타일 설정
        font_path = random.choice(font_paths)
        text = random.choice(korean_texts)
        bg_color = random.choice(background_colors)
        bold = random.choice([True, False])
        italic = random.choice([True, False])
        tilt = random.randint(-15, 15)  # 단일 라인은 회전 각도를 줄임
        shadow = random.choice([True, False])

        # 이미지 생성
        img = generate_single_text_image(
            text=text,
            font_path=font_path,
            background_color=bg_color,
            bold=bold,
            italic=italic,
            tilt=tilt,
            shadow=shadow,
        )

        # 이미지 저장
        image_filename = f"image_{idx:04d}.png"
        image_path = output_path / image_filename
        img.save(image_path)

        # 메타데이터 저장
        image_text_pairs.append({"file_name": str(image_path), "text": text})

        if (idx + 1) % 100 == 0:
            print(f"... {idx + 1:,} / {num_images:,} 이미지 생성 완료")

    # 메타데이터 파일 저장 (JSONL 형식)
    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in image_text_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"총 {len(image_text_pairs):,}개의 이미지 및 메타데이터 생성을 완료했습니다.")
    return str(output_path)
