import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# numpy와 ImageFilter, ImageEnhance를 import합니다.
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from utils import read_txt


def find_coeffs(
    source_coords: List[Tuple[int, int]], target_coords: List[Tuple[int, int]]
) -> np.ndarray:
    """
    원근 왜곡에 필요한 8개의 계수(coefficients)를 계산합니다.
    numpy를 사용하여 Ax = B 형태의 선형 방정식을 풉니다.
    """
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([s[0], s[1], 1, 0, 0, 0, -t[0] * s[0], -t[0] * s[1]])
        matrix.append([0, 0, 0, s[0], s[1], 1, -t[1] * s[0], -t[1] * s[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(target_coords).reshape(8)

    # 선형 방정식을 풀어 계수를 구합니다.
    res = np.linalg.solve(A, B)
    return res.flatten()


def _generate_word_image(
    text: str,
    font_path: str,
    background_color: Tuple[int, int, int],
    font_size: int = 80,
    bold: bool = False,
    tilt: int = 0,
    shadow: bool = False,
    distortion: bool = False,
    blur: bool = False,
    contrast: bool = False,
) -> Image.Image:
    """
    단일 문장을 다양한 스타일로 렌더링한 이미지를 생성합니다.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    padding = font_size // 2
    img_width, img_height = text_width + padding * 2, text_height + padding * 2
    if tilt != 0:
        img_width, img_height = int(img_width * 1.5), int(img_height * 1.5)

    img = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    x, y = (img_width - text_width) // 2, (img_height - text_height) // 2

    text_color = (0, 0, 0) if sum(background_color[:3]) > 384 else (255, 255, 255)

    if shadow:
        shadow_offset = max(1, font_size // 16)
        shadow_color = (50, 50, 50)
        draw.text(
            (x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color
        )

    draw.text((x, y), text, font=font, fill=text_color)

    if bold:
        for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            draw.text((x + offset[0], y + offset[1]), text, font=font, fill=text_color)

    if distortion:
        width, height = img.size
        source_coords = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        max_distort = font_size // 8
        target_coords = []
        for sx, sy in source_coords:
            dx = random.randint(-max_distort, max_distort)
            dy = random.randint(-max_distort, max_distort)
            target_coords.append((sx + dx, sy + dy))

        coeffs = find_coeffs(source_coords, target_coords)
        img = img.transform(
            (width, height),
            Image.PERSPECTIVE,
            coeffs,
            Image.BICUBIC,
            fillcolor=background_color,
        )

    if tilt != 0:
        img = img.rotate(tilt, expand=True, fillcolor=background_color)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

    if blur:
        blur_radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 대비(Contrast) 조정 (마지막 단계에서 적용)
    if contrast:
        # 대비 조절 강도를 무작위로 설정 (0.7은 대비 감소, 1.5는 대비 증가)
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.7, 1.5)
        img = enhancer.enhance(factor)

    return img


# --------------------------------------------------------------------------
# 1.4 단일 문장 이미지 생성 메인 함수
# --------------------------------------------------------------------------
def generate_word_images(
    corpus_path: str,
    num_images: int = 1000,
    output_dir: str = "images",
    resolution_range: Tuple[int, int] = (70, 90),
) -> Optional[str]:
    """
    주어진 코퍼스 파일에서 텍스트를 읽어 단일 문장 이미지를 생성하고,
    이미지-텍스트 쌍 메타데이터를 저장합니다.
    """
    print(
        f"\n'{corpus_path}' 파일을 사용하여 [단일 문장] 이미지 생성을 시작합니다. 목표 이미지 수: {num_images:,}"
    )

    font_dir = Path("fonts")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]
    if not font_paths:
        print("오류: 'fonts' 디렉토리에 .ttf 폰트 파일이 없습니다. 작업을 중단합니다.")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    corpus = read_txt(corpus_path)
    lines = corpus.splitlines()
    korean_texts = [line[:20].strip() for line in lines if line.strip()]

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
        font_path = random.choice(font_paths)
        text = random.choice(korean_texts)
        bg_color = random.choice(background_colors)
        font_size = random.randint(*resolution_range)

        bold = random.choice([True, False])
        tilt = random.randint(-15, 15)
        shadow = random.choice([True, False])
        distortion = random.choice([True, False])
        blur = random.choice([True, False])
        contrast = random.choice([True, False])  # 대비 효과 랜덤 선택

        img = _generate_word_image(
            text=text,
            font_path=font_path,
            background_color=bg_color,
            font_size=font_size,
            bold=bold,
            tilt=tilt,
            shadow=shadow,
            distortion=distortion,
            blur=blur,
            contrast=contrast,  # 인자 전달
        )

        image_filename = f"image_{idx:04d}.png"
        image_path = output_path / image_filename
        img.save(image_path)

        image_text_pairs.append({"file_name": str(image_path), "text": text})

        if (idx + 1) % 100 == 0:
            print(f"... {idx + 1:,} / {num_images:,} 이미지 생성 완료")

    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in image_text_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"총 {len(image_text_pairs):,}개의 이미지 및 메타데이터 생성을 완료했습니다.")
    return str(output_path)


if __name__ == "__main__":
    generate_word_images(corpus_path="korean_char_corpus.txt")
