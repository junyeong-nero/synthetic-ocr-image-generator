import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

from PIL import Image, ImageDraw, ImageFont
from utils import read_txt

logger = logging.getLogger(__name__)


def draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box: Tuple[int, int, int, int],
) -> str:
    """지정된 box 영역 안에 텍스트를 자동으로 줄바꿈하여 그립니다."""
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    words = text.split()

    lines: List[str] = []
    current_line = ""
    # 단어 단위로 줄바꿈 처리
    for word in words:
        test_line = f"{current_line} {word}".strip()
        # 텍스트 너비 측정
        line_bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = line_bbox[2] - line_bbox[0]

        if line_width <= box_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # 실제 이미지에 텍스트 그리기
    drawn_text_lines: List[str] = []
    y = y1
    # 'A' 텍스트 높이의 0.5배를 줄 간격으로 사용
    try:
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
    except AttributeError:  # 이전 PIL 버전 호환
        line_height = font.getsize("A")[1]
    line_spacing = line_height * 0.5

    for line in lines:
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_height = line_bbox[3] - line_bbox[1]

        if y + line_height > y2:
            break  # Box 높이를 벗어나면 중단

        draw.text((x1, y), line, font=font, fill=(0, 0, 0))
        y += line_height + line_spacing
        drawn_text_lines.append(line)

    return " ".join(drawn_text_lines)


# --------------------------------------------------------------------------
# 2.2 문서 레이아웃 생성 함수들
# --------------------------------------------------------------------------
def create_single_column_layout(
    text: str, font: ImageFont.ImageFont
) -> Tuple[Image.Image, str]:
    """세로 방향의 단일 컬럼 문서 이미지를 생성합니다 (A4 세로 비율)."""
    width, height = 1240, 1754
    margin = 100
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    box = (margin, margin, width - margin, height - margin)
    drawn_text = draw_text_in_box(draw, text, font, box)
    return img, drawn_text


def create_two_column_layout(
    text: str, font: ImageFont.ImageFont
) -> Tuple[Image.Image, str]:
    """세로 방향의 두 개 컬럼 문서 이미지를 생성합니다 (A4 세로 비율)."""
    width, height = 1240, 1754
    margin = 80
    gutter = 60  # 컬럼 사이 간격
    col_width = (width - 2 * margin - gutter) // 2

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 텍스트를 대략 반으로 나눔 (컬럼 경계를 맞추기 위해 실제로는 더 복잡해야 하나, 여기서는 단순화)
    split_point = len(text) // 2
    text1 = text[:split_point]
    text2 = text[split_point:]

    # 왼쪽 컬럼 (Column 1)
    box1 = (margin, margin, margin + col_width, height - margin)
    drawn_text1 = draw_text_in_box(draw, text1, font, box1)

    # 오른쪽 컬럼 (Column 2)
    box2 = (margin + col_width + gutter, margin, width - margin, height - margin)
    drawn_text2 = draw_text_in_box(draw, text2, font, box2)

    return img, f"{drawn_text1} {drawn_text2}".strip()


def create_horizontal_layout(
    text: str, font: ImageFont.ImageFont
) -> Tuple[Image.Image, str]:
    """가로 방향의 단일 컬럼 문서 이미지를 생성합니다 (A4 가로 비율)."""
    width, height = 1754, 1240
    margin = 100
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    box = (margin, margin, width - margin, height - margin)
    drawn_text = draw_text_in_box(draw, text, font, box)
    return img, drawn_text


def generate_document_images(
    corpus_path: str, num_images: int = 100, output_dir: str = "documents"
) -> Optional[str]:
    """
    다양한 레이아웃(단일/이중 컬럼, 가로/세로)의 문서 이미지를 생성합니다.

    :param corpus_path: 텍스트 코퍼스 파일 경로.
    :param num_images: 생성할 이미지 개수.
    :param output_dir: 이미지를 저장할 디렉토리.
    :return: 생성된 이미지 디렉토리 경로.
    """
    logger.info(
        f"\n'{corpus_path}' 파일을 사용하여 [문서] 이미지 생성을 시작합니다. 목표 이미지 수: {num_images:,}"
    )

    # 출력 디렉토리 및 폰트 경로 준비
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    font_dir = Path("fonts")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]

    if not font_paths:
        logger.error(
            "오류: 'fonts' 디렉토리에 .ttf 폰트 파일이 없습니다. 작업을 중단합니다."
        )
        return None

    # 코퍼스 텍스트 로드
    corpus = read_txt(corpus_path)
    lines = corpus.splitlines()
    korean_texts = [line[:20].strip() for line in lines if line.strip()]

    # 레이아웃 함수 리스트
    layout_functions = [
        create_single_column_layout,
        create_two_column_layout,
        create_horizontal_layout,
    ]

    metadata: List[Dict[str, str]] = []

    for i in range(num_images):
        # 1. 폰트 및 텍스트 덩어리 준비
        font_path = random.choice(font_paths)
        font_size = random.randint(28, 42)  # 문서에 적합한 폰트 크기
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # 여러 문장을 합쳐서 긴 텍스트 생성 (문서 내용)
        num_sentences_to_join = random.randint(15, 40)
        text_chunk = " ".join(random.choices(lines, k=num_sentences_to_join))

        # 2. 랜덤 레이아웃 선택 및 이미지 생성
        layout_func = random.choice(layout_functions)
        img, drawn_text = layout_func(text_chunk, font)

        # 3. 이미지 및 메타데이터 저장
        if not drawn_text or drawn_text.strip() == "":
            # print(f"... {i+1}/{num_images} 건너뜀 (내용 없음)")
            continue

        filename = f"doc_{i:04d}.png"
        filepath = output_path / filename
        img.save(filepath)

        metadata.append({"file_name": str(filepath), "text": drawn_text})

        if (i + 1) % 20 == 0:
            logger.info(f"... {i + 1} / {num_images} 문서 이미지 생성 완료")

    # 메타데이터 파일 저장 (JSONL 형식)
    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"총 {len(metadata):,}개의 문서 이미지 및 메타데이터 생성을 완료했습니다.")
    return str(output_path)
