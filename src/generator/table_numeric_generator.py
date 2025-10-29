import json
import random
import string  # 구두점 제거를 위해 추가
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from utils import read_txt

logger = logging.getLogger(__name__)


def draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box: Tuple[int, int, int, int],
) -> Tuple[str, int]:
    """
    지정된 상자 영역 내에서 자동 줄 바꿈으로 텍스트를 그립니다.
    그려진 텍스트와 텍스트의 실제 높이를 반환합니다.
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    words = text.split()

    lines: List[str] = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
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

    drawn_text_lines: List[str] = []
    y = y1
    total_text_height = 0

    try:
        # Pillow 10.0.0 이상
        line_height_A = font.getbbox("A")[3] - font.getbbox("A")[1]
    except AttributeError:
        # 이전 버전
        line_height_A = font.getsize("A")[1]
    line_spacing = line_height_A * 0.5

    for i, line in enumerate(lines):
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_height = line_bbox[3] - line_bbox[1]

        if y + line_height > y2:
            break

        if i > 0:
            y += line_spacing
            total_text_height += line_spacing

        draw.text((x1, y), line, font=font, fill=(0, 0, 0))
        y += line_height
        total_text_height += line_height
        drawn_text_lines.append(line)

    return " ".join(drawn_text_lines), int(total_text_height)


def create_table_layout(
    table_data: List[List[str]], font: ImageFont.ImageFont
) -> Tuple[Image.Image, str]:
    """테이블 형식의 이미지를 생성합니다."""
    width, height = 1240, 1754
    margin = 80
    cell_padding = 10

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    num_rows = len(table_data)
    num_cols = len(table_data[0]) if num_rows > 0 else 0
    if num_cols == 0:
        return img, ""

    # 1. 열 너비 계산 (균등 분할)
    drawable_width = width - 2 * margin
    col_widths = [drawable_width // num_cols] * num_cols

    # 2. 행 높이 계산 (내용에 따라 동적 결정)
    row_heights = []
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for row_data in table_data:
        max_height_in_row = 0
        for i, cell_text in enumerate(row_data):
            box_width = col_widths[i] - 2 * cell_padding
            words = cell_text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = f"{current_line} {word}".strip()
                line_bbox = temp_draw.textbbox((0, 0), test_line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                if line_width <= box_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            try:
                line_height_A = font.getbbox("A")[3] - font.getbbox("A")[1]
            except AttributeError:
                line_height_A = font.getsize("A")[1]
            line_spacing = line_height_A * 0.5

            total_text_height = sum(
                temp_draw.textbbox((0, 0), line, font=font)[3]
                - temp_draw.textbbox((0, 0), line, font=font)[1]
                for line in lines
            )
            total_text_height += (
                line_spacing * (len(lines) - 1) if len(lines) > 0 else 0
            )

            max_height_in_row = max(max_height_in_row, total_text_height)

        row_heights.append(max_height_in_row + 2 * cell_padding)

    # 3. 테이블 그리기
    all_drawn_text = []
    current_y = margin
    for i, row_data in enumerate(table_data):
        if current_y + row_heights[i] > height - margin:
            break

        current_x = margin
        for j, cell_text in enumerate(row_data):
            box = (
                current_x,
                current_y,
                current_x + col_widths[j],
                current_y + row_heights[i],
            )
            draw.rectangle(box, outline="black")
            text_box = (
                box[0] + cell_padding,
                box[1] + cell_padding,
                box[2] - cell_padding,
                box[3] - cell_padding,
            )
            drawn_text, _ = draw_text_in_box(draw, cell_text, font, text_box)
            all_drawn_text.append(drawn_text)
            current_x += col_widths[j]
        current_y += row_heights[i]

    return img, " ".join(filter(None, all_drawn_text))


def generate_table_numeric_images(
    corpus_path: str,
    num_images: int = 100,
    output_dir: str = "tables",
    num_rows: Tuple[int, int] = (5, 20),
    num_cols: Tuple[int, int] = (2, 5),
    lang: str = "ko",
    digits: int = 4,
) -> Optional[str]:
    """
    가변적인 행과 열 개수를 가진 테이블 이미지를 생성합니다.
    첫 행과 첫 열은 단어로, 나머지는 숫자로 채웁니다.

    :param corpus_path: 텍스트 코퍼스 파일 경로.
    :param num_images: 생성할 이미지 수.
    :param output_dir: 이미지를 저장할 디렉토리.
    :param num_rows: 테이블의 행 수 범위 (최소, 최대).
    :param num_cols: 테이블의 열 수 범위 (최소, 최대).
    :param lang: 폰트 선택을 위한 언어 코드.
    :param digits: 숫자 값의 소수점 이하 자릿수.
    :return: 이미지가 생성된 디렉토리 경로.
    """
    logger.info(
        f"\n[table] 이미지 생성을 시작합니다. 코퍼스: '{corpus_path}', 목표 이미지 수: {num_images:,}"
    )
    logger.info(
        f"테이블 크기: {num_rows[0]}-{num_rows[1]} 행, {num_cols[0]}-{num_cols[1]} 열"
    )

    # --- 인수 유효성 검사 ---
    if not (
        isinstance(num_rows, (list, tuple))
        and len(num_rows) == 2
        and num_rows[0] <= num_rows[1]
    ):
        logger.error(
            "오류: num_rows는 (최소, 최대) 형식의 튜플이어야 합니다 (최소 <= 최대)."
        )
        return None
    if not (
        isinstance(num_cols, (list, tuple))
        and len(num_cols) == 2
        and num_cols[0] <= num_cols[1]
    ):
        logger.error(
            "오류: num_cols는 (최소, 최대) 형식의 튜플이어야 합니다 (최소 <= 최대)."
        )
        return None

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    font_dir = Path(f"fonts/{lang}")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]

    if not font_paths:
        logger.error(
            f"오류: 'fonts/{lang}' 디렉토리에서 .ttf 폰트 파일을 찾을 수 없습니다. 중단합니다."
        )
        return None

    corpus = read_txt(corpus_path)
    if not corpus:
        logger.error(
            f"오류: '{corpus_path}'가 비어 있거나 읽을 수 없습니다. 중단합니다."
        )
        return None

    # --- 변경된 부분 시작 ---
    # 1. 구두점 제거
    translator = str.maketrans("", "", string.punctuation)
    clean_corpus = corpus.translate(translator)

    # 2. 구두점이 제거된 텍스트에서 단어 목록 생성
    words = list(set(clean_corpus.split()))
    if not words:
        logger.error(f"오류: '{corpus_path}'에서 단어를 찾을 수 없습니다. 중단합니다.")
        return None
    # --- 변경된 부분 끝 ---

    metadata: List[Dict[str, str]] = []

    for i in tqdm(range(num_images), desc="테이블 이미지 생성 중"):
        font_path = random.choice(font_paths)
        font_size = random.randint(12, 32)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        current_num_cols = random.randint(*num_cols)
        current_num_rows = random.randint(*num_rows)

        table_data = []
        for r in range(current_num_rows):
            row_data = []
            for c in range(current_num_cols):
                cell_text = ""
                if r == 0:
                    cell_text = random.choice(words)
                elif c == 0:
                    cell_text = random.choice(words)
                else:
                    # --- 변경된 부분 시작 ---
                    # 3. 지정된 자릿수(digits)까지 소수점 표현
                    number = random.uniform(0, 10000)
                    cell_text = f"{number:.{digits}f}"
                    # --- 변경된 부분 끝 ---
                row_data.append(cell_text)
            table_data.append(row_data)

        img, drawn_text = create_table_layout(table_data, font)

        if not drawn_text or not drawn_text.strip():
            continue

        filename = f"table_{i:04d}.png"
        filepath = output_path / filename
        img.save(filepath)
        metadata.append({"file_name": str(filepath), "text": drawn_text})

    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        f"성공적으로 {len(metadata):,}개의 테이블 이미지와 메타데이터를 생성했습니다."
    )
    return str(output_path)
