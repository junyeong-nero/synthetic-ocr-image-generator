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
) -> Tuple[str, int]:
    """
    지정된 box 영역 안에 텍스트를 자동으로 줄바꿈하여 그립니다.
    그려진 텍스트와 실제 텍스트의 높이를 반환합니다.
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
        line_height_A = font.getbbox("A")[3] - font.getbbox("A")[1]
    except AttributeError:
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
    cell_padding = 10  # 셀 내부 여백

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
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))  # 높이 계산용 임시 draw 객체
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


def generate_table_images(
    corpus_path: str,
    num_images: int = 100,
    output_dir: str = "tables",
    num_rows: Tuple[int, int] = (5, 20),
    num_cols: Tuple[int, int] = (2, 5),
) -> Optional[str]:
    """
    다양한 행/열 개수를 가진 테이블 이미지를 생성합니다.

    :param corpus_path: 텍스트 코퍼스 파일 경로.
    :param num_images: 생성할 이미지 개수.
    :param output_dir: 이미지를 저장할 디렉토리.
    :param num_rows: 생성될 테이블의 행 개수 범위 (min, max).
    :param num_cols: 생성될 테이블의 열 개수 범위 (min, max).
    :return: 생성된 이미지 디렉토리 경로.
    """
    logger.info(
        f"\n'{corpus_path}' 파일을 사용하여 [테이블] 이미지 생성을 시작합니다. 목표 이미지 수: {num_images:,}"
    )
    logger.info(
        f"테이블 크기: {num_rows[0]}~{num_rows[1]}행, {num_cols[0]}~{num_cols[1]}열"
    )

    # --- Argument Validation ---
    if not (
        isinstance(num_rows, (list, tuple))
        and len(num_rows) == 2
        and num_rows[0] <= num_rows[1]
    ):
        logger.error(
            "오류: num_rows는 (min, max) 형식의 튜플이어야 하며 min <= max 여야 합니다."
        )
        return None
    if not (
        isinstance(num_cols, (list, tuple))
        and len(num_cols) == 2
        and num_cols[0] <= num_cols[1]
    ):
        logger.error(
            "오류: num_cols는 (min, max) 형식의 튜플이어야 하며 min <= max 여야 합니다."
        )
        return None

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    font_dir = Path("fonts")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]

    if not font_paths:
        logger.error(
            "오류: 'fonts' 디렉토리에 .ttf 폰트 파일이 없습니다. 작업을 중단합니다."
        )
        return None

    corpus = read_txt(corpus_path)
    if not corpus:
        logger.error(
            f"오류: '{corpus_path}' 파일이 비어있거나 읽을 수 없습니다. 작업을 중단합니다."
        )
        return None

    lines = [line[:10].strip() for line in corpus.splitlines() if line.strip()]
    if not lines:
        logger.error(
            f"오류: '{corpus_path}' 파일에 내용이 없습니다. 작업을 중단합니다."
        )
        return None

    metadata: List[Dict[str, str]] = []

    for i in range(num_images):
        font_path = random.choice(font_paths)
        font_size = random.randint(22, 32)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # 지정된 범위 내에서 랜덤 행/열 개수 결정
        current_num_cols = random.randint(*num_cols)
        current_num_rows = random.randint(*num_rows)

        table_data = []
        for _ in range(current_num_rows):
            row_data = []
            for _ in range(current_num_cols):
                num_words = random.randint(1, 10)
                cell_text = " ".join(random.choices(lines, k=num_words))
                row_data.append(cell_text)
            table_data.append(row_data)

        img, drawn_text = create_table_layout(table_data, font)

        if not drawn_text or not drawn_text.strip():
            continue

        filename = f"table_{i:04d}.png"
        filepath = output_path / filename
        img.save(filepath)
        metadata.append({"file_name": str(filepath), "text": drawn_text})

        if (i + 1) % 20 == 0:
            logger.info(f"... {i + 1} / {num_images} 테이블 이미지 생성 완료")

    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"총 {len(metadata):,}개의 테이블 이미지 및 메타데이터 생성을 완료했습니다.")
    return str(output_path)


# --- 이 아래 부분은 실제 실행 환경에 맞게 수정하여 사용하세요 ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def read_txt(path: str) -> str:
        """텍스트 파일을 읽어 내용을 반환합니다."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    font_dir = Path("fonts")
    if not font_dir.exists() or not any(font_dir.glob("*.ttf")):
        font_dir.mkdir(exist_ok=True)
        logger.warning(
            f"'{font_dir}' 디렉토리에 .ttf 폰트 파일이 없습니다. 한글 폰트를 추가해주세요."
        )

    corpus_path = Path("data/corpus.txt")
    if not corpus_path.parent.exists():
        logger.info(f"'{corpus_path.parent}' 디렉토리를 생성합니다.")
        corpus_path.parent.mkdir(parents=True)
    if not corpus_path.exists():
        logger.warning(f"'{corpus_path}' 파일이 없어 샘플 파일을 생성합니다.")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(
                "데이터\n분석\n테이블\n이미지\n생성\n파이썬\n라이브러리\n자동화\n프로그래밍\n인공지능\n"
            )

    # --- 테이블 이미지 생성 함수 호출 (새로운 인자 사용 예시) ---

    # 예시 1: 기본값 사용 (5~20행, 2~5열)
    # logger.info("기본 설정으로 테이블 이미지 생성을 시작합니다.")
    # generate_table_images(corpus_path=str(corpus_path), num_images=10, output_dir="table_images_default")

    # 예시 2: 좁은 범위의 행과 열 지정 (3~5행, 2~3열)
    logger.info("3~5행, 2~3열의 작은 테이블 이미지 생성을 시작합니다.")
    generate_table_images(
        corpus_path=str(corpus_path),
        num_images=10,
        output_dir="table_images_small",
        num_rows=(3, 10),
        num_cols=(2, 5),
    )

    # 예시 3: 고정된 크기의 테이블 지정 (정확히 10행 4열)
    # logger.info("정확히 10행 4열의 고정 크기 테이블 이미지 생성을 시작합니다.")
    # generate_table_images(
    #     corpus_path=str(corpus_path),
    #     num_images=10,
    #     output_dir="table_images_fixed",
    #     num_rows=(10, 10),
    #     num_cols=(4, 4)
    # )
