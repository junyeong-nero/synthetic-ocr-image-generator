import json
import random
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
) -> str:
    """
    Draws text within a specified box area with automatic line wrapping.
    Modified to return the drawn text with newline characters (\n) between lines.
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    words = text.split()

    lines: List[str] = []
    current_line = ""
    # Process line breaks on a word-by-word basis
    for word in words:
        test_line = f"{current_line} {word}".strip()
        # Measure text width
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

    # Draw text on the actual image
    drawn_text_lines: List[str] = []
    y = y1
    # Use 0.5 times the height of the text 'A' as line spacing
    try:
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
    except AttributeError:  # Compatibility for older PIL versions
        line_height = font.getsize("A")[1]
    line_spacing = line_height * 0.5

    for line in lines:
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_height = line_bbox[3] - line_bbox[1]

        if y + line_height > y2:
            break  # Stop if it exceeds the box height

        draw.text((x1, y), line, font=font, fill=(0, 0, 0))
        y += line_height + line_spacing
        drawn_text_lines.append(line)

    # *** 수정된 부분: drawn_text_lines를 '\n'으로 연결하여 반환 ***
    return "\n".join(drawn_text_lines)


# --------------------------------------------------------------------------
# 2.2 Document Layout Generation Functions (Unchanged)
# --------------------------------------------------------------------------
def create_single_column_layout(
    text: str, font: ImageFont.ImageFont
) -> Tuple[Image.Image, str]:
    """Creates a single-column document image in portrait orientation (A4 portrait ratio)."""
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
    """Creates a two-column document image in portrait orientation (A4 portrait ratio)."""
    width, height = 1240, 1754
    margin = 80
    gutter = 60  # Gutter space between columns
    col_width = (width - 2 * margin - gutter) // 2

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Roughly split the text in half (in reality, this should be more complex to align column boundaries, but simplified here)
    split_point = len(text) // 2
    # NOTE: It's better to split on a space/sentence boundary, but keeping the original
    # simple split for consistency unless a more robust solution is required.
    text1 = text[:split_point]
    text2 = text[split_point:]

    # Left column (Column 1)
    box1 = (margin, margin, margin + col_width, height - margin)
    drawn_text1 = draw_text_in_box(draw, text1, font, box1)

    # Right column (Column 2)
    box2 = (margin + col_width + gutter, margin, width - margin, height - margin)
    drawn_text2 = draw_text_in_box(draw, text2, font, box2)

    # *** 수정된 부분: 두 컬럼 텍스트 사이에 한 칸 띄어쓰기 대신 줄 바꿈 문자를 추가하여 분리 (선택적) ***
    # V1: 띄어쓰기로 분리 (원본과 동일)
    # return img, f"{drawn_text1} {drawn_text2}".strip()
    # V2: 줄 바꿈 문자로 분리 (각 컬럼의 텍스트가 명확히 분리됨)
    return img, f"{drawn_text1}\n\n{drawn_text2}".strip()


def create_horizontal_layout(
    text: str, font: ImageFont.ImageFont
) -> Tuple[Image.Image, str]:
    """Creates a single-column document image in landscape orientation (A4 landscape ratio)."""
    width, height = 1754, 1240
    margin = 100
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    box = (margin, margin, width - margin, height - margin)
    drawn_text = draw_text_in_box(draw, text, font, box)
    return img, drawn_text


def generate_document_images(
    corpus_path: str,
    num_images: int = 100,
    output_dir: str = "documents",
    lang: str = "ko",
) -> Optional[str]:
    """
    Generates document images with various layouts (single/double column, landscape/portrait).

    :param corpus_path: Path to the text corpus file.
    :param num_images: Number of images to generate.
    :param output_dir: Directory to save the images.
    :return: Path to the directory where images were generated.
    """
    logger.info(
        f"\nStarting [document] image generation using '{corpus_path}'. Target number of images: {num_images:,}"
    )

    # Prepare output directory and font paths
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    font_dir = Path(f"fonts/{lang}")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]

    if not font_paths:
        logger.error(
            f"Error: No .ttf font files found in the 'fonts/{lang}' directory. Aborting."
        )
        return None

    # Load corpus text
    corpus = read_txt(corpus_path)
    lines = corpus.splitlines()
    korean_texts = [line[:20].strip() for line in lines if line.strip()]

    # List of layout functions
    layout_functions = [
        create_single_column_layout,
        create_two_column_layout,
        create_horizontal_layout,
    ]

    metadata: List[Dict[str, str]] = []

    for i in tqdm(range(num_images), desc="Generating document images"):
        # 1. Prepare font and text chunk
        font_path = random.choice(font_paths)
        font_size = random.randint(28, 42)  # Font size suitable for documents
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # Join multiple sentences to create a long text (document content)
        num_sentences_to_join = random.randint(15, 40)
        text_chunk = " ".join(random.choices(lines, k=num_sentences_to_join))

        # 2. Select a random layout and generate the image
        layout_func = random.choice(layout_functions)
        img, drawn_text = layout_func(text_chunk, font)

        # 3. Save image and metadata
        if not drawn_text or drawn_text.strip() == "":
            # logger.info(f"... {i+1}/{num_images} skipped (no content)")
            continue

        filename = f"doc_{i:04d}.png"
        filepath = output_path / filename
        img.save(filepath)

        metadata.append({"file_name": str(filepath), "text": drawn_text})

    # Save metadata file (JSONL format)
    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        f"Successfully generated {len(metadata):,} document images and metadata."
    )
    return str(output_path)
