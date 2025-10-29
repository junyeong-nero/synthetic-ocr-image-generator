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
) -> Tuple[str, int]:
    """
    Draws text within a specified box area with automatic line wrapping.
    Returns the drawn text and the actual height of the text.
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
    """Creates an image in a table format."""
    width, height = 1240, 1754
    margin = 80
    cell_padding = 10  # Cell inner padding

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    num_rows = len(table_data)
    num_cols = len(table_data[0]) if num_rows > 0 else 0
    if num_cols == 0:
        return img, ""

    # 1. Calculate column widths (equal division)
    drawable_width = width - 2 * margin
    col_widths = [drawable_width // num_cols] * num_cols

    # 2. Calculate row heights (dynamically determined by content)
    row_heights = []
    temp_draw = ImageDraw.Draw(
        Image.new("RGB", (1, 1))
    )  # Temporary draw object for height calculation
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

    # 3. Draw the table
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
    Generates table images with a variable number of rows and columns.

    :param corpus_path: Path to the text corpus file.
    :param num_images: Number of images to generate.
    :param output_dir: Directory to save the images.
    :param num_rows: Range for the number of rows in the table (min, max).
    :param num_cols: Range for the number of columns in the table (min, max).
    :return: Path to the directory where images were generated.
    """
    logger.info(
        f"\nStarting [table] image generation using '{corpus_path}'. Target number of images: {num_images:,}"
    )
    logger.info(
        f"Table size: {num_rows[0]}-{num_rows[1]} rows, {num_cols[0]}-{num_cols[1]} columns"
    )

    # --- Argument Validation ---
    if not (
        isinstance(num_rows, (list, tuple))
        and len(num_rows) == 2
        and num_rows[0] <= num_rows[1]
    ):
        logger.error("Error: num_rows must be a tuple of (min, max) where min <= max.")
        return None
    if not (
        isinstance(num_cols, (list, tuple))
        and len(num_cols) == 2
        and num_cols[0] <= num_cols[1]
    ):
        logger.error("Error: num_cols must be a tuple of (min, max) where min <= max.")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    font_dir = Path("fonts")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]

    if not font_paths:
        logger.error(
            "Error: No .ttf font files found in the 'fonts' directory. Aborting."
        )
        return None

    corpus = read_txt(corpus_path)
    if not corpus:
        logger.error(f"Error: '{corpus_path}' is empty or cannot be read. Aborting.")
        return None

    lines = [line[:10].strip() for line in corpus.splitlines() if line.strip()]
    if not lines:
        logger.error(f"Error: No content found in '{corpus_path}'. Aborting.")
        return None

    metadata: List[Dict[str, str]] = []

    for i in tqdm(range(num_images), desc="Generating table images"):
        font_path = random.choice(font_paths)
        font_size = random.randint(22, 32)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # Determine random number of rows/columns within the specified range
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

    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(metadata):,} table images and metadata.")
    return str(output_path)
