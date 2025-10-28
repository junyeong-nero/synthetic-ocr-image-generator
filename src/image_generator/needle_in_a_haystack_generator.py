import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Import numpy, ImageFilter, and ImageEnhance.
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from utils import read_txt
from character_similarity import find_similar_chars, read_txt, read_json


def find_coeffs(
    source_coords: List[Tuple[int, int]], target_coords: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Calculates the 8 coefficients required for perspective distortion.
    Uses numpy to solve a system of linear equations in the form Ax = B.
    """
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([s[0], s[1], 1, 0, 0, 0, -t[0] * s[0], -t[0] * s[1]])
        matrix.append([0, 0, 0, s[0], s[1], 1, -t[1] * s[0], -t[1] * s[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(target_coords).reshape(8)

    # Solve the linear equations to get the coefficients.
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
    Generates an image by rendering a single sentence with various styles.
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

    # Adjust contrast (applied at the last step)
    if contrast:
        # Randomly set the contrast adjustment intensity (0.7 for decreased contrast, 1.5 for increased contrast)
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.7, 1.5)
        img = enhancer.enhance(factor)

    return img


def needle_in_a_haystack_generator(
    corpus_path, top_n=3, num_size=(5, 10), needle_ratio=0.2
):
    corpus = read_txt(corpus_path)
    db = read_json("data/char_similarity_db.json")

    result = []
    chars = set(list(corpus))
    for char in chars:

        size = random.randint(num_size[0], num_size[1])
        num_needle = int(size * size * needle_ratio)

        sim_chars = find_similar_chars(char, db, top_n=top_n)
        if not sim_chars:
            continue

        temp = [[char for _ in range(size)] for _ in range(size)]
        x = random.choices(list(range(size)), k=num_needle)
        y = random.choices(list(range(size)), k=num_needle)
        for _x, _y in zip(x, y):
            temp[_x][_y] = sim_chars[0][0]

        result.append("\n".join(["".join(row) for row in temp]))

    return result


def generate_needle_in_a_haystack_images(
    corpus_path: str,
    num_images: int = 1000,
    output_dir: str = "images",
    resolution_range: Tuple[int, int] = (70, 90),
) -> Optional[str]:
    """
    Reads text from a given corpus file, generates single-sentence images,
    and saves the image-text pair metadata.
    """
    print(
        f"\nStarting [single sentence] image generation using '{corpus_path}'. Target number of images: {num_images:,}"
    )

    font_dir = Path("fonts")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]
    if not font_paths:
        print("Error: No .ttf font files found in the 'fonts' directory. Aborting.")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    texts = needle_in_a_haystack_generator(corpus_path, top_n=3)

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
        text = random.choice(texts)
        bg_color = random.choice(background_colors)
        font_size = random.randint(*resolution_range)

        bold = random.choice([True, False])
        tilt = random.randint(-15, 15)
        shadow = random.choice([True, False])
        distortion = random.choice([True, False])
        blur = random.choice([True, False])
        contrast = random.choice([True, False])

        img = _generate_word_image(
            text=text,
            font_path=font_path,
            background_color=bg_color,
            font_size=font_size,
            # bold=bold,
            # tilt=tilt,
            # shadow=shadow,
            # distortion=distortion,
            # blur=blur,
            # contrast=contrast,
        )

        image_filename = f"image_{idx:04d}.png"
        image_path = output_path / image_filename
        img.save(image_path)

        image_text_pairs.append({"file_name": str(image_path), "text": text})

        if (idx + 1) % 100 == 0:
            print(f"... {idx + 1:,} / {num_images:,} images generated")

    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in image_text_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(image_text_pairs):,} images and metadata.")
    return str(output_path)


if __name__ == "__main__":
    generate_needle_in_a_haystack_images(corpus_path="korean_char_corpus.txt")
