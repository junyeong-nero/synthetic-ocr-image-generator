import json
import random
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils import read_txt, read_json
from generator.basic_generator import _generate_text_image
from character_similarity import find_similar_chars

logger = logging.getLogger(__name__)

BACKGROUND_COLORS: List[Tuple[int, int, int]] = [
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


def generate_typos(texts: List[str], db_path: str, top_n: int = 1) -> List[str]:
    """
    주어진 텍스트 목록에 대해 유사 문자를 기반으로 오타를 생성합니다.

    Args:
        texts: 오타를 생성할 원본 문자열의 리스트.
        db_path: 유사 문자 데이터베이스 파일의 경로.
        top_n: 각 문자에 대해 고려할 유사 문자의 최대 개수.

    Returns:
        생성된 모든 오타 문장들의 리스트.
    """
    db: Dict[str, Any] = read_json(db_path)

    def _generate_typos(word: str) -> List[str]:
        """하나의 단어에 대한 오타 후보들을 생성합니다."""
        if not word:  # 빈 문자열 처리
            return [""]

        n = len(word)
        # 단어의 맨 앞/뒤가 아닌, 문자 사이에서 오타를 생성하려면 randint(0, n-1) 사용
        index = random.randint(0, n - 1)

        # 숫자인 경우 오타를 생성하지 않음
        if word[index].isnumeric():
            return [word]

        original_char = word[index]
        word_list: List[str] = list(word)

        similar_chars: List[str] = find_similar_chars(original_char, db, top_n=top_n)

        result: List[str] = []
        for similar_char in similar_chars:
            word_list[index] = similar_char[0]
            result.append("".join(word_list))

        # 원래 단어도 후보에 포함시키려면 아래 주석 해제
        # if original_char not in similar_chars:
        #     result.append(word)

        return result

    all_generated_sentences: List[str] = []
    for text in texts:
        words: List[str] = text.split()

        # 각 단어에 대한 오타 후보들의 리스트 (예: [["안녕", "안녕"], ["하세여", "하새요"]])
        words_candidate: List[List[str]] = [_generate_typos(word) for word in words]

        # 생성된 오타 단어들의 모든 조합을 생성
        # (이 부분은 조합이 기하급수적으로 늘어날 수 있어 주의가 필요합니다)
        sentences: List[str] = [""]
        for typo_words in words_candidate:
            new_sentences: List[str] = [
                f"{sentence} {word}".strip()
                for sentence in sentences
                for word in typo_words
            ]
            sentences = new_sentences
        all_generated_sentences.extend(sentences)

    return all_generated_sentences


def generate_sentence_typos_images(
    corpus_path: str,
    db_path: str,
    num_images: int = 1000,
    output_dir: str = "images",
    resolution_range: Tuple[int, int] = (70, 90),
    bold: Optional[bool] = None,
    tilt: Optional[int] = None,
    shadow: Optional[bool] = None,
    distortion: Optional[bool] = None,
    blur: Optional[bool] = None,
    contrast: Optional[bool] = None,
    lang: str = "ko",
) -> Optional[str]:
    """
    Generates a dataset of images from a text corpus.

    Args:
        corpus_path: Path to the input text file.
        num_images: The total number of images to generate.
        output_dir: The directory to save the images and metadata.
        resolution_range: A tuple (min, max) for random font sizes.
        bold: Force bold effect. If None, it's randomly applied.
        tilt: Force a specific tilt angle. If None, it's randomly set.
        shadow: Force shadow effect. If None, it's randomly applied.
        distortion: Force distortion. If None, it's randomly applied.
        blur: Force blur. If None, it's randomly applied.
        contrast: Force contrast adjustment. If None, it's randomly applied.

    Returns:
        The path to the output directory, or None if an error occurs.
    """
    logger.info(
        f"\nStarting image generation from '{corpus_path}'. "
        f"Target: {num_images:,} images."
    )

    font_dir = Path(f"fonts/{lang}")
    font_paths = [str(p) for p in font_dir.glob("*.ttf")]
    if not font_paths:
        logger.error(
            f"Error: No .ttf font files found in 'fonts/{lang}' directory. Aborting."
        )
        return None

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    lines = read_txt(corpus_path).splitlines()
    korean_texts = [line[:20].strip() for line in lines if line.strip()]
    if not korean_texts:
        logger.error(f"Error: No text found in '{corpus_path}'. Aborting.")
        return None

    korean_texts = generate_typos(korean_texts, db_path, top_n=1)

    image_text_pairs: List[Dict[str, str]] = []

    for idx in tqdm(range(num_images), desc="Generating Images"):
        # --- Determine Parameters for this Image ---
        font_path = random.choice(font_paths)
        text = random.choice(korean_texts)
        bg_color = random.choice(BACKGROUND_COLORS)
        font_size = random.randint(*resolution_range)

        # Apply effects randomly if not specified by the user
        apply_bold = random.choice([True, False]) if bold is None else bold
        apply_tilt = random.randint(-15, 15) if tilt is None else tilt
        apply_shadow = random.choice([True, False]) if shadow is None else shadow
        apply_dist = random.choice([True, False]) if distortion is None else distortion
        apply_blur = random.choice([True, False]) if blur is None else blur
        apply_contrast = random.choice([True, False]) if contrast is None else contrast

        # --- Generate and Save Image ---
        img = _generate_text_image(
            text=text,
            font_path=font_path,
            background_color=bg_color,
            font_size=font_size,
            bold=apply_bold,
            tilt=apply_tilt,
            shadow=apply_shadow,
            distortion=apply_dist,
            blur=apply_blur,
            contrast=apply_contrast,
        )

        image_filename = f"image_{idx:05d}.png"
        image_filepath = output_path / image_filename
        img.save(image_filepath)

        image_text_pairs.append({"file_name": str(image_filepath), "text": text})

    # --- Save Metadata ---
    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in image_text_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Successfully generated {len(image_text_pairs):,} images.")
    return str(output_path)
