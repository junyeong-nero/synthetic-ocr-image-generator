import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, Any

from corpus_generator import create_corpus_from_wiki
from character_similarity import generate_similar_chars_db

from generator.document_generator import generate_document_images
from generator.sentence_generator import generate_sentence_images
from generator.sentence_typo_generator import generate_sentence_typos_images
from generator.document_typo_generator import generate_document_images as generate_document_typos_images
from generator.table_generator import generate_table_images
from generator.table_numeric_generator import generate_table_numeric_images
from generator.needle_in_a_haystack_generator import (
    generate_needle_in_a_haystack_images,
)

from utils import upload_subset_to_hub

logger = logging.getLogger(__name__)


def pipeline(
    repo_id: str,
    font_path: str,
    num_sentences: int,
    num_sentence_images: int,
    num_sentence_noise_images: int,
    num_sentence_typos_images: int,
    num_document_images: int,
    num_document_noise_images: int,
    num_document_typos_images: int,
    num_table_images: int,
    num_table_numeric_images: int,
    num_needle_images: int,
    output_dir: str,
    lang: str,
    **kwargs: Any,
) -> None:
    """
    전체 데이터 생성 및 업로드 파이프라인을 실행합니다.

    이 함수는 다음 과정을 조율합니다:
    1. 텍스트 코퍼스가 없는 경우 위키피디아에서 생성합니다.
    2. 문장, 문서, 표, "건초더미 속 바늘 찾기" 형식의 합성 이미지를 생성합니다.
    3. 생성된 데이터셋을 허깅페이스 허브에 업로드합니다.
    """
    logger.info("=" * 80)
    logger.info(" VDG: Visual Document Generation Pipeline ".center(80))
    logger.info("=" * 80)

    # --- 1. 경로 초기화 ---
    logger.info(f"\n[SETUP] '{output_dir}'에 경로 및 디렉토리 초기화 중...")
    base_dir = Path(output_dir) / lang
    db_path = base_dir / f"char_similarity_db_{lang}.json"
    corpus_path = base_dir / f"corpus_{lang}.txt"

    # --- 2. 생성 작업 통합 설정 ---
    # paths와 GENERATION_TASKS를 하나로 통합하여 관리
    GENERATION_CONFIG = {
        "sentence": {
            "name": "Sentence",
            "func": generate_sentence_images,
            "dir_suffix": "images_sentence",
            "config_name": "sentence",
            "args": {
                "lang": lang,
                "bold": False,
                "tilt": 0,
                "shadow": False,
                "distortion": False,
                "blur": False,
                "contrast": False,
            },
        },
        "sentence_noise": {
            "name": "Sentence Noise",
            "func": generate_sentence_images,
            "dir_suffix": "images_sentence_noise",
            "config_name": "sentence_noise",
            "args": {"lang": lang},
        },
        "sentence_typos": {
            "name": "Sentence Typos",
            "func": generate_sentence_typos_images,
            "dir_suffix": "images_sentence_typos",
            "config_name": "sentence_typos",
            "args": {"db_path": str(db_path), "lang": lang},
        },
        "document": {
            "name": "Document",
            "func": generate_document_images,
            "dir_suffix": "images_document",
            "config_name": "document",
            "args": {
                "lang": lang,
                "bold": False,
                "tilt": 0,
                "shadow": False,
                "distortion": False,
                "blur": False,
                "contrast": False,
            },
        },
        "document_noise": {
            "name": "Document Noise",
            "func": generate_document_images,
            "dir_suffix": "images_document_noise",
            "config_name": "document_noise",
            "args": {"lang": lang},
        },
        "document_typos": {
            "name": "Document Typos",
            "func": generate_document_typos_images,
            "dir_suffix": "images_document_typos",
            "config_name": "document_typos",
            "args": {"db_path": str(db_path), "lang": lang},
        },
        "table": {
            "name": "Table",
            "func": generate_table_images,
            "dir_suffix": "images_table",
            "config_name": "table",
            "args": {"lang": lang},
        },
        "table_numeric": {
            "name": "Table Numeric",
            "func": generate_table_numeric_images,
            "dir_suffix": "images_table_numeric",
            "config_name": "table_numeric",
            "args": {"lang": lang},
        },
        "needle": {
            "name": "Needle in a Haystack",
            "func": generate_needle_in_a_haystack_images,
            "dir_suffix": "images_needle_in_a_haystack",
            "config_name": "needle_in_a_haystack",
            "args": {"db_path": str(db_path), "lang": lang},
        },
    }

    # 파이프라인 인자와 설정 키를 매핑
    num_images_map = {
        "sentence": num_sentence_images,
        "sentence_noise": num_sentence_noise_images,
        "sentence_typos": num_sentence_typos_images,
        "document": num_document_images,
        "document_noise": num_document_noise_images,
        "document_typos": num_document_typos_images,
        "table": num_table_images,
        "table_numeric": num_table_numeric_images,
        "needle": num_needle_images,
    }

    # --- 3. 디렉토리 설정 ---
    for task_config in GENERATION_CONFIG.values():
        path = base_dir / task_config["dir_suffix"]
        path.mkdir(parents=True, exist_ok=True)

    if not corpus_path.parent.exists():
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[SETUP] 모든 디렉토리가 준비되었습니다.")

    # --- 4. 코퍼스 생성 ---
    if not corpus_path.exists():
        logger.info(
            f"\n[CORPUS] '{corpus_path}'에서 코퍼스를 찾을 수 없습니다. 위키피디아에서 생성합니다..."
        )
        create_corpus_from_wiki(
            output_path=str(corpus_path), lang=lang, num_sentences=num_sentences
        )
        logger.info("[CORPUS] 코퍼스를 성공적으로 생성했습니다.")
    else:
        logger.info(f"\n[CORPUS] 기존 코퍼스 '{corpus_path}'를 사용합니다.")

    if not db_path.exists():
        logger.info(
            f"\n[DB] '{db_path}'에서 문자 유사성 DB를 찾을 수 없습니다. 생성합니다..."
        )
        generate_similar_chars_db(
            corpus_path=str(corpus_path), db_path=str(db_path), font_path=font_path
        )
        logger.info("[DB] 문자 유사성 DB를 성공적으로 생성했습니다.")
    else:
        logger.info(f"\n[DB] 기존 문자 유사성 DB '{db_path}'를 사용합니다.")

    # --- 5. 이미지 생성 작업 ---
    generated_dirs: Dict[str, Path] = {}
    for task_key, task_config in GENERATION_CONFIG.items():
        name = task_config["name"]
        num_images = num_images_map.get(task_key, 0)

        logger.info(f"\n--- {name} 이미지 생성 ---")
        if num_images > 0:
            logger.info(f"{num_images}개의 이미지를 요청했습니다.")

            output_dir_path = base_dir / task_config["dir_suffix"]

            # 함수에 필요한 모든 인자 조합
            current_args = task_config["args"].copy()
            current_args["num_images"] = num_images
            current_args["output_dir"] = str(output_dir_path)

            generated_dir = task_config["func"](
                corpus_path=str(corpus_path), **current_args
            )

            if generated_dir is None:
                logger.error(f"오류: {name} 이미지 생성에 실패했습니다. 중단합니다.")
                return

            config_name = task_config["config_name"]
            generated_dirs[config_name] = Path(generated_dir)
            logger.info(f"'{generated_dir}'에 {name} 이미지를 성공적으로 생성했습니다.")
        else:
            logger.info(f"{name} 생성을 건너뜁니다 (요청된 이미지 0개).")

    # --- 6. 허깅페이스 허브에 업로드 ---
    logger.info(f"\n--- 허깅페이스 허브에 업로드: {repo_id} ---")
    if not generated_dirs:
        logger.info("생성된 데이터셋이 없어 업로드할 내용이 없습니다.")
    else:
        for config_name, dir_path in generated_dirs.items():
            logger.info(f"'{config_name}' 서브셋을 '{dir_path}'에서 업로드 중...")
            upload_subset_to_hub(str(dir_path), repo_id, config_name=config_name)
            logger.info(f"'{config_name}'을(를) 성공적으로 업로드했습니다.")

    logger.info("\n" + " 파이프라인이 성공적으로 완료되었습니다! ".center(80, "="))
    logger.info(f"허브에서 데이터셋 확인: https://huggingface.co/datasets/{repo_id}")
