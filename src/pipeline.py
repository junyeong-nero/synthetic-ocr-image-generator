import argparse
import logging
from pathlib import Path
from typing import Any

# 각 모듈이 실제로 존재한다고 가정합니다.
from corpus_generator import create_corpus_from_wiki
from character_similarity import generate_similar_chars_db
from generator.sentence_generator import generate_sentence_typos_images
from utils import upload_subset_to_hub

# 로거 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_paths_and_prerequisites(
    base_dir: Path, font_path: str, lang: str, num_sentences: int
) -> None:
    """
    파이프라인 실행에 필요한 기본 경로, 코퍼스, DB를 준비합니다.
    """
    corpus_path = base_dir / f"corpus_{lang}.txt"
    db_path = base_dir / f"char_similarity_db_{lang}.json"

    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[SETUP] 기본 디렉토리 준비 완료: '{base_dir}'")

    if not corpus_path.exists():
        logger.info(f"[CORPUS] 코퍼스를 찾을 수 없어 위키피디아에서 생성합니다...")
        create_corpus_from_wiki(
            output_path=corpus_path, lang=lang, num_sentences=num_sentences
        )
        logger.info(f"[CORPUS] 코퍼스 생성 완료: '{corpus_path}'")
    else:
        logger.info(f"[CORPUS] 기존 코퍼스 사용: '{corpus_path}'")

    if not db_path.exists():
        logger.info(f"[DB] 문자 유사성 DB를 찾을 수 없어 생성합니다...")
        generate_similar_chars_db(
            corpus_path=corpus_path, db_path=db_path, font_path=font_path
        )
        logger.info(f"[DB] 문자 유사성 DB 생성 완료: '{db_path}'")
    else:
        logger.info(f"[DB] 기존 문자 유사성 DB 사용: '{db_path}'")


def pipeline(
    repo_id: str,
    font_path: str,
    size: int,
    corpus_size: int,
    output_dir: str,
    lang: str,
    typo_ratio: float = 0.15,
    **kwargs: Any,  # kwargs는 argparse의 다른 인자를 받기 위해 유지
) -> None:
    """
    'Sentence Typos' 데이터 생성 및 업로드 파이프라인을 실행합니다.
    """
    logger.info("=" * 80)
    logger.info(" Synthetic OCR Datasets - Sentence Typos ".center(80))
    logger.info("=" * 80)

    # --- 1. 경로 설정 및 코퍼스/DB 준비 ---
    base_dir = Path(output_dir) / lang
    setup_paths_and_prerequisites(base_dir, font_path, lang, corpus_size)

    corpus_path = base_dir / f"corpus_{lang}.txt"
    db_path = base_dir / f"char_similarity_db_{lang}.json"

    # --- 2. 'Sentence Typos' 이미지 생성 ---
    task_name = "Sentence Typos"
    logger.info(f"\n--- [TASK] {task_name} 이미지 생성 시작 ---")

    if size <= 0:
        logger.warning("요청된 이미지 수가 0개이므로 파이프라인을 종료합니다.")
        return

    logger.info(f"요청된 이미지 수: {size}개")

    task_output_dir = base_dir / "images_sentence_typos"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = None
    try:
        # 생성 함수를 직접 호출합니다.
        generated_dir_path = generate_sentence_typos_images(
            corpus_path=corpus_path,
            db_path=db_path,
            lang=lang,
            num_images=size,
            output_dir=task_output_dir,
            typo_ratio=typo_ratio,
        )

        if generated_dir_path is None or not Path(generated_dir_path).exists():
            raise RuntimeError(
                "생성 함수가 유효한 디렉토리 경로를 반환하지 않았습니다."
            )

        generated_dir = Path(generated_dir_path)
        logger.info(f"✓ 성공: '{generated_dir}'에 이미지를 생성했습니다.")

    except Exception as e:
        logger.error(
            f"✗ 실패: {task_name} 이미지 생성 중 오류 발생: {e}", exc_info=True
        )
        return  # 오류 발생 시 파이프라인 중단

    # --- 3. 허깅페이스 허브에 업로드 ---
    if generated_dir:
        logger.info(f"\n--- [UPLOAD] 허깅페이스 허브에 업로드 시작: {repo_id} ---")
        config_name = "default"  # 단일 서브셋이므로 'default' 또는 'main'으로 명시

        try:
            logger.info(f"'{config_name}' 서브셋을 '{generated_dir}'에서 업로드 중...")
            upload_subset_to_hub(
                repo_id=repo_id, subset_dir=generated_dir, config_name="default"
            )
            logger.info(f"✓ 성공: '{config_name}' 서브셋을 업로드했습니다.")
        except Exception as e:
            logger.error(f"✗ 실패: 서브셋 업로드 중 오류 발생: {e}", exc_info=True)
    else:
        logger.warning("생성된 데이터셋이 없어 업로드를 건너뜁니다.")

    logger.info("\n" + " 파이프라인이 성공적으로 완료되었습니다! ".center(80, "="))
    logger.info(f"허브에서 데이터셋 확인: https://huggingface.co/datasets/{repo_id}")
