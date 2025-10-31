import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from datasets import (
    Dataset,
    Features,
    Value,
    Image as HFImage,
)
from huggingface_hub import HfFolder

logger = logging.getLogger(__name__)


def read_json(file_path):
    """
    JSON 파일을 읽어 파이썬 객체로 반환합니다.

    Args:
        file_path (str): 읽을 JSON 파일의 경로.

    Returns:
        dict or list: JSON 파일의 내용을 담은 파이썬 객체.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"오류: 파일 '{file_path}'를 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError:
        logger.error(f"오류: '{file_path}' 파일이 올바른 JSON 형식이 아닙니다.")
        return None


import json


def save_json(data, file_path):
    """
    파이썬 객체를 JSON 파일로 저장합니다.

    Args:
        data (dict or list): 저장할 파이썬 객체.
        file_path (str): 저장할 JSON 파일의 경로.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"데이터가 '{file_path}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"파일 저장 중 오류가 발생했습니다: {e}")


def read_txt(file_path):
    """
    텍스트 파일의 모든 줄을 읽어 리스트로 반환합니다.

    Args:
        file_path (str): 읽을 텍스트 파일의 경로.

    Returns:
        list: 파일의 각 줄을 요소로 하는 문자열 리스트.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # 각 줄의 끝에 있는 개행 문자(\n) 제거
        return text
    except FileNotFoundError:
        logger.error(f"오류: 파일 '{file_path}'를 찾을 수 없습니다.")
        return None


def save_txt(file_path, text):
    """
    문자열 리스트를 텍스트 파일에 씁니다. 각 요소는 한 줄에 해당합니다.

    Args:
        lines (list): 파일에 쓸 문자열의 리스트.
        file_path (str): 저장할 텍스트 파일의 경로.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"데이터가 '{file_path}' 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"파일 저장 중 오류가 발생했습니다: {e}")


def upload_subset_to_hub(dataset_dir: str, repo_id: str, config_name: str):
    """
    지정된 디렉토리의 데이터를 특정 config(subset)으로 Hub 저장소에 업로드합니다.

    :param dataset_dir: 이미지와 metadata.jsonl이 있는 디렉토리 경로.
    :param repo_id: Hugging Face 저장소 ID (예: 'user/repo-name').
    :param config_name: 데이터셋의 config 이름 (예: 'single_line', 'document').
    """
    logger.info(f"\n▶ Subset '{config_name}'을(를) '{repo_id}' 저장소에 업로드 시작...")

    try:
        # Hugging Face 로그인 확인
        if HfFolder.get_token() is None:
            raise ConnectionError(
                "Hugging Face 로그인이 필요합니다. 'huggingface-cli login'을 실행해주세요."
            )

        dataset_path = Path(dataset_dir)
        metadata_path = dataset_path / "metadata.jsonl"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"'{metadata_path}' 파일을 찾을 수 없습니다. 업로드 중단."
            )

        image_paths: List[str] = []
        texts: List[str] = []
        prompts: List[str] = []

        # 메타데이터 로드
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 파일 경로를 dataset_dir을 기준으로 절대/상대 경로 조정 필요 (여기서는 현재 경로 기준 유지)
                image_paths.append(str(Path(data["file_name"])))
                texts.append(data["text"])
                prompts.append(
                    data.get("prompt", "")
                )  # prompt 필드가 없을 경우 빈 문자열 할당

        logger.info(
            f"  '{config_name}' subset: 총 {len(image_paths):,}개의 이미지-텍스트 쌍을 찾았습니다."
        )

        # Hugging Face Dataset 객체 생성
        dataset = Dataset.from_dict(
            {"image": image_paths, "text": texts, "prompt": prompts},
            features=Features(
                {"image": HFImage(), "text": Value("string"), "prompt": Value("string")}
            ),
        )

        # Hugging Face Hub에 업로드 (config_name 지정)
        dataset.push_to_hub(repo_id, config_name=config_name)

        logger.info(f"✔ Subset '{config_name}' 업로드 완료!")

    except ConnectionError as ce:
        logger.error(f"오류: {ce}")
    except FileNotFoundError as fnfe:
        logger.error(f"오류: {fnfe}")
    except Exception as e:
        logger.error(
            f"오류: Subset '{config_name}' 업로드 중 예상치 못한 오류 발생: {e}"
        )
