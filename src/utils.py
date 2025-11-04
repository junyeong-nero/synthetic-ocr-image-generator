import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd

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


def upload_subset_to_hub(repo_id: str, subset_dir: Path, config_name: str):
    """
    지정된 디렉토리의 데이터를 특정 config(subset)으로 Hub 저장소에 업로드합니다.
    metadata.jsonl에 있는 모든 필드를 동적으로 감지하여 컬럼으로 사용합니다.

    Args:
        repo_id (str): Hugging Face 저장소 ID (예: 'user/repo-name').
        subset_dir (Path): 이미지와 metadata.jsonl이 있는 디렉토리 경로.
        config_name (str): 데이터셋의 config 이름 (예: 'sentence_typos').
    """
    logger.info(f"\n▶ Subset '{config_name}'을(를) '{repo_id}' 저장소에 업로드 시작...")

    try:
        if HfFolder.get_token() is None:
            raise ConnectionError(
                "Hugging Face 로그인이 필요합니다. 'huggingface-cli login'을 실행해주세요."
            )

        metadata_path = subset_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"'{metadata_path}' 파일을 찾을 수 없습니다. 업로드 중단."
            )

        # --- [수정된 부분 1] ---
        # 1. metadata.jsonl의 첫 줄을 읽어 동적으로 컬럼과 **데이터 타입**을 파악합니다.
        feature_dict = {}
        column_names = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                logger.warning(
                    f"'{metadata_path}' 파일이 비어있습니다. 업로드를 건너뜁니다."
                )
                return

            sample_data = json.loads(first_line)
            if "file_name" not in sample_data:
                raise KeyError("'metadata.jsonl'에 필수 키인 'file_name'이 없습니다.")

            # 'file_name'은 항상 'image' 컬럼으로 처리
            feature_dict["image"] = HFImage()
            column_names.append("image")

            # 나머지 키들에 대해 타입을 확인하고 Features를 구성
            for key, value in sample_data.items():
                if key == "file_name":
                    continue

                value_type = type(value)
                if value_type is bool:
                    feature_dict[key] = Value("bool")
                elif value_type is Tuple or value_type is List:
                    feature_dict[key] = Value("string")
                elif value_type is int:
                    feature_dict[key] = Value("int64")
                elif value_type is float:
                    feature_dict[key] = Value("float32")
                elif value_type is str:
                    feature_dict[key] = Value("string")
                else:
                    # 지원하지 않는 타입은 경고를 출력하고 문자열로 처리
                    logger.warning(
                        f"'{key}'의 타입({value_type})을 지원하지 않습니다. 문자열로 처리합니다."
                    )
                    feature_dict[key] = Value("string")

                column_names.append(key)

        features = Features(feature_dict)
        logger.info(f"  감지된 컬럼 및 타입: {features}")
        # --- [수정 완료] ---

        # 2. 모든 데이터를 딕셔너리의 리스트 형태로 수집
        all_data: List[Dict[str, Any]] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                record = {"image": str(data["file_name"])}
                for key in feature_dict.keys():
                    if key != "image":
                        record[key] = data.get(key)

                all_data.append(record)

        if not all_data:
            logger.warning("처리할 유효한 데이터가 없습니다. 업로드를 중단합니다.")
            return

        logger.info(
            f"  '{config_name}' subset: 총 {len(all_data):,}개의 유효한 데이터를 찾았습니다."
        )

        # 3. Hugging Face Dataset 객체 생성
        df = pd.DataFrame(all_data)
        dataset = Dataset.from_pandas(df, features=features)

        # Hugging Face Hub에 업로드 (config_name 지정)
        dataset.push_to_hub(repo_id, config_name=config_name)

        logger.info(f"✔ Subset '{config_name}' 업로드 완료!")

    except (ConnectionError, FileNotFoundError, KeyError) as e:
        logger.error(f"오류: {e}")
    except Exception as e:
        logger.error(
            f"오류: Subset '{config_name}' 업로드 중 예상치 못한 오류 발생: {e}",
            exc_info=True,
        )
