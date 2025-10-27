import re
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from datasets import (
    load_dataset,
    Dataset,
    Features,
    Value,
    Image as HFImage,
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from huggingface_hub import HfFolder


def upload_subset_to_hub(dataset_dir: str, repo_id: str, config_name: str):
    """
    지정된 디렉토리의 데이터를 특정 config(subset)으로 Hub 저장소에 업로드합니다.

    :param dataset_dir: 이미지와 metadata.jsonl이 있는 디렉토리 경로.
    :param repo_id: Hugging Face 저장소 ID (예: 'user/repo-name').
    :param config_name: 데이터셋의 config 이름 (예: 'single_line', 'document').
    """
    print(f"\n▶ Subset '{config_name}'을(를) '{repo_id}' 저장소에 업로드 시작...")

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

        # 메타데이터 로드
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 파일 경로를 dataset_dir을 기준으로 절대/상대 경로 조정 필요 (여기서는 현재 경로 기준 유지)
                image_paths.append(str(Path(data["file_name"])))
                texts.append(data["text"])

        print(
            f"  '{config_name}' subset: 총 {len(image_paths):,}개의 이미지-텍스트 쌍을 찾았습니다."
        )

        # Hugging Face Dataset 객체 생성
        dataset = Dataset.from_dict(
            {"image": image_paths, "text": texts},
            features=Features({"image": HFImage(), "text": Value("string")}),
        )

        # Hugging Face Hub에 업로드 (config_name 지정)
        dataset.push_to_hub(repo_id, config_name=config_name)

        print(f"✔ Subset '{config_name}' 업로드 완료!")

    except ConnectionError as ce:
        print(f"오류: {ce}")
    except FileNotFoundError as fnfe:
        print(f"오류: {fnfe}")
    except Exception as e:
        print(f"오류: Subset '{config_name}' 업로드 중 예상치 못한 오류 발생: {e}")
