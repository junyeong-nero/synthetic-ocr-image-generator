import sys
import argparse

sys.path.insert(0, "src")
from pipeline import pipeline


def main():
    """명령줄 인자를 파싱하고 파이프라인을 실행합니다."""
    parser = argparse.ArgumentParser(description="Synthetic OCR Image Generator")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="데이터셋을 업로드할 Hugging Face Hub 리포지토리 ID",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        required=True,
        help="문자 유사성 DB 생성에 사용할 폰트 파일 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="생성된 모든 데이터를 저장할 기본 디렉토리",
    )
    parser.add_argument(
        "--lang", type=str, default="ko", help="생성할 데이터의 언어 코드 (예: ko, en)"
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=10000,
        help="코퍼스 생성 시 위키피디아에서 추출할 문장 수",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="생성할 이미지 수",
    )

    args = parser.parse_args()
    pipeline(**vars(args))


if __name__ == "__main__":
    main()
