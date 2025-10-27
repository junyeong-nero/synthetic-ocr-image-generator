import re
from typing import List
from datasets import load_dataset


# --------------------------------------------------------------------------
# 1.1 텍스트 정제 함수 (기존과 동일)
# --------------------------------------------------------------------------
def clean_wiki_text(text: str) -> str:
    """위키피디아 텍스트에서 불필요한 마크업 및 특수 문자를 제거하고 정제합니다."""
    # [[링크|표시 텍스트]] -> 표시 텍스트
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r" ", text)
    # [[링크]] -> 공백
    text = re.sub(r"\[\[([^\]]+)\]\]", r" ", text)
    # URL 제거
    text = re.sub(r"https?://[^ ]+", "", text)
    # 굵게/이탤릭체 마크업 제거 ('{2,5})
    text = re.sub(r"'{2,5}", "", text)
    # 섹션 제목 제거 (== 제목 ==)
    text = re.sub(r"==+\s*(.*?)\s*==+", r" .", text)
    # 한글, 숫자, 공백, 마침표/물음표/느낌표 외 문자 제거
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.?!]", "", text)
    # 다중 공백을 단일 공백으로 변환하고 양쪽 공백 제거
    text = " ".join(text.split()).strip()
    return text


# --------------------------------------------------------------------------
# 1.2 위키피디아에서 코퍼스 파일을 생성하는 함수 (기존과 동일)
# --------------------------------------------------------------------------
def create_corpus_from_wiki(output_path: str, num_sentences: int = 5000):
    """
    위키미디어 데이터셋에서 한국어 문장을 수집하여 코퍼스 파일을 생성합니다.

    :param output_path: 코퍼스를 저장할 파일 경로.
    :param num_sentences: 수집할 목표 문장 수.
    """
    print(f"'{output_path}' 생성을 시작합니다. 목표 문장 수: {num_sentences:,}")

    try:
        # 스트리밍 모드로 데이터셋 로드
        dataset = load_dataset(
            "wikimedia/wikipedia", "20231101.ko", split="train", streaming=True
        )
        # 셔플 버퍼를 사용하여 데이터 순서 섞기 (스트리밍에서는 제한적)
        shuffled_dataset = dataset.shuffle(buffer_size=10000)
    except Exception as e:
        print(f"데이터셋 로드 중 오류 발생: {e}")
        return

    collected_sentences: List[str] = []
    for data in shuffled_dataset:
        if len(collected_sentences) >= num_sentences:
            break

        cleaned_text = clean_wiki_text(data["text"])
        # 문장 단위로 분리 (마침표, 물음표, 느낌표 뒤의 공백을 기준으로)
        sentences = re.split(r"(?<=[.?!])\s+", cleaned_text)

        for sentence in sentences:
            s = sentence.strip()
            # 10자 초과 100자 미만의 문장만 수집
            if 10 < len(s) < 100:
                collected_sentences.append(s)
                if len(collected_sentences) % 100 == 0:
                    print(
                        f"... {len(collected_sentences):,} / {num_sentences:,} 문장 수집 완료"
                    )
                if len(collected_sentences) >= num_sentences:
                    break

    # 파일에 문장 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in collected_sentences:
            f.write(sentence + "\n")

    print(
        f"'{output_path}' 파일에 총 {len(collected_sentences):,}개의 문장을 저장했습니다."
    )


# --------------------------------------------------------------------------
# 1.3 [추가된 함수] 한국어 모든 문자 코퍼스를 생성하는 함수
# --------------------------------------------------------------------------
def create_korean_char_corpus(output_path: str):
    """
    한글의 초성, 중성, 종성을 조합하여 이론적으로 가능한 모든 한글 음절 문자를
    포함하는 코퍼스 파일을 생성합니다. 개별 자음과 모음도 포함됩니다.

    :param output_path: 코퍼스를 저장할 파일 경로.
    """
    print(f"모든 한글 문자 코퍼스 '{output_path}' 생성을 시작합니다.")

    # 초성 (19개)
    CHOSUNG = [
        "ㄱ",
        "ㄲ",
        "ㄴ",
        "ㄷ",
        "ㄸ",
        "ㄹ",
        "ㅁ",
        "ㅂ",
        "ㅃ",
        "ㅅ",
        "ㅆ",
        "ㅇ",
        "ㅈ",
        "ㅉ",
        "ㅊ",
        "ㅋ",
        "ㅌ",
        "ㅍ",
        "ㅎ",
    ]
    # 중성 (21개)
    JUNGSUNG = [
        "ㅏ",
        "ㅐ",
        "ㅑ",
        "ㅒ",
        "ㅓ",
        "ㅔ",
        "ㅕ",
        "ㅖ",
        "ㅗ",
        "ㅘ",
        "ㅙ",
        "ㅚ",
        "ㅛ",
        "ㅜ",
        "ㅝ",
        "ㅞ",
        "ㅟ",
        "ㅠ",
        "ㅡ",
        "ㅢ",
        "ㅣ",
    ]
    # 종성 (28개, 받침 없음 '' 포함)
    JONGSUNG = [
        "",
        "ㄱ",
        "ㄲ",
        "ㄳ",
        "ㄴ",
        "ㄵ",
        "ㄶ",
        "ㄷ",
        "ㄹ",
        "ㄺ",
        "ㄻ",
        "ㄼ",
        "ㄽ",
        "ㄾ",
        "ㄿ",
        "ㅀ",
        "ㅁ",
        "ㅂ",
        "ㅄ",
        "ㅅ",
        "ㅆ",
        "ㅇ",
        "ㅈ",
        "ㅊ",
        "ㅋ",
        "ㅌ",
        "ㅍ",
        "ㅎ",
    ]

    all_korean_chars = []

    # 1. 모든 한글 음절(11,172자) 조합 생성
    # 유니코드 한글 코드 포인트 계산: (초성 인덱스 * 21 * 28) + (중성 인덱스 * 28) + 종성 인덱스 + 0xAC00('가')
    for i, _ in enumerate(CHOSUNG):
        for j, _ in enumerate(JUNGSUNG):
            for k, _ in enumerate(JONGSUNG):
                code_point = (i * 21 * 28) + (j * 28) + k + 0xAC00
                all_korean_chars.append(chr(code_point))

    # 2. 독립적인 자음 및 모음 추가
    all_korean_chars.extend(CHOSUNG)
    all_korean_chars.extend(JUNGSUNG)

    # 파일에 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for char in all_korean_chars:
            f.write(char + "\n")

    total_chars = len(all_korean_chars)
    print(f"'{output_path}' 파일에 총 {total_chars:,}개의 한글 문자를 저장했습니다.")
    print(f"(음절: {11172:,}개, 자음: {len(CHOSUNG):,}개, 모음: {len(JUNGSUNG):,}개)")


# --- 사용 예시 ---
if __name__ == "__main__":
    # 방법 1: 위키피디아에서 문장 기반 코퍼스 생성
    create_corpus_from_wiki("data/corpus.txt", num_sentences=10000)

    # 방법 2: 모든 한글 문자로 구성된 문자 기반 코퍼스 생성
    create_korean_char_corpus("data/korean_char_corpus.txt")
