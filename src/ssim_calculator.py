import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm  # 진행 상황을 보여주기 위한 라이브러리
import os

# --- 가정: 이 함수들은 별도의 파일에 존재하고 임포트됨 ---
from utils import save_json, read_json, read_txt
from image_generator.text_generator import generate_text_image

# --- 1단계: 유사도 데이터베이스 구축 ---


def calculate_ssim_with_pil_images(imageA_pil, imageB_pil):
    """두 개의 Pillow Image 객체의 SSIM 점수를 계산합니다."""
    imageA = cv2.cvtColor(np.array(imageA_pil), cv2.COLOR_RGB2GRAY)
    imageB = cv2.cvtColor(np.array(imageB_pil), cv2.COLOR_RGB2GRAY)

    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

    score, _ = ssim(imageA, imageB, full=True)
    return score


def build_similarity_database(char_list, font_path, db_path, threshold=0.5):
    """
    문자 리스트를 기반으로 SSIM 유사도 데이터베이스를 구축합니다.

    Args:
        char_list (list): 비교할 문자들의 리스트.
        font_path (str): 폰트 파일 경로.
        db_path (str): 결과를 저장할 JSON 파일 경로.
        threshold (float): 이 값 이상의 유사도를 가질 때만 저장.
    """
    print("유사도 데이터베이스 구축을 시작합니다...")
    similarity_db = {}

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 수정된 부분: 이미지 생성 단계에 tqdm 추가
    print("1. 문자 이미지를 사전 생성합니다...")
    char_images = {
        char: generate_text_image(char, font_path, background_color=(255, 255, 255))
        for char in tqdm(char_list, desc="Generating images")
    }
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # 각 문자에 대한 이미지를 미리 생성하여 중복 계산 방지
    # char_images = {char: generate_text_image(char, font_path) for char in char_list}

    # tqdm을 사용하여 진행률 표시 (기존 코드)
    print("\n2. 문자 쌍의 유사도를 계산합니다...")
    for i in tqdm(range(len(char_list)), desc="Comparing characters"):
        char1 = char_list[i]
        img1 = char_images[char1]

        for j in range(i + 1, len(char_list)):
            char2 = char_list[j]
            img2 = char_images[char2]

            score = calculate_ssim_with_pil_images(img1, img2)

            if score >= threshold:
                # 양방향으로 관계를 저장
                if char1 not in similarity_db:
                    similarity_db[char1] = {}
                similarity_db[char1][char2] = score

                if char2 not in similarity_db:
                    similarity_db[char2] = {}
                similarity_db[char2][char1] = score

    save_json(similarity_db, db_path)
    print("데이터베이스 구축 완료.")
    return similarity_db


# --- 2단계: 데이터베이스를 이용한 유사 문자 검색 ---


def find_similar_chars(query_char, db, top_n=5):
    """
    데이터베이스에서 주어진 문자와 유사한 문자들을 찾습니다.

    Args:
        query_char (str): 검색할 기준 문자.
        db (dict): 미리 구축된 유사도 데이터베이스.
        top_n (int): 반환할 최대 결과 수.

    Returns:
        list: (유사 문자, 유사도 점수) 튜플의 리스트.
    """
    if query_char not in db:
        return []

    similar_items = sorted(
        db[query_char].items(), key=lambda item: item[1], reverse=True
    )
    return similar_items[:top_n]


def run(
    corpus_path="data/korean_char_corpus.txt",
    db_path="data/char_similarity_db.json",
    font_path="/System/Library/Fonts/Supplemental/AppleGothic.ttf",
):

    chars = read_txt(corpus_path)

    # 1. 데이터베이스 구축 (파일이 없을 경우에만 실행)
    if not os.path.exists(db_path):
        build_similarity_database(chars, font_path, db_path, threshold=0.6)

    # 2. 데이터베이스 로드 및 유사 문자 검색
    similarity_database = read_json(db_path)

    if similarity_database:
        search_char = "각"
        similar_results = find_similar_chars(search_char, similarity_database, top_n=5)

        print(f"\n'{search_char}' 문자와 비슷한 문자들 (Top 5):")
        if similar_results:
            for char, score in similar_results:
                print(f"- {char} (유사도: {score:.4f})")
        else:
            print(f"'{search_char}'에 대한 유사 문자를 찾을 수 없습니다.")


# --- 메인 실행 ---
if __name__ == "__main__":
    run()
