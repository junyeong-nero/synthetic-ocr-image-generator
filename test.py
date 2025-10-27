import sys
from pathlib import Path

sys.path.insert(0, "src")

from ssim_calculator import calculate_ssim


def main():
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"

    text_a = "의"
    text_b = "익"
    text_c = "잉"

    # 동일한 텍스트 비교
    ssim_score_same = calculate_ssim(text_a, text_b, font_path)
    print(
        f"'{text_a}'와 '{text_b}'의 SSIM 점수: {ssim_score_same}"
    )  # 1.0에 가까운 값이 나옵니다.

    # 다른 텍스트 비교
    ssim_score_diff = calculate_ssim(text_a, text_c, font_path)
    print(f"'{text_a}'와 '{text_c}'의 SSIM 점수: {ssim_score_diff}")


if __name__ == "__main__":
    main()
