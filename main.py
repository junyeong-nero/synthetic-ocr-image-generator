from utils import *
from pathlib import Path
import random


def generate():
    font_dir = Path("fonts")
    font_paths = [str(path) for path in font_dir.glob("*.ttf")]
    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True, parents=True)

    # 다양한 한국어 텍스트 샘플
    korean_texts = [
        "안녕하세요",
        "감사합니다",
        "사랑해요",
        "행복하세요",
        "좋은 하루",
        "환영합니다",
        "축하합니다",
        "고맙습니다",
        "미안합니다",
        "괜찮아요",
        "화이팅",
        "수고하셨습니다",
        "잘 먹겠습니다",
        "건강하세요",
        "평안하세요",
        "좋은 아침",
        "안녕히 가세요",
        "또 만나요",
        "잘 지내세요",
        "반갑습니다",
        "최고예요",
        "멋있어요",
        "예뻐요",
        "귀여워요",
        "멋지다",
        "대단해요",
        "훌륭합니다",
        "잘했어요",
        "최선을 다하세요",
        "꿈을 이루세요",
    ]

    # 배경색 옵션
    background_colors = [
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

    # 1000개 이미지 생성
    for idx in range(1000):
        # 랜덤으로 폰트 선택
        font_path = random.choice(font_paths)

        # 랜덤으로 텍스트 선택
        text = random.choice(korean_texts)

        # 랜덤으로 배경색 선택
        bg_color = random.choice(background_colors)

        # 랜덤으로 옵션 선택
        bold = random.choice([True, False])
        italic = random.choice([True, False])
        tilt = random.randint(-45, 45)  # -45도 ~ 45도

        img = generate_text_image(
            text=text,
            font_path=font_path,
            background_color=bg_color,
            bold=bold,
            italic=italic,
            tilt=tilt,
        )
        img.save(output_dir / f"image_{idx:04d}.png")

        # 진행상황 출력 (선택사항)
        if (idx + 1) % 100 == 0:
            print(f"{idx + 1}/1000 이미지 생성 완료")


# 사용 예제
if __name__ == "__main__":
    generate()
