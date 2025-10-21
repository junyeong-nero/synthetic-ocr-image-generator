from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import platform


def list_system_fonts():
    """
    시스템에서 사용 가능한 폰트 목록을 반환하는 함수

    Returns:
        list: 폰트 파일 경로 리스트
    """
    fonts = []
    system = platform.system()

    # 운영체제별 기본 폰트 디렉토리
    if system == "Windows":
        font_dirs = [
            "C:/Windows/Fonts",
            os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
        ]
    elif system == "Darwin":  # macOS
        font_dirs = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
            "/System/Library/Fonts/Supplemental",
        ]
    elif system == "Linux":
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]
    else:
        print(f"지원하지 않는 운영체제: {system}")
        return []

    # 폰트 파일 확장자
    font_extensions = (".ttf", ".otf", ".ttc", ".TTF", ".OTF", ".TTC")

    # 각 디렉토리에서 폰트 파일 검색
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for root, dirs, files in os.walk(font_dir):
                for file in files:
                    if file.endswith(font_extensions):
                        full_path = os.path.join(root, file)
                        fonts.append(full_path)

    return sorted(fonts)


def list_system_fonts_with_names():
    """
    시스템에서 사용 가능한 폰트를 이름과 경로로 반환하는 함수

    Returns:
        dict: {폰트명: 폰트경로} 형태의 딕셔너리
    """
    fonts = {}
    font_paths = list_system_fonts()

    for font_path in font_paths:
        font_name = os.path.basename(font_path)
        fonts[font_name] = font_path

    return fonts


def find_font(keyword):
    """
    키워드로 폰트를 검색하는 함수

    Args:
        keyword (str): 검색할 키워드 (예: 'arial', 'malgun', 'nanum')

    Returns:
        list: 키워드를 포함하는 폰트 경로 리스트
    """
    all_fonts = list_system_fonts()
    keyword_lower = keyword.lower()

    matching_fonts = [
        font for font in all_fonts if keyword_lower in os.path.basename(font).lower()
    ]

    return matching_fonts


def generate_text_image(
    text, font_path, background_color, bold=False, italic=False, tilt=0, shadow=False
):
    """
    텍스트를 이미지로 변환하는 함수

    Args:
        text (str): 렌더링할 텍스트
        font_path (str): 폰트 파일 경로
        background_color (tuple): 배경색 (R, G, B) 또는 (R, G, B, A)
        bold (bool): 볼드체 적용 여부
        italic (bool): 이탤릭체 적용 여부
        tilt (int): 기울기 각도 (도 단위)
        shadow (bool): 그림자 효과 적용 여부

    Returns:
        PIL.Image: 생성된 이미지 객체
    """

    # 폰트 크기 설정
    font_size = 80
    font = ImageFont.truetype(font_path, font_size)

    # 텍스트 크기 계산
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 여백 추가
    padding = 40
    img_width = text_width + padding * 2
    img_height = text_height + padding * 2

    # 기울기를 고려한 이미지 크기 조정
    if tilt != 0:
        img_width = int(img_width * 1.5)
        img_height = int(img_height * 1.5)

    # 이미지 생성
    if len(background_color) == 3:
        img = Image.new("RGB", (img_width, img_height), background_color)
    else:
        img = Image.new("RGBA", (img_width, img_height), background_color)

    draw = ImageDraw.Draw(img)

    # 텍스트 위치 계산 (중앙 정렬)
    x = (img_width - text_width) // 2
    y = (img_height - text_height) // 2

    # 그림자 효과
    if shadow:
        shadow_offset = 5
        shadow_color = (50, 50, 50, 180) if len(background_color) == 4 else (50, 50, 50)
        draw.text(
            (x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color
        )

    # 텍스트 색상 (배경과 대비되는 색상)
    if sum(background_color[:3]) > 384:  # 밝은 배경
        text_color = (0, 0, 0)
    else:  # 어두운 배경
        text_color = (255, 255, 255)

    # 기본 텍스트 렌더링
    draw.text((x, y), text, font=font, fill=text_color)

    # 볼드 효과 (텍스트를 약간 이동시켜 여러 번 그리기)
    if bold:
        for offset in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            draw.text((x + offset[0], y + offset[1]), text, font=font, fill=text_color)

    # 이탤릭 효과 (기울기 변환)
    if italic:
        img = img.transform(
            img.size, Image.AFFINE, (1, -0.3, 0, 0, 1, 0), resample=Image.BICUBIC
        )

    # 기울기 적용
    if tilt != 0:
        img = img.rotate(tilt, expand=True, fillcolor=background_color)
        # 회전 후 자르기
        img = img.crop(img.getbbox())

    return img


if __name__ == "__main__":
    # 폰트 목록 출력
    print("=== 시스템 폰트 검색 ===\n")

    # 방법 1: 모든 폰트 출력 (처음 10개만)
    all_fonts = list_system_fonts()
    print(f"총 {len(all_fonts)}개의 폰트 발견\n")
    print("처음 10개 폰트:")
    for i, font in enumerate(all_fonts, 1):
        print(f"{i}. {font}")

    # 방법 2: 특정 키워드로 검색
    print("\n=== 'arial' 검색 결과 ===")
    arial_fonts = find_font("arial")
    for font in arial_fonts:
        print(f"- {font}")

    # 방법 3: 폰트명과 경로 딕셔너리
    print("\n=== 'malgun' 검색 결과 ===")
    malgun_fonts = find_font("malgun")
    for font in malgun_fonts:
        print(f"- {font}")

    # 폰트를 찾아서 이미지 생성
    if arial_fonts:
        print("\n=== 이미지 생성 ===")
        font_path = arial_fonts[0]

        # 예제 1: 기본 텍스트
        img1 = generate_text_image(
            text="Hello World!", font_path=font_path, background_color=(255, 255, 255)
        )
        img1.save("text_image_1.png")

        # 예제 2: 볼드 + 그림자
        img2 = generate_text_image(
            text="안녕하세요",
            font_path=font_path,
            background_color=(100, 150, 200),
            bold=True,
            shadow=True,
        )
        img2.save("text_image_2.png")

        # 예제 3: 이탤릭 + 기울기
        img3 = generate_text_image(
            text="Italic & Tilted",
            font_path=font_path,
            background_color=(50, 50, 50),
            italic=True,
            tilt=15,
        )
        img3.save("text_image_3.png")

        # 예제 4: 모든 효과 적용
        img4 = generate_text_image(
            text="All Effects!",
            font_path=font_path,
            background_color=(200, 100, 100),
            bold=True,
            italic=True,
            tilt=-10,
            shadow=True,
        )
        img4.save("text_image_4.png")

        print(f"이미지 생성 완료! (사용된 폰트: {os.path.basename(font_path)})")
    else:
        print("\n폰트를 찾을 수 없습니다.")
