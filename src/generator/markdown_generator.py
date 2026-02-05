"""Markdown Generator Module for synthetic OCR markdown image generation.

This module provides comprehensive markdown document generation capabilities including:
- Various markdown elements (headers, paragraphs, lists, code blocks, tables, blockquotes)
- Multiple markdown templates (readme, technical_doc, blog_post, api_doc, tutorial)
- Layout variations (backgrounds, noise effects)
- Ground truth format with raw markdown text and rendered image
"""

import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from generator.base import BaseGenerator
from generator.data_provider import DataProvider

logger = logging.getLogger(__name__)


class MarkdownTemplate(Enum):
    """Markdown template types."""
    README = "readme"
    TECHNICAL_DOC = "technical_doc"
    BLOG_POST = "blog_post"
    API_DOC = "api_doc"
    TUTORIAL = "tutorial"


@dataclass
class MarkdownStyle:
    """Markdown rendering style options."""
    # Layout
    margin_top: int = 40
    margin_bottom: int = 40
    margin_left: int = 40
    margin_right: int = 40
    content_width: int = 600
    line_spacing: float = 1.5

    # Typography
    h1_font_size: int = 28
    h2_font_size: int = 22
    h3_font_size: int = 18
    body_font_size: int = 14
    code_font_size: int = 12

    # Colors
    text_color: Tuple[int, int, int] = (33, 33, 33)
    h1_color: Tuple[int, int, int] = (0, 0, 0)
    h2_color: Tuple[int, int, int] = (50, 50, 50)
    h3_color: Tuple[int, int, int] = (70, 70, 70)
    link_color: Tuple[int, int, int] = (0, 102, 204)
    code_bg_color: Tuple[int, int, int] = (245, 245, 245)
    code_text_color: Tuple[int, int, int] = (0, 0, 0)
    blockquote_color: Tuple[int, int, int] = (100, 100, 100)
    blockquote_border_color: Tuple[int, int, int] = (200, 200, 200)

    # Background
    background_color: Tuple[int, int, int] = (255, 255, 255)

    # Effects
    add_noise: bool = True
    add_blur: bool = False
    add_contrast: bool = False


class MarkdownDataGenerator:
    """Generates markdown content for various template types."""

    # Korean data
    KOREAN_TITLES = [
        "프로젝트 시작하기",
        "설치 가이드",
        "API 레퍼런스",
        "사용자 매뉴얼",
        "개발 환경 설정",
        "배포 가이드",
        "테스트 작성법",
        "성능 최적화",
    ]
    KOREAN_PARAGRAPHS = [
        "이 프로젝트는 사용자의 생산성을 높이기 위해 설계되었습니다.",
        "다양한 기능을 제공하며 확장 가능한 아키텍처를 가지고 있습니다.",
        "설치가 간단하고 문서화가 잘 되어 있어 빠르게 시작할 수 있습니다.",
        "커뮤니티의 지원을 받아 지속적으로 개선되고 있습니다.",
        "오픈소스로 제공되어 누구나 기여할 수 있습니다.",
    ]
    KOREAN_FEATURES = [
        "빠른 성능",
        "간편한 설치",
        "다양한 플러그인",
        "상세한 문서",
        "활발한 커뮤니티",
    ]
    KOREAN_CODE_COMMENTS = [
        "# 설정 파일을 로드합니다",
        "# 데이터베이스 연결을 설정합니다",
        "# 사용자 인증을 처리합니다",
    ]

    # English data
    ENGLISH_TITLES = [
        "Getting Started",
        "Installation Guide",
        "API Reference",
        "User Manual",
        "Development Setup",
        "Deployment Guide",
        "Writing Tests",
        "Performance Optimization",
    ]
    ENGLISH_PARAGRAPHS = [
        "This project is designed to enhance user productivity.",
        "It provides various features with an extensible architecture.",
        "Easy to install and well-documented for quick onboarding.",
        "Continuously improved with community support.",
        "Open source and open for contributions from anyone.",
    ]
    ENGLISH_FEATURES = [
        "Fast performance",
        "Easy installation",
        "Various plugins",
        "Detailed documentation",
        "Active community",
    ]
    ENGLISH_CODE_COMMENTS = [
        "# Load configuration file",
        "# Setup database connection",
        "# Handle user authentication",
    ]

    # Japanese data
    JAPANESE_TITLES = [
        "プロジェクトを始める",
        "インストールガイド",
        "APIリファレンス",
        "ユーザーマニュアル",
        "開発環境の設定",
        "デプロイガイド",
        "テストの書き方",
        "パフォーマンス最適化",
    ]
    JAPANESE_PARAGRAPHS = [
        "このプロジェクトは、ユーザーの生産性を向上させるために設計されています。",
        "拡張可能なアーキテクチャで様々な機能を提供しています。",
        "インストールが簡単で、ドキュメントが充実しているため、すぐに始められます。",
        "コミュニティのサポートにより継続的に改善されています。",
        "オープンソースで提供され、誰でも貢献できます。",
    ]
    JAPANESE_FEATURES = [
        "高速なパフォーマンス",
        "簡単なインストール",
        "豊富なプラグイン",
        "詳細なドキュメント",
        "活発なコミュニティ",
    ]
    JAPANESE_CODE_COMMENTS = [
        "# 設定ファイルを読み込みます",
        "# データベース接続を設定します",
        "# ユーザー認証を処理します",
    ]

    # Hindi data
    HINDI_TITLES = [
        "प्रोजेक्ट शुरू करना",
        "इंस्टॉलेशन गाइड",
        "एपीआई संदर्भ",
        "उपयोगकर्ता मैनुअल",
        "विकास वातावरण सेटअप",
        "डिप्लॉयमेंट गाइड",
        "टेस्ट लिखना",
        "प्रदर्शन अनुकूलन",
    ]
    HINDI_PARAGRAPHS = [
        "यह प्रोजेक्ट उपयोगकर्ता की उत्पादकता बढ़ाने के लिए डिज़ाइन किया गया है।",
        "यह विस्तार योग्य आर्किटेक्चर के साथ विभिन्न सुविधाएं प्रदान करता है।",
        "इंस्टॉल करना आसान है और अच्छी तरह से प्रलेखित है।",
        "समुदाय के समर्थन से लगातार सुधार किया जा रहा है।",
        "ओपन सोर्स और किसी के भी योगदान के लिए खुला है।",
    ]
    HINDI_FEATURES = [
        "तेज़ प्रदर्शन",
        "आसान इंस्टॉलेशन",
        "विभिन्न प्लगइन्स",
        "विस्तृत दस्तावेज़",
        "सक्रिय समुदाय",
    ]
    HINDI_CODE_COMMENTS = [
        "# कॉन्फ़िगरेशन फ़ाइल लोड करें",
        "# डेटाबेस कनेक्शन सेटअप करें",
        "# उपयोगकर्ता प्रमाणीकरण संभालें",
    ]

    def __init__(self, lang: str = "ko", data_provider: Optional[DataProvider] = None):
        self.lang = lang
        self.data = data_provider or DataProvider(lang=lang)

    def generate_markdown(
        self,
        template: MarkdownTemplate = MarkdownTemplate.README,
    ) -> str:
        """Generate markdown content based on template type."""
        gen_func = getattr(self, f"_generate_{template.value}")
        return gen_func()

    def _generate_readme(self) -> str:
        """Generate README-style markdown."""
        title = self.data.title()
        lines = [
            f"# {title}",
            "",
            self.data.paragraph(),
            "",
            "## " + ("기능" if self.lang == "ko" else "Features"),
            "",
        ]

        # Add feature list
        num_features = random.randint(3, 5)
        for feature in self.data.features(num_features):
            lines.append(f"- {feature}")
        lines.append("")

        # Add installation section
        lines.extend([
            "## " + ("설치" if self.lang == "ko" else "Installation"),
            "",
            "```bash",
            "pip install my-package",
            "```",
            "",
        ])

        # Add usage section
        lines.extend([
            "## " + ("사용법" if self.lang == "ko" else "Usage"),
            "",
            "```python",
            self.data.code_comment(),
            "import my_package",
            "",
            "client = my_package.Client()",
            "result = client.run()",
            "```",
            "",
        ])

        # Add quote
        lines.extend([
            "> " + self.data.paragraph(),
            "",
        ])

        return "\n".join(lines)

    def _generate_technical_doc(self) -> str:
        """Generate technical documentation style markdown."""
        title = self.data.title()
        lines = [
            f"# {title}",
            "",
            f"## " + ("개요" if self.lang == "ko" else "Overview"),
            "",
            self.data.paragraph(),
            "",
            f"## " + ("요구사항" if self.lang == "ko" else "Requirements"),
            "",
        ]

        # Add requirements list
        requirements = [
            "Python >= 3.8",
            "pip >= 21.0",
            "Git >= 2.0",
        ]
        for i, req in enumerate(requirements, 1):
            lines.append(f"{i}. {req}")
        lines.append("")

        # Add table
        lines.extend([
            f"## " + ("지원 버전" if self.lang == "ko" else "Supported Versions"),
            "",
            "| " + ("버전" if self.lang == "ko" else "Version") + " | " + ("상태" if self.lang == "ko" else "Status") + " |",
            "|--------|--------|",
            "| 1.0.x | " + ("지원됨" if self.lang == "ko" else "Supported") + " |",
            "| 2.0.x | " + ("지원됨" if self.lang == "ko" else "Supported") + " |",
            "| 3.0.x | " + ("개발중" if self.lang == "ko" else "In Development") + " |",
            "",
        ])

        # Add code block
        lines.extend([
            f"## " + ("설정" if self.lang == "ko" else "Configuration"),
            "",
            "```yaml",
            "config:",
            "  debug: false",
            "  log_level: INFO",
            "  max_connections: 100",
            "```",
            "",
        ])

        return "\n".join(lines)

    def _generate_blog_post(self) -> str:
        """Generate blog post style markdown."""
        title = self.data.title()
        date = self.data.date()

        lines = [
            f"# {title}",
            "",
            f"*" + ("작성일" if self.lang == "ko" else "Published") + f": {date}*",
            "",
            "---",
            "",
            self.data.paragraph(),
            "",
            f"## " + ("주요 내용" if self.lang == "ko" else "Key Points"),
            "",
        ]

        # Add bullet points
        for feature in self.data.features(3):
            lines.append(f"- **{feature}**: " + self.data.paragraph()[:50] + "...")
        lines.append("")

        # Add blockquote
        lines.extend([
            "> " + ("중요" if self.lang == "ko" else "Important") + ": " + self.data.paragraph(),
            "",
        ])

        # Add inline code
        lines.extend([
            ("이 기능은 " if self.lang == "ko" else "This feature uses ") + "`config.yaml`" + (" 파일을 사용합니다." if self.lang == "ko" else " file."),
            "",
        ])

        # Add link
        lines.extend([
            ("자세한 내용은 " if self.lang == "ko" else "For more details, see ") + "[" + ("공식 문서" if self.lang == "ko" else "official docs") + "](https://example.com)" + ("를 참조하세요." if self.lang == "ko" else "."),
            "",
        ])

        return "\n".join(lines)

    def _generate_api_doc(self) -> str:
        """Generate API documentation style markdown."""
        lines = [
            "# API " + ("레퍼런스" if self.lang == "ko" else "Reference"),
            "",
            f"## " + ("엔드포인트" if self.lang == "ko" else "Endpoints"),
            "",
            "### GET /api/users",
            "",
            ("사용자 목록을 조회합니다." if self.lang == "ko" else "Retrieve a list of users."),
            "",
            "**" + ("파라미터" if self.lang == "ko" else "Parameters") + ":**",
            "",
            "| " + ("이름" if self.lang == "ko" else "Name") + " | " + ("타입" if self.lang == "ko" else "Type") + " | " + ("설명" if self.lang == "ko" else "Description") + " |",
            "|------|------|-------------|",
            "| page | int | " + ("페이지 번호" if self.lang == "ko" else "Page number") + " |",
            "| limit | int | " + ("페이지당 항목 수" if self.lang == "ko" else "Items per page") + " |",
            "",
            "**" + ("응답 예시" if self.lang == "ko" else "Example Response") + ":**",
            "",
            "```json",
            "{",
            '  "users": [',
            '    {"id": 1, "name": "John"},',
            '    {"id": 2, "name": "Jane"}',
            "  ],",
            '  "total": 100',
            "}",
            "```",
            "",
            "### POST /api/users",
            "",
            ("새 사용자를 생성합니다." if self.lang == "ko" else "Create a new user."),
            "",
            "**" + ("요청 본문" if self.lang == "ko" else "Request Body") + ":**",
            "",
            "```json",
            "{",
            '  "name": "New User",',
            '  "email": "user@example.com"',
            "}",
            "```",
            "",
        ]

        return "\n".join(lines)

    def _generate_tutorial(self) -> str:
        """Generate tutorial style markdown."""
        title = self.data.title()
        lines = [
            f"# " + ("튜토리얼" if self.lang == "ko" else "Tutorial") + f": {title}",
            "",
            self.data.paragraph(),
            "",
            f"## " + ("시작하기 전에" if self.lang == "ko" else "Before You Begin"),
            "",
            ("다음 항목이 필요합니다:" if self.lang == "ko" else "You will need:"),
            "",
            "- [ ] Python 3.8+",
            "- [ ] pip",
            "- [ ] " + ("텍스트 에디터" if self.lang == "ko" else "Text editor"),
            "",
            f"## " + ("1단계" if self.lang == "ko" else "Step 1") + ": " + ("설치" if self.lang == "ko" else "Installation"),
            "",
            ("먼저 패키지를 설치합니다:" if self.lang == "ko" else "First, install the package:"),
            "",
            "```bash",
            "pip install example-package",
            "```",
            "",
            f"## " + ("2단계" if self.lang == "ko" else "Step 2") + ": " + ("설정" if self.lang == "ko" else "Configuration"),
            "",
            ("설정 파일을 생성합니다:" if self.lang == "ko" else "Create a configuration file:"),
            "",
            "```python",
            self.data.code_comment(),
            "config = {",
            '    "api_key": "your-api-key",',
            '    "debug": True',
            "}",
            "```",
            "",
            "> **" + ("팁" if self.lang == "ko" else "Tip") + "**: " + self.data.paragraph()[:60],
            "",
            f"## " + ("3단계" if self.lang == "ko" else "Step 3") + ": " + ("실행" if self.lang == "ko" else "Run"),
            "",
            "```bash",
            "python main.py",
            "```",
            "",
            ("예상 출력:" if self.lang == "ko" else "Expected output:"),
            "",
            "```",
            "Success! Server running on http://localhost:8000",
            "```",
            "",
        ]

        return "\n".join(lines)


class MarkdownRenderer:
    """Renders markdown content to images."""

    def __init__(self, font_path: str, style: MarkdownStyle = None):
        self.style = style or MarkdownStyle()
        self.font_path = font_path

        try:
            self.body_font = ImageFont.truetype(font_path, self.style.body_font_size)
            self.h1_font = ImageFont.truetype(font_path, self.style.h1_font_size)
            self.h2_font = ImageFont.truetype(font_path, self.style.h2_font_size)
            self.h3_font = ImageFont.truetype(font_path, self.style.h3_font_size)
            self.code_font = ImageFont.truetype(font_path, self.style.code_font_size)
        except IOError:
            logger.warning(f"Font '{font_path}' not found. Using default.")
            self.body_font = ImageFont.load_default()
            self.h1_font = self.body_font
            self.h2_font = self.body_font
            self.h3_font = self.body_font
            self.code_font = self.body_font

    def render(self, markdown_text: str) -> Image.Image:
        """Render markdown text to image."""
        lines = markdown_text.split("\n")
        style = self.style

        # First pass: calculate required height
        total_height = style.margin_top + style.margin_bottom
        line_heights = []

        for line in lines:
            height = self._get_line_height(line)
            line_heights.append(height)
            total_height += height

        # Create image
        width = style.margin_left + style.content_width + style.margin_right
        height = max(total_height, 200)

        img = Image.new("RGB", (width, int(height)), style.background_color)
        draw = ImageDraw.Draw(img)

        # Second pass: render content
        current_y = style.margin_top
        in_code_block = False
        code_block_start_y = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Handle code block markers
            if stripped.startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_block_start_y = current_y
                else:
                    # Draw code block background
                    draw.rectangle(
                        [
                            style.margin_left - 5,
                            code_block_start_y - 5,
                            style.margin_left + style.content_width + 5,
                            current_y + 5,
                        ],
                        fill=style.code_bg_color,
                    )
                    # Redraw code lines on top of background
                    in_code_block = False

                current_y += line_heights[i]
                continue

            if in_code_block:
                current_y = self._draw_code_line(draw, line, current_y, style)
            elif stripped.startswith("# "):
                current_y = self._draw_h1(draw, stripped[2:], current_y, style)
            elif stripped.startswith("## "):
                current_y = self._draw_h2(draw, stripped[3:], current_y, style)
            elif stripped.startswith("### "):
                current_y = self._draw_h3(draw, stripped[4:], current_y, style)
            elif stripped.startswith("> "):
                current_y = self._draw_blockquote(draw, stripped[2:], current_y, style)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                current_y = self._draw_list_item(draw, stripped[2:], current_y, style, ordered=False)
            elif stripped.startswith("- [ ]") or stripped.startswith("- [x]"):
                checked = stripped.startswith("- [x]")
                current_y = self._draw_checkbox_item(draw, stripped[6:], current_y, style, checked)
            elif len(stripped) > 0 and stripped[0].isdigit() and ". " in stripped:
                idx = stripped.index(". ")
                current_y = self._draw_list_item(draw, stripped[idx+2:], current_y, style, ordered=True, number=stripped[:idx])
            elif stripped.startswith("|"):
                current_y = self._draw_table_row(draw, stripped, current_y, style)
            elif stripped == "---" or stripped == "***":
                current_y = self._draw_horizontal_rule(draw, current_y, style)
            elif stripped.startswith("*") and stripped.endswith("*"):
                current_y = self._draw_italic(draw, stripped.strip("*"), current_y, style)
            elif stripped:
                current_y = self._draw_paragraph(draw, stripped, current_y, style)
            else:
                current_y += int(self.style.body_font_size * 0.5)

        # Apply effects
        img = self._apply_effects(img, style)

        return img

    def _get_line_height(self, line: str) -> int:
        """Calculate height needed for a line."""
        stripped = line.strip()
        base_spacing = int(self.style.line_spacing * self.style.body_font_size)

        if stripped.startswith("# "):
            return int(self.style.h1_font_size * self.style.line_spacing) + 10
        elif stripped.startswith("## "):
            return int(self.style.h2_font_size * self.style.line_spacing) + 8
        elif stripped.startswith("### "):
            return int(self.style.h3_font_size * self.style.line_spacing) + 6
        elif stripped.startswith("```"):
            return 5
        elif stripped.startswith("> "):
            return base_spacing + 10
        elif stripped == "---" or stripped == "***":
            return 20
        elif stripped:
            return base_spacing
        else:
            return int(self.style.body_font_size * 0.5)

    def _draw_h1(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw H1 header."""
        draw.text((style.margin_left, y), text, font=self.h1_font, fill=style.h1_color)
        # Draw underline
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h1_font)
        line_y = bbox[3] + 5
        draw.line([(style.margin_left, line_y), (style.margin_left + style.content_width, line_y)],
                  fill=style.h2_color, width=2)
        return line_y + 15

    def _draw_h2(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw H2 header."""
        draw.text((style.margin_left, y), text, font=self.h2_font, fill=style.h2_color)
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h2_font)
        return bbox[3] + 12

    def _draw_h3(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw H3 header."""
        draw.text((style.margin_left, y), text, font=self.h3_font, fill=style.h3_color)
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h3_font)
        return bbox[3] + 10

    def _draw_paragraph(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw paragraph text with word wrapping."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=self.body_font)
            if bbox[2] - bbox[0] <= style.content_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        for line in lines:
            # Handle inline code
            if "`" in line:
                y = self._draw_inline_code_line(draw, line, y, style)
            else:
                draw.text((style.margin_left, y), line, font=self.body_font, fill=style.text_color)
                y += int(style.body_font_size * style.line_spacing)

        return y + 5

    def _draw_inline_code_line(self, draw: ImageDraw.ImageDraw, line: str, y: int, style: MarkdownStyle) -> int:
        """Draw a line that may contain inline code."""
        x = style.margin_left
        parts = line.split("`")

        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code part
                bbox = draw.textbbox((x, y), part, font=self.code_font)
                draw.rectangle([x - 2, y - 1, bbox[2] + 2, bbox[3] + 1], fill=style.code_bg_color)
                draw.text((x, y), part, font=self.code_font, fill=style.code_text_color)
                x = bbox[2] + 4
            else:  # Normal text
                draw.text((x, y), part, font=self.body_font, fill=style.text_color)
                bbox = draw.textbbox((x, y), part, font=self.body_font)
                x = bbox[2]

        return y + int(style.body_font_size * style.line_spacing)

    def _draw_code_line(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw a line of code."""
        draw.text((style.margin_left + 10, y), text, font=self.code_font, fill=style.code_text_color)
        return y + int(style.code_font_size * style.line_spacing)

    def _draw_blockquote(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw blockquote."""
        # Draw left border
        draw.line(
            [(style.margin_left, y), (style.margin_left, y + style.body_font_size + 10)],
            fill=style.blockquote_border_color,
            width=3,
        )
        # Draw text
        draw.text(
            (style.margin_left + 15, y),
            text,
            font=self.body_font,
            fill=style.blockquote_color,
        )
        return y + int(style.body_font_size * style.line_spacing) + 10

    def _draw_list_item(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        y: int,
        style: MarkdownStyle,
        ordered: bool = False,
        number: str = None,
    ) -> int:
        """Draw list item."""
        marker = f"{number}." if ordered and number else "•"
        draw.text((style.margin_left, y), marker, font=self.body_font, fill=style.text_color)
        bbox = draw.textbbox((style.margin_left, y), marker + " ", font=self.body_font)
        draw.text((bbox[2], y), text, font=self.body_font, fill=style.text_color)
        return y + int(style.body_font_size * style.line_spacing)

    def _draw_checkbox_item(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        y: int,
        style: MarkdownStyle,
        checked: bool = False,
    ) -> int:
        """Draw checkbox list item."""
        box_size = style.body_font_size - 2
        box_x = style.margin_left
        box_y = y + 2

        # Draw checkbox
        draw.rectangle([box_x, box_y, box_x + box_size, box_y + box_size], outline=style.text_color)
        if checked:
            draw.line([(box_x + 2, box_y + box_size // 2), (box_x + box_size // 2, box_y + box_size - 2)],
                      fill=style.text_color, width=2)
            draw.line([(box_x + box_size // 2, box_y + box_size - 2), (box_x + box_size - 2, box_y + 2)],
                      fill=style.text_color, width=2)

        draw.text((box_x + box_size + 8, y), text.strip(), font=self.body_font, fill=style.text_color)
        return y + int(style.body_font_size * style.line_spacing)

    def _draw_table_row(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw table row."""
        if text.replace("|", "").replace("-", "").strip() == "":
            # Separator row
            draw.line(
                [(style.margin_left, y + 5), (style.margin_left + style.content_width, y + 5)],
                fill=style.text_color,
                width=1,
            )
            return y + 10

        cells = [cell.strip() for cell in text.split("|") if cell.strip()]
        if not cells:
            return y + int(style.body_font_size * style.line_spacing)

        cell_width = style.content_width // max(len(cells), 1)
        for i, cell in enumerate(cells):
            x = style.margin_left + i * cell_width
            draw.text((x, y), cell, font=self.body_font, fill=style.text_color)

        return y + int(style.body_font_size * style.line_spacing) + 2

    def _draw_horizontal_rule(self, draw: ImageDraw.ImageDraw, y: int, style: MarkdownStyle) -> int:
        """Draw horizontal rule."""
        draw.line(
            [(style.margin_left, y + 10), (style.margin_left + style.content_width, y + 10)],
            fill=(200, 200, 200),
            width=1,
        )
        return y + 20

    def _draw_italic(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw italic text (simulated)."""
        draw.text((style.margin_left, y), text, font=self.body_font, fill=(100, 100, 100))
        return y + int(style.body_font_size * style.line_spacing)

    def _apply_effects(self, img: Image.Image, style: MarkdownStyle) -> Image.Image:
        """Apply noise and other effects."""
        if style.add_noise:
            img = self._add_noise(img)

        if style.add_blur:
            blur_radius = random.uniform(0.3, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if style.add_contrast:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.9, 1.1)
            img = enhancer.enhance(factor)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise to image."""
        width, height = img.size
        noise = Image.new("RGB", (width, height))
        noise_draw = ImageDraw.Draw(noise)

        for _ in range(300):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            gray = random.randint(0, 255)
            noise_draw.point((x, y), fill=(gray, gray, gray))

        return Image.blend(img, noise, 0.03)


class MarkdownGenerator(BaseGenerator):
    """Main generator class for markdown image generation."""

    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str = "ko",
    ):
        super().__init__(output_dir, font_dir, lang)
        self.data_generator = MarkdownDataGenerator(lang)

    def generate(
        self,
        num_images: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate markdown images."""
        self.template = kwargs.get("template")
        self.add_noise = kwargs.get("add_noise", True)
        self.add_blur = kwargs.get("add_blur", False)

        if self.template:
            try:
                md_template = MarkdownTemplate(self.template)
            except ValueError:
                logger.warning(f"Unknown template '{self.template}', using random templates")
                md_template = None
        else:
            md_template = None

        self.templates = (
            [MarkdownTemplate(self.template)] if md_template
            else list(MarkdownTemplate)
        )

        metadata = []
        for idx in tqdm(range(num_images), desc="Generating markdown images"):
            image, meta = self.generate_single()

            # Save image
            filename = f"markdown_{idx:05d}.png"
            self.save_image(image, filename)
            meta["file_name"] = str(self.output_dir / filename)

            metadata.append(meta)

        return metadata

    def generate_single(self, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        if not hasattr(self, "templates"):
             template = kwargs.get("template")
             if template:
                try:
                    md_template = MarkdownTemplate(template)
                except ValueError:
                    md_template = None
             else:
                md_template = None
             
             self.templates = ([MarkdownTemplate(template)] if md_template else list(MarkdownTemplate))
             self.add_noise = kwargs.get("add_noise", True)
             self.add_blur = kwargs.get("add_blur", False)

        selected_template = random.choice(self.templates)

        # Generate markdown content
        markdown_text = self.data_generator.generate_markdown(template=selected_template)

        # Create style with random variations
        style = self._random_style()
        style.add_noise = self.add_noise
        style.add_blur = self.add_blur

        # Render markdown
        font_path = random.choice(self.font_paths)
        renderer = MarkdownRenderer(font_path, style)
        image = renderer.render(markdown_text)

        metadata = {
            "template": selected_template.value,
            "markdown": markdown_text,
            "add_noise": self.add_noise,
            "add_blur": self.add_blur,
        }
        return image, metadata

    def _random_style(self) -> MarkdownStyle:
        """Generate random style variations."""
        styles = [
            MarkdownStyle(
                background_color=(255, 255, 255),
                h1_color=(0, 0, 0),
                add_noise=True,
            ),
            MarkdownStyle(
                background_color=(250, 250, 245),
                h1_color=(51, 51, 51),
                add_noise=True,
                add_blur=True,
            ),
            MarkdownStyle(
                background_color=(255, 253, 250),
                h1_color=(30, 30, 30),
                add_noise=True,
                add_contrast=True,
            ),
            MarkdownStyle(
                background_color=(248, 249, 250),
                h1_color=(36, 41, 46),
                link_color=(3, 102, 214),
                add_noise=False,
            ),
        ]
        return random.choice(styles)
