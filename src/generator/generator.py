"""Markdown Generator Module for synthetic OCR markdown image generation.

This module provides comprehensive markdown document generation capabilities including:
- Various markdown elements (headers, paragraphs, lists, code blocks, tables, blockquotes)
- Multiple markdown templates (readme, technical_doc, blog_post, api_doc, tutorial)
- Layout variations (backgrounds, noise effects)
- Ground truth format with raw markdown text and rendered image
"""

import random
import logging
import tempfile
import importlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from character_similarity import find_similar_chars
from generator.base import BaseGenerator
from generator.data_provider import DataProvider
from utils import markdown_to_json_ast, read_json

logger = logging.getLogger(__name__)


class MarkdownTemplate(Enum):
    """Markdown template types."""
    README = "readme"
    TECHNICAL_DOC = "technical_doc"
    BLOG_POST = "blog_post"
    API_DOC = "api_doc"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"
    MEETING_NOTES = "meeting_notes"
    INCIDENT_REPORT = "incident_report"
    RELEASE_NOTE = "release_note"
    COMPLIANCE_CHECKLIST = "compliance_checklist"


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

    def __init__(self, lang: str = "ko", data_provider: Optional[DataProvider] = None):
        self.lang = lang
        self.data = data_provider or DataProvider(lang=lang)

    def generate_markdown(
        self,
        template: MarkdownTemplate = MarkdownTemplate.README,
    ) -> str:
        gen_func = getattr(self, f"_generate_{template.value}")
        return gen_func()

    @staticmethod
    def _slugify(text: str, max_parts: int = 3) -> str:
        chunks: List[str] = []
        for token in text.lower().replace("_", "-").split():
            cleaned = "".join(ch for ch in token if ch.isalnum() or ch == "-").strip("-")
            if cleaned:
                chunks.append(cleaned)
            if len(chunks) >= max_parts:
                break
        return "-".join(chunks) if chunks else "sample-app"

    def _project_slug(self) -> str:
        title = self.data.title()
        return self._slugify(title)

    def _sample_requirements(self, count: int = 3) -> List[str]:
        items = set()
        while len(items) < count:
            items.add(self.data.requirement_line())
        return list(items)

    def _sample_config_lines(self, count: int = 3) -> List[str]:
        items = set()
        while len(items) < count:
            items.add(self.data.config_line())
        return list(items)

    @staticmethod
    def _to_config_entry(line: str) -> Tuple[str, str]:
        if ":" in line:
            key, value = line.split(":", 1)
            return key.strip(), value.strip()
        token = "".join(ch for ch in line.lower().replace(" ", "_") if ch.isalnum() or ch == "_")
        return token or "option", "true"

    @staticmethod
    def _to_runtime_version(requirement: str) -> Tuple[str, str]:
        if ">=" in requirement:
            name, version = requirement.split(">=", 1)
            return name.strip(), version.strip()
        if "==" in requirement:
            name, version = requirement.split("==", 1)
            return name.strip(), version.strip()
        words = requirement.split()
        if not words:
            return "Runtime", "1.0"
        if len(words) == 1:
            return words[0], "1.0"
        return words[0], words[-1]

    def _generate_readme(self) -> str:
        title = self.data.title()
        project_slug = self._slugify(title)
        install_command = self.data.install_command(package_name=project_slug)
        usage_command = self.data.usage_command(entrypoint="main.py")
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
            install_command,
            "```",
            "",
        ])

        # Add usage section
        lines.extend([
            "## " + ("사용법" if self.lang == "ko" else "Usage"),
            "",
            "```python",
            self.data.code_comment(),
            f"from {project_slug.replace('-', '_')} import Client",
            "",
            "client = Client()",
            "result = client.run()",
            "```",
            "",
            "```bash",
            usage_command,
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
        title = self.data.title()
        requirements = self._sample_requirements(random.randint(3, 5))
        cfg_lines = self._sample_config_lines(3)
        lines = [
            f"# {title}",
            "",
            "## " + ("개요" if self.lang == "ko" else "Overview"),
            "",
            self.data.paragraph(),
            "",
            "## " + ("요구사항" if self.lang == "ko" else "Requirements"),
            "",
        ]

        # Add requirements list
        for i, req in enumerate(requirements, 1):
            lines.append(f"{i}. {req}")
        lines.append("")

        # Add table
        lines.extend([
            "## " + ("지원 버전" if self.lang == "ko" else "Supported Versions"),
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
            "## " + ("설정" if self.lang == "ko" else "Configuration"),
            "",
            "```yaml",
            "config:",
            f"  {cfg_lines[0]}",
            f"  {cfg_lines[1]}",
            f"  {cfg_lines[2]}",
            "```",
            "",
        ])

        return "\n".join(lines)

    def _generate_blog_post(self) -> str:
        title = self.data.title()
        date = self.data.date()

        lines = [
            f"# {title}",
            "",
            "*" + ("작성일" if self.lang == "ko" else "Published") + f": {date}*",
            "",
            "---",
            "",
            self.data.paragraph(),
            "",
            "## " + ("주요 내용" if self.lang == "ko" else "Key Points"),
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
        endpoint_get = self.data.api_endpoint()
        endpoint_post = self.data.api_endpoint()
        if endpoint_post == endpoint_get:
            endpoint_post = endpoint_get.rstrip("s") + "s"

        user_one = self.data.name()
        user_two = self.data.name()
        user_email = self.data.email()
        page_name = "페이지 번호" if self.lang == "ko" else "Page number"
        limit_name = "페이지당 항목 수" if self.lang == "ko" else "Items per page"
        lines = [
            "# API " + ("레퍼런스" if self.lang == "ko" else "Reference"),
            "",
            "## " + ("엔드포인트" if self.lang == "ko" else "Endpoints"),
            "",
            f"### GET {endpoint_get}",
            "",
            ("사용자 목록을 조회합니다." if self.lang == "ko" else "Retrieve a list of users."),
            "",
            "**" + ("파라미터" if self.lang == "ko" else "Parameters") + ":**",
            "",
            "| " + ("이름" if self.lang == "ko" else "Name") + " | " + ("타입" if self.lang == "ko" else "Type") + " | " + ("설명" if self.lang == "ko" else "Description") + " |",
            "|------|------|-------------|",
            f"| page | int | {page_name} |",
            f"| limit | int | {limit_name} |",
            "",
            "**" + ("응답 예시" if self.lang == "ko" else "Example Response") + ":**",
            "",
            "```json",
            "{",
            '  "users": [',
            f'    {{"id": 1, "name": "{user_one}"}},',
            f'    {{"id": 2, "name": "{user_two}"}}',
            "  ],",
            f'  "total": {random.randint(20, 500)}',
            "}",
            "```",
            "",
            f"### POST {endpoint_post}",
            "",
            ("새 사용자를 생성합니다." if self.lang == "ko" else "Create a new user."),
            "",
            "**" + ("요청 본문" if self.lang == "ko" else "Request Body") + ":**",
            "",
            "```json",
            "{",
            f'  "name": "{self.data.name()}",',
            f'  "email": "{user_email}"',
            "}",
            "```",
            "",
        ]

        return "\n".join(lines)

    def _generate_tutorial(self) -> str:
        title = self.data.title()
        project_slug = self._slugify(title)
        install_command = self.data.install_command(package_name=project_slug)
        run_command = self.data.usage_command(entrypoint="main.py")
        requirements = self._sample_requirements(3)
        cfg_lines = self._sample_config_lines(2)
        cfg_1_key, cfg_1_value = self._to_config_entry(cfg_lines[0])
        cfg_2_key, cfg_2_value = self._to_config_entry(cfg_lines[1])
        lines = [
            "# " + ("튜토리얼" if self.lang == "ko" else "Tutorial") + f": {title}",
            "",
            self.data.paragraph(),
            "",
            "## " + ("시작하기 전에" if self.lang == "ko" else "Before You Begin"),
            "",
            ("다음 항목이 필요합니다:" if self.lang == "ko" else "You will need:"),
            "",
            f"- [ ] {requirements[0]}",
            f"- [ ] {requirements[1]}",
            f"- [ ] {requirements[2]}",
            "",
            "## " + ("1단계" if self.lang == "ko" else "Step 1") + ": " + ("설치" if self.lang == "ko" else "Installation"),
            "",
            ("먼저 패키지를 설치합니다:" if self.lang == "ko" else "First, install the package:"),
            "",
            "```bash",
            install_command,
            "```",
            "",
            "## " + ("2단계" if self.lang == "ko" else "Step 2") + ": " + ("설정" if self.lang == "ko" else "Configuration"),
            "",
            ("설정 파일을 생성합니다:" if self.lang == "ko" else "Create a configuration file:"),
            "",
            "```python",
            self.data.code_comment(),
            "config = {",
            f'    "{cfg_1_key}": "{cfg_1_value}",',
            f'    "{cfg_2_key}": "{cfg_2_value}"',
            "}",
            "```",
            "",
            "> **" + ("팁" if self.lang == "ko" else "Tip") + "**: " + self.data.paragraph()[:60],
            "",
            "## " + ("3단계" if self.lang == "ko" else "Step 3") + ": " + ("실행" if self.lang == "ko" else "Run"),
            "",
            "```bash",
            run_command,
            "```",
            "",
            ("예상 출력:" if self.lang == "ko" else "Expected output:"),
            "",
            "```",
            f"Success! Service started for {project_slug}",
            "```",
            "",
        ]

        return "\n".join(lines)

    def _generate_changelog(self) -> str:
        version = f"v{random.randint(1, 4)}.{random.randint(0, 9)}.{random.randint(0, 12)}"
        lines = [
            "# " + ("변경 이력" if self.lang == "ko" else "Changelog"),
            "",
            f"## {version} - {self.data.date()}",
            "",
            "### " + ("추가" if self.lang == "ko" else "Added"),
            "",
        ]
        for feature in self.data.features(3):
            lines.append(f"- {feature}")
        lines.extend([
            "",
            "### " + ("수정" if self.lang == "ko" else "Fixed"),
            "",
            f"- {self.data.paragraph()[:70]}",
            f"- {self.data.paragraph()[:70]}",
            "",
            "### " + ("변경" if self.lang == "ko" else "Changed"),
            "",
            "| " + ("항목" if self.lang == "ko" else "Item") + " | " + ("영향도" if self.lang == "ko" else "Impact") + " |",
            "|---|---|",
            "| API | High |",
            "| UI | Medium |",
            "| Docs | Low |",
            "",
        ])
        return "\n".join(lines)

    def _generate_meeting_notes(self) -> str:
        lines = [
            "# " + ("회의록" if self.lang == "ko" else "Meeting Notes"),
            "",
            "- " + ("일시" if self.lang == "ko" else "Date") + f": {self.data.date()}",
            "- " + ("참석자" if self.lang == "ko" else "Attendees") + f": {self.data.name()}, {self.data.name()}, {self.data.name()}",
            "",
            "## " + ("안건" if self.lang == "ko" else "Agenda"),
            "",
            "1. " + self.data.title(),
            "2. " + self.data.title(),
            "3. " + self.data.title(),
            "",
            "## " + ("결정사항" if self.lang == "ko" else "Decisions"),
            "",
            "- [x] " + self.data.features(1)[0],
            "- [x] " + self.data.features(1)[0],
            "- [ ] " + self.data.features(1)[0],
            "",
            "## " + ("액션 아이템" if self.lang == "ko" else "Action Items"),
            "",
            "| " + ("담당" if self.lang == "ko" else "Owner") + " | " + ("작업" if self.lang == "ko" else "Task") + " | " + ("기한" if self.lang == "ko" else "Due") + " |",
            "|---|---|---|",
            f"| {self.data.name()} | {self.data.paragraph()[:24]} | {self.data.date()} |",
            f"| {self.data.name()} | {self.data.paragraph()[:24]} | {self.data.date()} |",
            "",
        ]
        return "\n".join(lines)

    def _generate_incident_report(self) -> str:
        severity = random.choice(["SEV-1", "SEV-2", "SEV-3"])
        lines = [
            "# " + ("장애 보고서" if self.lang == "ko" else "Incident Report"),
            "",
            f"- Incident ID: INC-{random.randint(1000, 9999)}",
            f"- Severity: {severity}",
            "- " + ("발생 시각" if self.lang == "ko" else "Start Time") + f": {self.data.date()}",
            "",
            "## " + ("요약" if self.lang == "ko" else "Summary"),
            "",
            self.data.paragraph(),
            "",
            "## " + ("타임라인" if self.lang == "ko" else "Timeline"),
            "",
            f"- 09:10 - {self.data.paragraph()[:48]}",
            f"- 09:25 - {self.data.paragraph()[:48]}",
            f"- 09:41 - {self.data.paragraph()[:48]}",
            "",
            "## " + ("영향 범위" if self.lang == "ko" else "Impact"),
            "",
            "```json",
            "{",
            f"  \"affected_users\": {random.randint(100, 5000)},",
            f"  \"region\": \"{random.choice(['ap-northeast-2', 'us-east-1', 'eu-west-1'])}\",",
            f"  \"duration_min\": {random.randint(10, 180)}",
            "}",
            "```",
            "",
            "## " + ("재발 방지" if self.lang == "ko" else "Preventive Actions"),
            "",
            "- [ ] " + self.data.features(1)[0],
            "- [ ] " + self.data.features(1)[0],
            "",
        ]
        return "\n".join(lines)

    def _generate_release_note(self) -> str:
        release = f"{random.randint(2024, 2027)}.{random.randint(1, 12)}.{random.randint(1, 28)}"
        requirements = self._sample_requirements(3)
        runtime_1, min_1 = self._to_runtime_version(requirements[0])
        runtime_2, min_2 = self._to_runtime_version(requirements[1])
        runtime_3, min_3 = self._to_runtime_version(requirements[2])
        install_cmd = self.data.install_command(package_name="synthetic-ocr")
        run_cmd = self.data.usage_command(entrypoint="main.py")
        lines = [
            "# " + ("릴리즈 노트" if self.lang == "ko" else "Release Notes") + f" {release}",
            "",
            "## " + ("하이라이트" if self.lang == "ko" else "Highlights"),
            "",
            f"- **{self.data.features(1)[0]}**",
            f"- **{self.data.features(1)[0]}**",
            f"- **{self.data.features(1)[0]}**",
            "",
            "## " + ("호환성" if self.lang == "ko" else "Compatibility"),
            "",
            "| Runtime | Minimum | Recommended |",
            "|---|---|---|",
            f"| {runtime_1} | {min_1} | {min_1}+ |",
            f"| {runtime_2} | {min_2} | {min_2}+ |",
            f"| {runtime_3} | {min_3} | {min_3}+ |",
            "",
            "## " + ("업그레이드 가이드" if self.lang == "ko" else "Upgrade Guide"),
            "",
            "```bash",
            install_cmd.replace(" install ", " install -U "),
            f"{run_cmd} generate --lang en --size 10" if run_cmd.startswith("python") else run_cmd,
            "```",
            "",
        ]
        return "\n".join(lines)

    def _generate_compliance_checklist(self) -> str:
        lines = [
            "# " + ("컴플라이언스 체크리스트" if self.lang == "ko" else "Compliance Checklist"),
            "",
            "## " + ("데이터 보안" if self.lang == "ko" else "Data Security"),
            "",
            "- [x] Encryption at rest",
            "- [x] Encryption in transit",
            "- [ ] Data retention policy review",
            "",
            "## " + ("접근 통제" if self.lang == "ko" else "Access Control"),
            "",
            "1. MFA enabled for admins",
            "2. Role-based access matrix updated",
            "3. Quarterly permission audit completed",
            "",
            "## " + ("감사 로그" if self.lang == "ko" else "Audit Logs"),
            "",
            "```yaml",
            "audit:",
            "  enabled: true",
            "  retention_days: 365",
            "  export: s3://compliance-logs",
            "```",
            "",
            "> " + ("주의" if self.lang == "ko" else "Note") + ": " + self.data.paragraph(),
            "",
        ]
        return "\n".join(lines)


class MarkdownRenderer:
    """Renders markdown content to images."""

    _FONT_CACHE: Dict[Tuple[str, int], Any] = {}

    def __init__(self, font_path: str, style: Optional[MarkdownStyle] = None):
        self.style = style or MarkdownStyle()
        self.font_path = font_path

        try:
            self.body_font = self._get_font(font_path, self.style.body_font_size)
            self.h1_font = self._get_font(font_path, self.style.h1_font_size)
            self.h2_font = self._get_font(font_path, self.style.h2_font_size)
            self.h3_font = self._get_font(font_path, self.style.h3_font_size)
            self.code_font = self._get_font(font_path, self.style.code_font_size)
        except IOError:
            logger.warning("Font '%s' not found. Using default.", font_path)
            self.body_font = ImageFont.load_default()
            self.h1_font = self.body_font
            self.h2_font = self.body_font
            self.h3_font = self.body_font
            self.code_font = self.body_font

    @classmethod
    def _get_font(cls, font_path: str, size: int) -> ImageFont.ImageFont:
        key = (font_path, size)
        if key not in cls._FONT_CACHE:
            cls._FONT_CACHE[key] = ImageFont.truetype(font_path, size)
        return cls._FONT_CACHE[key]

    @staticmethod
    def _is_ordered_list_item(stripped: str) -> bool:
        return bool(stripped) and stripped[0].isdigit() and ". " in stripped

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
            elif stripped.startswith("- [ ]") or stripped.startswith("- [x]"):
                checked = stripped.startswith("- [x]")
                current_y = self._draw_checkbox_item(draw, stripped[6:], current_y, style, checked)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                current_y = self._draw_list_item(draw, stripped[2:], current_y, style, ordered=False)
            elif self._is_ordered_list_item(stripped):
                idx = stripped.index(". ")
                current_y = self._draw_list_item(
                    draw,
                    stripped[idx + 2 :],
                    current_y,
                    style,
                    ordered=True,
                    number=stripped[:idx],
                )
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
        if stripped.startswith("## "):
            return int(self.style.h2_font_size * self.style.line_spacing) + 8
        if stripped.startswith("### "):
            return int(self.style.h3_font_size * self.style.line_spacing) + 6
        if stripped.startswith("```"):
            return 5
        if stripped.startswith("> "):
            return base_spacing + 10
        if stripped == "---" or stripped == "***":
            return 20
        if stripped:
            return base_spacing
        return int(self.style.body_font_size * 0.5)

    def _draw_h1(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw H1 header."""
        draw.text((style.margin_left, y), text, font=self.h1_font, fill=style.h1_color)
        # Draw underline
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h1_font)
        line_y = int(bbox[3]) + 5
        draw.line([(style.margin_left, line_y), (style.margin_left + style.content_width, line_y)],
                  fill=style.h2_color, width=2)
        return int(line_y + 15)

    def _draw_h2(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw H2 header."""
        draw.text((style.margin_left, y), text, font=self.h2_font, fill=style.h2_color)
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h2_font)
        return int(bbox[3] + 12)

    def _draw_h3(self, draw: ImageDraw.ImageDraw, text: str, y: int, style: MarkdownStyle) -> int:
        """Draw H3 header."""
        draw.text((style.margin_left, y), text, font=self.h3_font, fill=style.h3_color)
        bbox = draw.textbbox((style.margin_left, y), text, font=self.h3_font)
        return int(bbox[3] + 10)

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
        number: Optional[str] = None,
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
        noise = np.zeros((height, width, 3), dtype=np.uint8)
        sample_count = 300
        xs = np.random.randint(0, width, size=sample_count)
        ys = np.random.randint(0, height, size=sample_count)
        grays = np.random.randint(0, 256, size=sample_count, dtype=np.uint8)
        noise[ys, xs] = np.stack([grays, grays, grays], axis=1)
        noise_img = Image.fromarray(noise, mode="RGB")

        return Image.blend(img, noise_img, 0.03)


class HtmlMarkdownRenderer:
    """Renders markdown through HTML and captures it as an image."""

    def __init__(self, font_path: str, style: Optional[MarkdownStyle] = None):
        self.style = style or MarkdownStyle()
        self.font_path = str(Path(font_path).resolve())

    @staticmethod
    def _coerce_markdown_html(markdown_text: str) -> str:
        try:
            markdown_pkg = importlib.import_module("markdown")
        except ImportError as exc:
            raise RuntimeError(
                "markdown package is required for markdown->html rendering. "
                "Install with: uv sync --group generate"
            ) from exc

        return markdown_pkg.markdown(
            markdown_text,
            extensions=["extra", "tables", "fenced_code", "sane_lists"],
        )

    def _estimate_viewport_height(self, markdown_text: str) -> int:
        lines = markdown_text.splitlines() or [""]
        body_line_px = int(self.style.body_font_size * self.style.line_spacing)
        chars_per_line = max(18, self.style.content_width // max(self.style.body_font_size - 1, 8))

        wrapped_line_count = 0
        header_bonus = 0
        code_bonus = 0
        table_bonus = 0
        for raw in lines:
            line = raw.strip()
            wrapped_line_count += max(1, (len(raw) // chars_per_line) + 1)
            if line.startswith("# "):
                header_bonus += self.style.h1_font_size
            elif line.startswith("## "):
                header_bonus += self.style.h2_font_size
            elif line.startswith("### "):
                header_bonus += self.style.h3_font_size
            if line.startswith("```"):
                code_bonus += int(self.style.code_font_size * self.style.line_spacing * 2)
            if line.startswith("|"):
                table_bonus += int(body_line_px * 0.6)

        estimated = (
            self.style.margin_top
            + self.style.margin_bottom
            + wrapped_line_count * body_line_px
            + header_bonus
            + code_bonus
            + table_bonus
            + 120
        )
        return max(300, min(9000, int(estimated)))

    def _build_html_document(self, markdown_text: str) -> str:
        rendered_html = self._coerce_markdown_html(markdown_text)
        css = f"""
@font-face {{
  font-family: 'RenderFont';
  src: url('file://{escape(self.font_path)}') format('truetype');
}}
html, body {{
  margin: 0;
  padding: 0;
  background: rgb{self.style.background_color};
}}
body {{
  width: {self.style.margin_left + self.style.content_width + self.style.margin_right}px;
}}
.markdown-body {{
  width: {self.style.content_width}px;
  padding: {self.style.margin_top}px {self.style.margin_right}px {self.style.margin_bottom}px {self.style.margin_left}px;
  color: rgb{self.style.text_color};
  font-family: 'RenderFont', sans-serif;
  font-size: {self.style.body_font_size}px;
  line-height: {self.style.line_spacing};
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}}
.markdown-body h1 {{ font-size: {self.style.h1_font_size}px; color: rgb{self.style.h1_color}; margin: 0 0 16px 0; }}
.markdown-body h2 {{ font-size: {self.style.h2_font_size}px; color: rgb{self.style.h2_color}; margin: 18px 0 12px 0; }}
.markdown-body h3 {{ font-size: {self.style.h3_font_size}px; color: rgb{self.style.h3_color}; margin: 16px 0 8px 0; }}
.markdown-body a {{ color: rgb{self.style.link_color}; text-decoration: none; }}
.markdown-body p {{ margin: 0 0 10px 0; }}
.markdown-body ul, .markdown-body ol {{ margin: 0 0 12px 18px; padding: 0; }}
.markdown-body blockquote {{
  margin: 0 0 12px 0;
  padding: 0 0 0 12px;
  border-left: 3px solid rgb{self.style.blockquote_border_color};
  color: rgb{self.style.blockquote_color};
}}
.markdown-body pre, .markdown-body code {{
  font-family: 'RenderFont', monospace;
  font-size: {self.style.code_font_size}px;
}}
.markdown-body pre {{
  margin: 0 0 12px 0;
  padding: 8px 10px;
  background: rgb{self.style.code_bg_color};
  color: rgb{self.style.code_text_color};
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}}
.markdown-body code {{
  background: rgb{self.style.code_bg_color};
  color: rgb{self.style.code_text_color};
  padding: 1px 3px;
}}
.markdown-body table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 12px 0;
  table-layout: fixed;
}}
.markdown-body th, .markdown-body td {{
  border: 1px solid rgba(0, 0, 0, 0.25);
  text-align: left;
  padding: 6px;
  overflow-wrap: anywhere;
}}
"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>{css}</style>
</head>
<body>
  <div class="markdown-body">{rendered_html}</div>
</body>
</html>"""

    def _trim_bottom_whitespace(self, image: Image.Image) -> Image.Image:
        background = Image.new("RGB", image.size, self.style.background_color)
        diff = ImageChops.difference(image, background)
        bbox = diff.getbbox()
        if not bbox:
            return image
        cropped_bottom = min(image.height, int(bbox[3] + self.style.margin_bottom))
        return image.crop((0, 0, image.width, max(cropped_bottom, 200)))

    def _apply_effects(self, img: Image.Image) -> Image.Image:
        if self.style.add_noise:
            img = self._add_noise(img)

        if self.style.add_blur:
            blur_radius = random.uniform(0.3, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if self.style.add_contrast:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.9, 1.1)
            img = enhancer.enhance(factor)

        return img

    @staticmethod
    def _add_noise(img: Image.Image) -> Image.Image:
        width, height = img.size
        noise = np.zeros((height, width, 3), dtype=np.uint8)
        sample_count = 300
        xs = np.random.randint(0, width, size=sample_count)
        ys = np.random.randint(0, height, size=sample_count)
        grays = np.random.randint(0, 256, size=sample_count, dtype=np.uint8)
        noise[ys, xs] = np.stack([grays, grays, grays], axis=1)
        noise_img = Image.fromarray(noise, mode="RGB")
        return Image.blend(img, noise_img, 0.03)

    def render(self, markdown_text: str) -> Image.Image:
        try:
            Html2Image = importlib.import_module("html2image").Html2Image
        except ImportError as exc:
            raise RuntimeError(
                "html2image package is required for html->image rendering. "
                "Install with: uv sync --group generate"
            ) from exc

        width = self.style.margin_left + self.style.content_width + self.style.margin_right
        height = self._estimate_viewport_height(markdown_text)
        html_doc = self._build_html_document(markdown_text)

        with tempfile.TemporaryDirectory(prefix="markdown-html2image-") as temp_dir:
            hti = Html2Image(
                output_path=temp_dir,
                size=(width, height),
                custom_flags=[
                    "--headless=new",
                    "--hide-scrollbars",
                    "--disable-gpu",
                    "--force-device-scale-factor=1",
                ],
            )
            out_name = "rendered.png"
            hti.screenshot(html_str=html_doc, save_as=out_name)
            rendered_path = Path(temp_dir) / out_name
            image = Image.open(rendered_path).convert("RGB")
            image.load()

        image = self._trim_bottom_whitespace(image)
        return self._apply_effects(image)


class Generator(BaseGenerator):
    """Main generator class for markdown image generation."""

    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str = "ko",
    ):
        super().__init__(output_dir, font_dir, lang)
        self.data_generator = MarkdownDataGenerator(lang)
        self.similarity_db: Dict[str, Any] = {}
        self.similarity_db_path = ""
        self._similarity_db_source: Optional[str] = None
        self._protected_chars = set("#`|[](){}<>!+-=_~*/\\")
        self.templates: List[MarkdownTemplate] = list(MarkdownTemplate)
        self.add_noise = True
        self.add_blur = False
        self.noise_ratio = 0.1
        self.blur_ratio = 0.1
        self.similar_char_ratio = 0.08
        self.markdown_renderer = "pil"

    def _load_similarity_db(self, db_path: Optional[str]) -> None:
        source_key = db_path or "__auto__"
        if self._similarity_db_source == source_key:
            return

        if db_path:
            candidates = [Path(db_path)]
        else:
            candidates = [
                Path("data") / self.lang / f"char_similarity_db_{self.lang}.json",
                Path("data") / f"char_similarity_db_{self.lang}.json",
                Path("data") / self.lang / "char_similarity_db.json",
                Path("data") / "char_similarity_db.json",
            ]

        self._similarity_db_source = source_key
        resolved = next((p for p in candidates if p.exists()), None)
        if resolved is None:
            self.similarity_db = {}
            self.similarity_db_path = ""
            return

        loaded = read_json(str(resolved))
        if isinstance(loaded, dict):
            self.similarity_db = loaded
            self.similarity_db_path = str(resolved)
            return

        self.similarity_db = {}
        self.similarity_db_path = ""

    def _resolve_templates(self, template: Optional[str]) -> List[MarkdownTemplate]:
        if not template:
            return list(MarkdownTemplate)

        try:
            return [MarkdownTemplate(template)]
        except ValueError:
            logger.warning("Unknown template '%s', using random templates", template)
            return list(MarkdownTemplate)

    def _configure_generation(self, **kwargs) -> None:
        def _coerce_ratio(value: Any, default: float) -> float:
            if value is None:
                return default
            try:
                ratio = float(value)
            except (TypeError, ValueError):
                return default
            return max(0.0, min(1.0, ratio))

        def _coerce_bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "y", "on"}:
                    return True
                if lowered in {"0", "false", "no", "n", "off"}:
                    return False
            return bool(value)

        template = kwargs.get("template")
        self.templates = self._resolve_templates(template)
        self.add_noise = _coerce_bool(kwargs.get("add_noise"), True)
        self.add_blur = _coerce_bool(kwargs.get("add_blur"), False)
        self.noise_ratio = _coerce_ratio(kwargs.get("noise_ratio"), 0.1)
        self.blur_ratio = _coerce_ratio(kwargs.get("blur_ratio"), 0.1)
        if "add_noise" in kwargs and kwargs.get("add_noise") is not None:
            self.noise_ratio = 1.0 if self.add_noise else 0.0
        if "add_blur" in kwargs and kwargs.get("add_blur") is not None:
            self.blur_ratio = 1.0 if self.add_blur else 0.0
        self.similar_char_ratio = float(kwargs.get("similar_char_ratio", 0.08))
        requested_renderer = str(kwargs.get("markdown_renderer", self.markdown_renderer)).strip().lower()
        if requested_renderer not in {"pil", "html2image"}:
            logger.warning(
                "Unknown markdown renderer '%s'. Falling back to 'pil'.",
                requested_renderer,
            )
            requested_renderer = "pil"
        self.markdown_renderer = requested_renderer
        self._load_similarity_db(kwargs.get("similarity_db_path"))

    def _mutate_similar_text(self, text: str, ratio: float) -> Tuple[str, int]:
        if ratio <= 0 or not self.similarity_db:
            return text, 0

        chars = list(text)
        candidate_indices: List[int] = []
        cached_candidates: Dict[str, List[Tuple[str, float]]] = {}

        def get_candidates(ch: str) -> List[Tuple[str, float]]:
            if ch not in cached_candidates:
                cached_candidates[ch] = find_similar_chars(ch, self.similarity_db, top_n=5)
            return cached_candidates[ch]

        for idx, ch in enumerate(chars):
            if ch in self._protected_chars or ch.isspace():
                continue
            if get_candidates(ch):
                candidate_indices.append(idx)

        if not candidate_indices:
            return text, 0

        target = int(len(candidate_indices) * ratio)
        if target == 0:
            target = 1
        target = min(target, len(candidate_indices))

        mutated_count = 0
        for idx in random.sample(candidate_indices, target):
            source = chars[idx]
            candidates = get_candidates(source)
            if not candidates:
                continue
            replacement, _ = random.choice(candidates)
            if (
                not replacement
                or any(c in self._protected_chars or c.isspace() for c in replacement)
            ):
                continue
            if replacement == source:
                continue
            chars[idx] = replacement
            mutated_count += 1

        return "".join(chars), mutated_count

    def generate(
        self,
        num_images: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate markdown images."""
        self._configure_generation(**kwargs)

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
        if kwargs:
            self._configure_generation(**kwargs)

        selected_template = random.choice(self.templates)

        # Generate markdown content
        original_markdown = self.data_generator.generate_markdown(template=selected_template)
        markdown_text, mutation_count = self._mutate_similar_text(
            original_markdown,
            self.similar_char_ratio,
        )

        # Create style with random variations
        style = self._random_style()
        style.add_noise = random.random() < self.noise_ratio
        style.add_blur = random.random() < self.blur_ratio

        # Render markdown
        font_path = random.choice(self.font_paths)
        if self.markdown_renderer == "html2image":
            renderer = HtmlMarkdownRenderer(font_path, style)
        else:
            renderer = MarkdownRenderer(font_path, style)
        image = renderer.render(markdown_text)

        metadata = {
            "template": selected_template.value,
            "GT_markdown": markdown_text,
            "GT_json": markdown_to_json_ast(markdown_text),
            "similar_char_mutations": mutation_count,
            "renderer": self.markdown_renderer,
        }
        return image, metadata

    def _random_style(self) -> MarkdownStyle:
        """Generate random style variations."""
        styles = [
            MarkdownStyle(
                background_color=(255, 255, 255),
                h1_color=(0, 0, 0),
                add_noise=True,
                margin_left=34,
                margin_right=34,
                content_width=620,
            ),
            MarkdownStyle(
                background_color=(250, 250, 245),
                h1_color=(51, 51, 51),
                add_noise=True,
                add_blur=True,
                margin_left=48,
                margin_right=48,
                content_width=560,
                line_spacing=1.45,
            ),
            MarkdownStyle(
                background_color=(255, 253, 250),
                h1_color=(30, 30, 30),
                add_noise=True,
                add_contrast=True,
                margin_left=56,
                margin_right=56,
                content_width=540,
                h1_font_size=30,
            ),
            MarkdownStyle(
                background_color=(248, 249, 250),
                h1_color=(36, 41, 46),
                link_color=(3, 102, 214),
                add_noise=False,
                margin_left=40,
                margin_right=40,
                content_width=640,
                body_font_size=13,
                code_font_size=11,
            ),
            MarkdownStyle(
                background_color=(244, 240, 232),
                h1_color=(44, 38, 31),
                h2_color=(70, 64, 58),
                text_color=(42, 42, 42),
                add_noise=True,
                add_blur=False,
                add_contrast=True,
                margin_left=60,
                margin_right=52,
                content_width=520,
                line_spacing=1.58,
            ),
            MarkdownStyle(
                background_color=(236, 242, 246),
                h1_color=(12, 42, 68),
                h2_color=(29, 72, 102),
                link_color=(12, 96, 158),
                text_color=(25, 36, 46),
                code_bg_color=(222, 232, 240),
                add_noise=False,
                add_blur=True,
                margin_left=44,
                margin_right=44,
                content_width=600,
                line_spacing=1.4,
            ),
        ]
        selected = random.choice(styles)
        selected.margin_top += random.randint(-8, 14)
        selected.margin_bottom += random.randint(-8, 14)
        selected.content_width += random.randint(-24, 24)
        selected.line_spacing = max(1.3, min(1.7, selected.line_spacing + random.uniform(-0.08, 0.1)))
        return selected
