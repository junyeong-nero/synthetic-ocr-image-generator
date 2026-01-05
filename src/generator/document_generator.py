"""Document Generator Module for synthetic OCR document image generation.

This module provides comprehensive document generation capabilities including:
- Multi-section document layout (header, body, footer)
- Multiple document templates (invoice, receipt, form, letter, report)
- Mixed content support (text, tables, lists)
- Layout variations (single/multi-column, backgrounds, noise)
- Ground truth format with bounding boxes and reading order
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

logger = logging.getLogger(__name__)


class DocumentTemplate(Enum):
    """Document template types."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    FORM = "form"
    LETTER = "letter"
    REPORT = "report"


class LayoutStyle(Enum):
    """Document layout styles."""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    MULTI_COLUMN = "multi_column"


class TextAlignment(Enum):
    """Text alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"


@dataclass
class DocumentElement:
    """Base class for document elements."""
    element_type: str
    text: str = ""
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    reading_order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TitleElement(DocumentElement):
    """Title element with optional subtitle."""
    def __init__(self, title: str, subtitle: str = "", level: int = 1):
        super().__init__(element_type="title", text=title)
        self.subtitle = subtitle
        self.level = level


@dataclass
class ParagraphElement(DocumentElement):
    """Paragraph element with text content."""
    def __init__(self, text: str, alignment: TextAlignment = TextAlignment.LEFT):
        super().__init__(element_type="paragraph", text=text)
        self.alignment = alignment


@dataclass
class TableElement(DocumentElement):
    """Table element reference (table data stored in metadata)."""
    def __init__(self, table_data: List[List[str]], headers: List[str] = None):
        super().__init__(element_type="table", text="table")
        self.table_data = table_data
        self.headers = headers


@dataclass
class ListElement(DocumentElement):
    """List element (ordered or unordered)."""
    def __init__(self, items: List[str], ordered: bool = False):
        super().__init__(element_type="list", text="\n".join(items))
        self.items = items
        self.ordered = ordered


@dataclass
class DocumentBlock:
    """A block of content in the document (header, body, footer)."""
    block_type: str
    elements: List[DocumentElement]
    bounding_box: Optional[Tuple[int, int, int, int]] = None


@dataclass
class DocumentStyle:
    """Document styling options."""
    # Layout
    layout_style: LayoutStyle = LayoutStyle.SINGLE_COLUMN
    margin_top: int = 50
    margin_bottom: int = 50
    margin_left: int = 50
    margin_right: int = 50
    content_width: int = 500

    # Typography
    title_font_size: int = 24
    heading_font_size: int = 18
    body_font_size: int = 12
    text_color: Tuple[int, int, int] = (0, 0, 0)
    title_color: Tuple[int, int, int] = (0, 0, 0)

    # Background
    background_color: Tuple[int, int, int] = (255, 255, 255)
    line_color: Tuple[int, int, int] = (200, 200, 200)

    # Effects
    add_noise: bool = True
    add_blur: bool = False
    add_contrast: bool = False
    add_watermark: bool = False


@dataclass
class Document:
    """Complete document with metadata."""
    template: DocumentTemplate
    blocks: List[DocumentBlock]
    style: DocumentStyle = field(default_factory=DocumentStyle)
    page_number: Optional[str] = None
    date_stamp: Optional[str] = None

    def to_ground_truth(self) -> Dict[str, Any]:
        """Convert document to ground truth format with bounding boxes."""
        elements = []
        reading_order = 0

        for block in self.blocks:
            for element in block.elements:
                element.reading_order = reading_order
                reading_order += 1
                elements.append({
                    "type": element.element_type,
                    "text": element.text,
                    "bounding_box": element.bounding_box,
                    "reading_order": element.reading_order,
                    "metadata": element.metadata,
                })

        return {
            "template": self.template.value,
            "elements": elements,
            "blocks": [
                {
                    "type": block.block_type,
                    "element_count": len(block.elements),
                    "bounding_box": block.bounding_box,
                }
                for block in self.blocks
            ],
            "page_number": self.page_number,
            "date_stamp": self.date_stamp,
        }


class DocumentRenderer:
    """Renders Document objects to images with bounding box annotations."""

    def __init__(self, font_path: str, font_size: int = 12):
        self.font_size = font_size
        try:
            self.font = ImageFont.truetype(font_path, font_size)
            self.title_font = ImageFont.truetype(font_path, font_size + 12)
            self.heading_font = ImageFont.truetype(font_path, font_size + 6)
        except IOError:
            logger.warning(f"Font '{font_path}' not found. Using default.")
            self.font = ImageFont.load_default()
            self.title_font = self.font
            self.heading_font = self.font

    def render(self, document: Document) -> Tuple[Image.Image, Dict[str, Any]]:
        """Render document to image with ground truth."""
        style = document.style

        # Calculate image dimensions
        width = style.margin_left + style.content_width + style.margin_right
        height = style.margin_top + style.margin_bottom + 800  # Approximate

        img = Image.new("RGB", (width, height), style.background_color)
        draw = ImageDraw.Draw(img)

        ground_truth = {
            "elements": [],
            "bounding_boxes": [],
        }

        current_y = style.margin_top
        reading_order = 0

        for block in document.blocks:
            block_elements = []
            block_start_y = current_y

            for element in block.elements:
                if isinstance(element, TitleElement):
                    current_y = self._draw_title(
                        draw, element, current_y, style, reading_order
                    )
                elif isinstance(element, ParagraphElement):
                    current_y = self._draw_paragraph(
                        draw, element, current_y, style, reading_order
                    )
                elif isinstance(element, TableElement):
                    current_y = self._draw_table_placeholder(
                        draw, element, current_y, style, reading_order
                    )
                elif isinstance(element, ListElement):
                    current_y = self._draw_list(
                        draw, element, current_y, style, reading_order
                    )

                if element.bounding_box:
                    ground_truth["elements"].append({
                        "type": element.element_type,
                        "text": element.text,
                        "bounding_box": element.bounding_box,
                        "reading_order": reading_order,
                    })
                    ground_truth["bounding_boxes"].append(element.bounding_box)

                reading_order += 1

            block.bounding_box = (
                style.margin_left,
                block_start_y,
                width - style.margin_right,
                current_y,
            )
            block_elements.append(block.bounding_box)

        # Draw header/footer
        if document.date_stamp:
            self._draw_date_stamp(draw, document.date_stamp, width, style)
        if document.page_number:
            self._draw_page_number(draw, document.page_number, width, height, style)

        # Draw margin lines
        self._draw_margin_lines(draw, width, height, style)

        # Apply effects
        img = self._apply_effects(img, style)

        return img, ground_truth

    def _draw_title(
        self,
        draw: ImageDraw.ImageDraw,
        element: TitleElement,
        current_y: int,
        style: DocumentStyle,
        reading_order: int,
    ) -> int:
        font = self.title_font if element.level == 1 else self.heading_font
        color = style.title_color if element.level == 1 else style.text_color

        bbox = draw.textbbox((style.margin_left, current_y), element.text, font=font)
        text_height = bbox[3] - bbox[1]

        draw.text(
            (style.margin_left, current_y),
            element.text,
            font=font,
            fill=color,
        )

        if element.subtitle:
            sub_bbox = draw.textbbox(
                (style.margin_left, current_y + text_height + 5),
                element.subtitle,
                font=self.heading_font,
            )
            sub_height = sub_bbox[3] - sub_bbox[1]
            draw.text(
                (style.margin_left, current_y + text_height + 5),
                element.subtitle,
                font=self.heading_font,
                fill=style.text_color,
            )
            text_height += sub_height + 5

        element.bounding_box = (
            style.margin_left,
            current_y,
            style.margin_left + (bbox[2] - bbox[0]),
            current_y + text_height,
        )

        return current_y + text_height + 20

    def _draw_paragraph(
        self,
        draw: ImageDraw.ImageDraw,
        element: ParagraphElement,
        current_y: int,
        style: DocumentStyle,
        reading_order: int,
    ) -> int:
        max_width = style.content_width
        words = element.text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=self.font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        line_height = self.font_size + 4
        text_height = len(lines) * line_height

        for i, line in enumerate(lines):
            x = style.margin_left
            if element.alignment == TextAlignment.CENTER:
                bbox = draw.textbbox((0, 0), line, font=self.font)
                line_width = bbox[2] - bbox[0]
                x = style.margin_left + (style.content_width - line_width) // 2
            elif element.alignment == TextAlignment.RIGHT:
                bbox = draw.textbbox((0, 0), line, font=self.font)
                line_width = bbox[2] - bbox[0]
                x = style.margin_left + style.content_width - line_width

            draw.text(
                (x, current_y + i * line_height),
                line,
                font=self.font,
                fill=style.text_color,
            )

        element.bounding_box = (
            style.margin_left,
            current_y,
            style.margin_left + style.content_width,
            current_y + text_height,
        )

        return current_y + text_height + 15

    def _draw_table_placeholder(
        self,
        draw: ImageDraw.ImageDraw,
        element: TableElement,
        current_y: int,
        style: DocumentStyle,
        reading_order: int,
    ) -> int:
        # Draw a simple table representation
        row_height = self.font_size + 10
        col_width = style.content_width // len(element.headers) if element.headers else 100

        # Draw header
        header_bg_color = (220, 220, 220)
        draw.rectangle(
            [
                style.margin_left,
                current_y,
                style.margin_left + style.content_width,
                current_y + row_height,
            ],
            fill=header_bg_color,
            outline=style.line_color,
        )

        if element.headers:
            for i, header in enumerate(element.headers):
                x = style.margin_left + i * col_width + 5
                draw.text((x, current_y + 3), header, font=self.font, fill=style.text_color)

        current_y += row_height

        # Draw data rows
        for row in element.table_data[:5]:  # Limit to 5 rows for display
            for i, cell in enumerate(row):
                x = style.margin_left + i * col_width + 5
                draw.text((x, current_y + 3), str(cell)[:15], font=self.font, fill=style.text_color)
            draw.line(
                [
                    style.margin_left,
                    current_y + row_height,
                    style.margin_left + style.content_width,
                    current_y + row_height,
                ],
                fill=style.line_color,
            )
            current_y += row_height

        element.bounding_box = (
            style.margin_left,
            current_y - len(element.table_data[:5]) * row_height - row_height,
            style.margin_left + style.content_width,
            current_y,
        )

        return current_y + 20

    def _draw_list(
        self,
        draw: ImageDraw.ImageDraw,
        element: ListElement,
        current_y: int,
        style: DocumentStyle,
        reading_order: int,
    ) -> int:
        line_height = self.font_size + 8

        for i, item in enumerate(element.items):
            marker = f"{i + 1}." if element.ordered else "•"
            marker_text = f"{marker} "
            draw.text(
                (style.margin_left, current_y),
                marker_text,
                font=self.font,
                fill=style.text_color,
            )

            bbox = draw.textbbox((0, 0), marker_text, font=self.font)
            marker_width = bbox[2] - bbox[0]

            draw.text(
                (style.margin_left + marker_width, current_y),
                item,
                font=self.font,
                fill=style.text_color,
            )

            current_y += line_height

        element.bounding_box = (
            style.margin_left,
            current_y - len(element.items) * line_height,
            style.margin_left + style.content_width,
            current_y,
        )

        return current_y + 10

    def _draw_date_stamp(
        self,
        draw: ImageDraw.ImageDraw,
        date_stamp: str,
        width: int,
        style: DocumentStyle,
    ):
        draw.text(
            (width - style.margin_right - 100, style.margin_top - 30),
            date_stamp,
            font=self.font,
            fill=(100, 100, 100),
        )

    def _draw_page_number(
        self,
        draw: ImageDraw.ImageDraw,
        page_number: str,
        width: int,
        height: int,
        style: DocumentStyle,
    ):
        draw.text(
            (width // 2 - 20, height - style.margin_bottom + 20),
            f"Page {page_number}",
            font=self.font,
            fill=(100, 100, 100),
        )

    def _draw_margin_lines(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        style: DocumentStyle,
    ):
        # Draw subtle margin guides
        draw.line(
            [(style.margin_left, 0), (style.margin_left, height)],
            fill=(230, 230, 230),
            width=1,
        )
        draw.line(
            [(width - style.margin_right, 0), (width - style.margin_right, height)],
            fill=(230, 230, 230),
            width=1,
        )

    def _apply_effects(self, img: Image.Image, style: DocumentStyle) -> Image.Image:
        """Apply noise and other effects to simulate scanned documents."""
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
        """Add Gaussian noise to image."""
        width, height = img.size
        img_array = img.copy()

        from PIL import Image

        noise = Image.new("RGB", (width, height))
        noise_draw = ImageDraw.Draw(noise)

        for _ in range(500):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            gray = random.randint(0, 255)
            noise_draw.point((x, y), fill=(gray, gray, gray))

        noise_img = noise.convert("RGB")
        img_array = Image.blend(img_array, noise_img, 0.05)

        return img_array


class DocumentDataGenerator:
    """Generates document data for various template types."""

    TEMPLATES = {
        DocumentTemplate.INVOICE: {
            "generate": "_generate_invoice",
        },
        DocumentTemplate.RECEIPT: {
            "generate": "_generate_receipt",
        },
        DocumentTemplate.FORM: {
            "generate": "_generate_form",
        },
        DocumentTemplate.LETTER: {
            "generate": "_generate_letter",
        },
        DocumentTemplate.REPORT: {
            "generate": "_generate_report",
        },
    }

    # Korean data
    KOREAN_COMPANIES = ["한국전자", "서울상사", "부산산업", "대구무역", "인천기업"]
    KOREAN_NAMES = ["김민수", "이영희", "박지영", "최동훈", "정수연"]
    KOREAN_ADDRESSES = ["서울시 강남구 테헤란로 123", "부산시 해운대구 센텀대로 45", "대구시 수성구 동대구로 78"]
    KOREAN_PRODUCTS = ["노트북", "스마트폰", "태블릿", "모니터", "키보드"]
    KOREAN_DEPARTMENTS = ["영업부", "마케팅부", "재무부", "인사부", "개발부"]

    # English data
    ENGLISH_COMPANIES = ["TechCorp Inc.", "Global Solutions", "Pacific Trading", "Atlantic Industries"]
    ENGLISH_NAMES = ["John Smith", "Jane Doe", "Mike Johnson", "Sarah Wilson", "Tom Brown"]
    ENGLISH_ADDRESSES = ["123 Main St, New York", "456 Oak Ave, Los Angeles", "789 Pine Rd, Chicago"]
    ENGLISH_PRODUCTS = ["Laptop", "Smartphone", "Tablet", "Monitor", "Keyboard"]
    ENGLISH_DEPARTMENTS = ["Sales", "Marketing", "Finance", "HR", "Engineering"]

    def __init__(self, lang: str = "ko"):
        self.lang = lang
        if lang == "ko":
            self.companies = self.KOREAN_COMPANIES
            self.names = self.KOREAN_NAMES
            self.addresses = self.KOREAN_ADDRESSES
            self.products = self.KOREAN_PRODUCTS
            self.departments = self.KOREAN_DEPARTMENTS
        else:
            self.companies = self.ENGLISH_COMPANIES
            self.names = self.ENGLISH_NAMES
            self.addresses = self.ENGLISH_ADDRESSES
            self.products = self.ENGLISH_PRODUCTS
            self.departments = self.ENGLISH_DEPARTMENTS

    def generate_document(
        self,
        template: DocumentTemplate = DocumentTemplate.INVOICE,
        style: DocumentStyle = None,
    ) -> Document:
        if style is None:
            style = self._random_style()

        gen_func = getattr(self, self.TEMPLATES[template]["generate"])
        blocks, page_number, date_stamp = gen_func(style)

        return Document(
            template=template,
            blocks=blocks,
            style=style,
            page_number=page_number,
            date_stamp=date_stamp,
        )

    def _generate_invoice(self, style: DocumentStyle) -> Tuple[List[DocumentBlock], str, str]:
        blocks = []

        # Header block
        header_elements = [
            TitleElement(
                f"Invoice" if self.lang == "en" else "인보이스",
                f"{random.choice(self.companies)}",
            ),
            ParagraphElement(
                f"Date: {self._random_date()}" if self.lang == "en"
                else f"일자: {self._random_date()}"
            ),
            ParagraphElement(
                f"Invoice #: INV-{random.randint(10000, 99999)}" if self.lang == "en"
                else f"인보이스 번호: INV-{random.randint(10000, 99999)}"
            ),
        ]
        blocks.append(DocumentBlock(block_type="header", elements=header_elements))

        # Bill To block
        bill_elements = [
            TitleElement(f"Bill To:" if self.lang == "en" else "청구 대상:", level=2),
            ParagraphElement(random.choice(self.names)),
            ParagraphElement(random.choice(self.addresses)),
        ]
        blocks.append(DocumentBlock(block_type="bill_to", elements=bill_elements))

        # Items block
        item_elements = [
            TitleElement(
                f"Items:" if self.lang == "en" else "품목:", level=2
            ),
        ]

        table_data = []
        headers = ["Item", "Qty", "Price", "Total"]
        for _ in range(random.randint(3, 6)):
            product = random.choice(self.products)
            qty = random.randint(1, 5)
            price = random.randint(100, 1000) * 1000
            total = qty * price
            table_data.append([product, str(qty), f"{price:,}", f"{total:,}"])

        item_elements.append(TableElement(table_data=table_data, headers=headers))
        blocks.append(DocumentBlock(block_type="items", elements=item_elements))

        # Total block
        subtotal = sum(int(row[3].replace(",", "")) for row in table_data)
        tax = int(subtotal * 0.1)
        total = subtotal + tax

        total_elements = [
            ParagraphElement(
                f"Subtotal: {subtotal:,}" if self.lang == "en"
                else f"소계: {subtotal:,}"
            ),
            ParagraphElement(
                f"Tax (10%): {tax:,}" if self.lang == "en"
                else f"세금 (10%): {tax:,}"
            ),
            TitleElement(
                f"Total: {total:,}" if self.lang == "en" else f"합계: {total:,}",
                level=2,
            ),
        ]
        blocks.append(DocumentBlock(block_type="total", elements=total_elements))

        # Footer
        footer_elements = [
            ParagraphElement(
                f"Thank you for your business!" if self.lang == "en"
                else f"거래해 주셔서 감사합니다!"
            ),
        ]
        blocks.append(DocumentBlock(block_type="footer", elements=footer_elements))

        date = self._random_date()
        page_num = str(random.randint(1, 10))

        return blocks, page_num, date

    def _generate_receipt(self, style: DocumentStyle) -> Tuple[List[DocumentBlock], str, str]:
        blocks = []

        # Store info
        header_elements = [
            TitleElement(
                f"Receipt" if self.lang == "en" else "영수증",
                random.choice(self.companies),
            ),
            ParagraphElement(
                f"Date: {self._random_date()}" if self.lang == "en"
                else f"일자: {self._random_date()}"
            ),
            ParagraphElement(
                f"Receipt #: RCP-{random.randint(10000, 99999)}" if self.lang == "en"
                else f"영수증 번호: RCP-{random.randint(10000, 99999)}"
            ),
        ]
        blocks.append(DocumentBlock(block_type="header", elements=header_elements))

        # Items
        item_elements = [
            TitleElement(
                f"Purchased Items:" if self.lang == "en" else "구매 품목:", level=2
            ),
        ]

        table_data = []
        headers = ["Item", "Qty", "Price"]
        for _ in range(random.randint(2, 5)):
            product = random.choice(self.products)
            qty = random.randint(1, 3)
            price = random.randint(10, 200) * 1000
            table_data.append([product, str(qty), f"{price:,}"])

        item_elements.append(TableElement(table_data=table_data, headers=headers))
        blocks.append(DocumentBlock(block_type="items", elements=item_elements))

        # Total
        total = sum(int(row[2].replace(",", "")) for row in table_data)
        total_elements = [
            TitleElement(
                f"Total: {total:,}" if self.lang == "en" else f"합계: {total:,}",
                level=2,
            ),
            ParagraphElement(
                f"Payment: Cash" if self.lang == "en" else f"결제: 현금"
            ),
        ]
        blocks.append(DocumentBlock(block_type="total", elements=total_elements))

        # Footer
        footer_elements = [
            ParagraphElement(
                f"No returns after 30 days." if self.lang == "en"
                else f"30일 이내 반품 가능."
            ),
        ]
        blocks.append(DocumentBlock(block_type="footer", elements=footer_elements))

        date = self._random_date()
        page_num = "1"

        return blocks, page_num, date

    def _generate_form(self, style: DocumentStyle) -> Tuple[List[DocumentBlock], str, str]:
        blocks = []

        header_elements = [
            TitleElement(
                f"Application Form" if self.lang == "en" else "신청서",
                f"{random.choice(self.companies)}",
            ),
        ]
        blocks.append(DocumentBlock(block_type="header", elements=header_elements))

        # Form fields
        form_elements = [
            TitleElement(f"Personal Information", level=2),
            ListElement([
                f"Name: {random.choice(self.names)}",
                f"Department: {random.choice(self.departments)}",
                f"Date: {self._random_date()}",
            ], ordered=True),
        ]
        blocks.append(DocumentBlock(block_type="form", elements=form_elements))

        # Checkboxes
        check_elements = [
            TitleElement(f"Options", level=2),
            ListElement([
                "[ ] Option 1" if self.lang == "en" else "[ ] 옵션 1",
                "[x] Option 2" if self.lang == "en" else "[x] 옵션 2",
                "[ ] Option 3" if self.lang == "en" else "[ ] 옵션 3",
            ], ordered=False),
        ]
        blocks.append(DocumentBlock(block_type="checkboxes", elements=check_elements))

        # Signature
        sig_elements = [
            ParagraphElement(""),
            ParagraphElement("_" * 30),
            ParagraphElement(
                f"Signature / Date" if self.lang == "en" else f"서명 / 일자"
            ),
        ]
        blocks.append(DocumentBlock(block_type="signature", elements=sig_elements))

        date = self._random_date()
        page_num = "1"

        return blocks, page_num, date

    def _generate_letter(self, style: DocumentStyle) -> Tuple[List[DocumentBlock], str, str]:
        blocks = []

        # Header
        header_elements = [
            TitleElement(
                f"Official Letter" if self.lang == "en" else "공문",
                f"{random.choice(self.companies)}",
            ),
            ParagraphElement(f"Date: {self._random_date()}"),
            ParagraphElement(f"To: {random.choice(self.names)}"),
        ]
        blocks.append(DocumentBlock(block_type="header", elements=header_elements))

        # Subject
        subject_elements = [
            ParagraphElement(
                f"Subject: Request for Information" if self.lang == "en"
                else f"제목: 정보 요청 사항"
            ),
        ]
        blocks.append(DocumentBlock(block_type="subject", elements=subject_elements))

        # Body
        body_text = (
            f"Dear {random.choice(self.names)},\n\n"
            f"We would like to request your assistance with the following matter. "
            f"Please review the attached documents and provide your feedback by next week."
            if self.lang == "en"
            else f"{random.choice(self.names)} 귀하께,\n\n"
            f"다음 사항에 대해 귀하의 협조를 요청드립니다. "
            f"첨부 문서를 검토하시고 다음 주까지 피드백을 제공해 주시기 바랍니다."
        )
        body_elements = [ParagraphElement(body_text)]
        blocks.append(DocumentBlock(block_type="body", elements=body_elements))

        # Closing
        closing_elements = [
            ParagraphElement(
                f"Best regards," if self.lang == "en" else f"감사합니다."
            ),
            ParagraphElement(""),
            ParagraphElement(random.choice(self.names)),
            ParagraphElement(random.choice(self.departments)),
        ]
        blocks.append(DocumentBlock(block_type="closing", elements=closing_elements))

        date = self._random_date()
        page_num = "1"

        return blocks, page_num, date

    def _generate_report(self, style: DocumentStyle) -> Tuple[List[DocumentBlock], str, str]:
        blocks = []

        # Title
        title_elements = [
            TitleElement(
                f"Monthly Report" if self.lang == "en" else "월간 보고서",
                f"{self._random_date().split('-')[1]} Month Summary" if self.lang == "en"
                else f"{self._random_date().split('-')[1]}월 요약",
            ),
        ]
        blocks.append(DocumentBlock(block_type="title", elements=title_elements))

        # Summary
        summary_elements = [
            TitleElement(f"Executive Summary", level=2),
            ParagraphElement(
                f"This report summarizes the key activities and achievements for the month. "
                f"Overall performance has been satisfactory with notable improvements in key areas."
                if self.lang == "en"
                else f"이 보고서는 이번 달 주요 활동과 성과를 요약합니다. "
                f"전반적인 성과는 양호하며 주요 분야에서 주목할 만한 개선이 있었습니다."
            ),
        ]
        blocks.append(DocumentBlock(block_type="summary", elements=summary_elements))

        # Highlights
        highlights = [
            f"• Achieved {random.randint(90, 120)}% of monthly targets" if self.lang == "en"
            else f"• 월간 목표의 {random.randint(90, 120)}% 달성",
            f"• Launched {random.randint(1, 3)} new initiatives" if self.lang == "en"
            else f"• {random.randint(1, 3)}개 신규 계획 발동",
            f"• Customer satisfaction: {random.randint(85, 99)}%" if self.lang == "en"
            else f"• 고객 만족도: {random.randint(85, 99)}%",
        ]
        highlights_elements = [
            TitleElement(f"Key Highlights", level=2),
            ListElement(highlights, ordered=False),
        ]
        blocks.append(DocumentBlock(block_type="highlights", elements=highlights_elements))

        # Conclusion
        conclusion_elements = [
            TitleElement(f"Next Steps", level=2),
            ListElement([
                f"1. Review action items" if self.lang == "en" else f"1. 조치 사항 검토",
                f"2. Schedule follow-up meetings" if self.lang == "en" else f"2. 후속 회의 일정 잡기",
                f"3. Prepare detailed plans" if self.lang == "en" else f"3. 상세 계획 수립",
            ], ordered=True),
        ]
        blocks.append(DocumentBlock(block_type="conclusion", elements=conclusion_elements))

        date = self._random_date()
        page_num = "1"

        return blocks, page_num, date

    def _random_date(self) -> str:
        year = random.randint(2023, 2025)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        return f"{year}-{month:02d}-{day:02d}"

    def _random_style(self) -> DocumentStyle:
        styles = [
            DocumentStyle(
                background_color=(255, 255, 255),
                title_color=(0, 0, 150),
                add_noise=True,
            ),
            DocumentStyle(
                background_color=(250, 250, 245),
                title_color=(0, 100, 0),
                add_noise=True,
                add_blur=True,
            ),
            DocumentStyle(
                background_color=(255, 253, 240),
                title_color=(139, 69, 19),
                add_noise=True,
                add_contrast=True,
            ),
            DocumentStyle(
                background_color=(255, 255, 255),
                title_color=(128, 0, 128),
                add_noise=False,
            ),
        ]
        return random.choice(styles)


class DocumentGenerator(BaseGenerator):
    """Main generator class for document image generation."""

    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str = "ko",
    ):
        super().__init__(output_dir, font_dir, lang)
        self.data_generator = DocumentDataGenerator(lang)

    def generate(
        self,
        num_images: int,
        template: Optional[str] = None,
        font_size_range: Tuple[int, int] = (10, 14),
        add_noise: bool = True,
        add_blur: bool = False,
    ) -> List[Dict[str, Any]]:
        if template:
            try:
                doc_template = DocumentTemplate(template)
            except ValueError:
                logger.warning(f"Unknown template '{template}', using random templates")
                doc_template = None
        else:
            doc_template = None

        templates = (
            [DocumentTemplate(template)] if doc_template
            else list(DocumentTemplate)
        )

        metadata = []
        for idx in tqdm(range(num_images), desc="Generating document images"):
            selected_template = random.choice(templates)

            # Generate document
            style = self.data_generator._random_style()
            style.add_noise = add_noise
            style.add_blur = add_blur

            document = self.data_generator.generate_document(
                template=selected_template,
                style=style,
            )

            # Render document
            font_path = random.choice(self.font_paths)
            font_size = random.randint(*font_size_range)
            renderer = DocumentRenderer(font_path, font_size)
            image, ground_truth = renderer.render(document)

            # Save image
            filename = f"document_{idx:05d}.png"
            self.save_image(image, filename)

            # Build metadata
            gt = document.to_ground_truth()
            metadata.append({
                "file_name": str(self.output_dir / filename),
                "template": selected_template.value,
                "ground_truth": gt,
                "elements_count": len(gt["elements"]),
                "font_size": font_size,
                "add_noise": add_noise,
                "add_blur": add_blur,
            })

        return metadata
