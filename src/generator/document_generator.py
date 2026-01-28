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

    # Japanese data
    JAPANESE_COMPANIES = ["東京電子", "大阪商事", "名古屋産業", "横浜貿易", "福岡企業"]
    JAPANESE_NAMES = ["田中太郎", "鈴木花子", "山田一郎", "佐藤美咲", "伊藤健太"]
    JAPANESE_ADDRESSES = ["東京都渋谷区神宮前1-2-3", "大阪府大阪市北区梅田4-5-6", "愛知県名古屋市中区栄7-8-9"]
    JAPANESE_PRODUCTS = ["ノートパソコン", "スマートフォン", "タブレット", "モニター", "キーボード"]
    JAPANESE_DEPARTMENTS = ["営業部", "マーケティング部", "財務部", "人事部", "開発部"]

    # Hindi data
    HINDI_COMPANIES = ["भारत इलेक्ट्रॉनिक्स", "दिल्ली ट्रेडिंग", "मुंबई इंडस्ट्रीज", "चेन्नई कॉर्पोरेशन", "कोलकाता एंटरप्राइज"]
    HINDI_NAMES = ["राहुल शर्मा", "प्रिया गुप्ता", "अमित कुमार", "सुनीता देवी", "विकास सिंह"]
    HINDI_ADDRESSES = ["123 महात्मा गांधी रोड, नई दिल्ली", "456 जुहू बीच रोड, मुंबई", "789 एमजी रोड, बैंगलोर"]
    HINDI_PRODUCTS = ["लैपटॉप", "स्मार्टफोन", "टैबलेट", "मॉनिटर", "कीबोर्ड"]
    HINDI_DEPARTMENTS = ["बिक्री विभाग", "मार्केटिंग विभाग", "वित्त विभाग", "मानव संसाधन विभाग", "विकास विभाग"]

    # Language-specific labels
    LABELS = {
        "ko": {
            "invoice": "인보이스", "receipt": "영수증", "form": "신청서", "letter": "공문", "report": "월간 보고서",
            "date": "일자", "invoice_no": "인보이스 번호", "receipt_no": "영수증 번호",
            "bill_to": "청구 대상", "items": "품목", "purchased_items": "구매 품목",
            "subtotal": "소계", "tax": "세금", "total": "합계", "payment": "결제", "cash": "현금",
            "thank_you": "거래해 주셔서 감사합니다!", "return_policy": "30일 이내 반품 가능.",
            "personal_info": "개인 정보", "name": "이름", "department": "부서",
            "options": "옵션", "option": "옵션", "signature": "서명", "to": "수신",
            "subject": "제목", "request_info": "정보 요청 사항", "dear": "귀하께",
            "letter_body": "다음 사항에 대해 귀하의 협조를 요청드립니다. 첨부 문서를 검토하시고 다음 주까지 피드백을 제공해 주시기 바랍니다.",
            "regards": "감사합니다.", "month_summary": "월 요약", "exec_summary": "요약",
            "summary_text": "이 보고서는 이번 달 주요 활동과 성과를 요약합니다. 전반적인 성과는 양호하며 주요 분야에서 주목할 만한 개선이 있었습니다.",
            "highlights": "주요 성과", "achieved": "월간 목표의 {}% 달성", "launched": "{}개 신규 계획 발동",
            "satisfaction": "고객 만족도: {}%", "next_steps": "다음 단계",
            "review_items": "조치 사항 검토", "schedule_meetings": "후속 회의 일정 잡기", "prepare_plans": "상세 계획 수립",
        },
        "en": {
            "invoice": "Invoice", "receipt": "Receipt", "form": "Application Form", "letter": "Official Letter", "report": "Monthly Report",
            "date": "Date", "invoice_no": "Invoice #", "receipt_no": "Receipt #",
            "bill_to": "Bill To", "items": "Items", "purchased_items": "Purchased Items",
            "subtotal": "Subtotal", "tax": "Tax", "total": "Total", "payment": "Payment", "cash": "Cash",
            "thank_you": "Thank you for your business!", "return_policy": "No returns after 30 days.",
            "personal_info": "Personal Information", "name": "Name", "department": "Department",
            "options": "Options", "option": "Option", "signature": "Signature", "to": "To",
            "subject": "Subject", "request_info": "Request for Information", "dear": "Dear",
            "letter_body": "We would like to request your assistance with the following matter. Please review the attached documents and provide your feedback by next week.",
            "regards": "Best regards,", "month_summary": "Month Summary", "exec_summary": "Executive Summary",
            "summary_text": "This report summarizes the key activities and achievements for the month. Overall performance has been satisfactory with notable improvements in key areas.",
            "highlights": "Key Highlights", "achieved": "Achieved {}% of monthly targets", "launched": "Launched {} new initiatives",
            "satisfaction": "Customer satisfaction: {}%", "next_steps": "Next Steps",
            "review_items": "Review action items", "schedule_meetings": "Schedule follow-up meetings", "prepare_plans": "Prepare detailed plans",
        },
        "ja": {
            "invoice": "請求書", "receipt": "領収書", "form": "申請書", "letter": "公文書", "report": "月次報告書",
            "date": "日付", "invoice_no": "請求書番号", "receipt_no": "領収書番号",
            "bill_to": "請求先", "items": "品目", "purchased_items": "購入品目",
            "subtotal": "小計", "tax": "税金", "total": "合計", "payment": "支払い", "cash": "現金",
            "thank_you": "ご利用ありがとうございます！", "return_policy": "30日以内に返品可能。",
            "personal_info": "個人情報", "name": "名前", "department": "部署",
            "options": "オプション", "option": "オプション", "signature": "署名", "to": "宛先",
            "subject": "件名", "request_info": "情報依頼", "dear": "様",
            "letter_body": "以下の件についてご協力をお願いいたします。添付の書類をご確認の上、来週までにフィードバックをお願いいたします。",
            "regards": "敬具", "month_summary": "月の概要", "exec_summary": "概要",
            "summary_text": "この報告書は今月の主要な活動と成果をまとめたものです。全体的なパフォーマンスは良好で、主要分野で顕著な改善が見られました。",
            "highlights": "主なハイライト", "achieved": "月間目標の{}%を達成", "launched": "{}件の新規施策を開始",
            "satisfaction": "顧客満足度: {}%", "next_steps": "次のステップ",
            "review_items": "アクション項目の確認", "schedule_meetings": "フォローアップ会議の予定", "prepare_plans": "詳細計画の作成",
        },
        "hi": {
            "invoice": "चालान", "receipt": "रसीद", "form": "आवेदन पत्र", "letter": "आधिकारिक पत्र", "report": "मासिक रिपोर्ट",
            "date": "तारीख", "invoice_no": "चालान संख्या", "receipt_no": "रसीद संख्या",
            "bill_to": "बिल प्राप्तकर्ता", "items": "वस्तुएं", "purchased_items": "खरीदी गई वस्तुएं",
            "subtotal": "उप-योग", "tax": "कर", "total": "कुल", "payment": "भुगतान", "cash": "नकद",
            "thank_you": "आपके व्यापार के लिए धन्यवाद!", "return_policy": "30 दिनों के बाद वापसी नहीं।",
            "personal_info": "व्यक्तिगत जानकारी", "name": "नाम", "department": "विभाग",
            "options": "विकल्प", "option": "विकल्प", "signature": "हस्ताक्षर", "to": "प्रति",
            "subject": "विषय", "request_info": "जानकारी का अनुरोध", "dear": "प्रिय",
            "letter_body": "हम निम्नलिखित मामले में आपकी सहायता का अनुरोध करना चाहते हैं। कृपया संलग्न दस्तावेजों की समीक्षा करें और अगले सप्ताह तक अपनी प्रतिक्रिया दें।",
            "regards": "सादर,", "month_summary": "माह सारांश", "exec_summary": "कार्यकारी सारांश",
            "summary_text": "यह रिपोर्ट इस महीने की प्रमुख गतिविधियों और उपलब्धियों का सारांश प्रस्तुत करती है। समग्र प्रदर्शन संतोषजनक रहा है।",
            "highlights": "मुख्य विशेषताएं", "achieved": "मासिक लक्ष्य का {}% प्राप्त", "launched": "{} नई पहल शुरू",
            "satisfaction": "ग्राहक संतुष्टि: {}%", "next_steps": "अगले कदम",
            "review_items": "कार्य आइटम की समीक्षा", "schedule_meetings": "फॉलो-अप बैठकें निर्धारित करें", "prepare_plans": "विस्तृत योजना तैयार करें",
        },
    }

    def __init__(self, lang: str = "ko"):
        self.lang = lang
        # Set labels based on language (default to English for unknown languages)
        self.labels = self.LABELS.get(lang, self.LABELS["en"])

        if lang == "ko":
            self.companies = self.KOREAN_COMPANIES
            self.names = self.KOREAN_NAMES
            self.addresses = self.KOREAN_ADDRESSES
            self.products = self.KOREAN_PRODUCTS
            self.departments = self.KOREAN_DEPARTMENTS
        elif lang == "ja":
            self.companies = self.JAPANESE_COMPANIES
            self.names = self.JAPANESE_NAMES
            self.addresses = self.JAPANESE_ADDRESSES
            self.products = self.JAPANESE_PRODUCTS
            self.departments = self.JAPANESE_DEPARTMENTS
        elif lang == "hi":
            self.companies = self.HINDI_COMPANIES
            self.names = self.HINDI_NAMES
            self.addresses = self.HINDI_ADDRESSES
            self.products = self.HINDI_PRODUCTS
            self.departments = self.HINDI_DEPARTMENTS
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
        L = self.labels

        # Header block
        header_elements = [
            TitleElement(
                L["invoice"],
                f"{random.choice(self.companies)}",
            ),
            ParagraphElement(f"{L['date']}: {self._random_date()}"),
            ParagraphElement(f"{L['invoice_no']}: INV-{random.randint(10000, 99999)}"),
        ]
        blocks.append(DocumentBlock(block_type="header", elements=header_elements))

        # Bill To block
        bill_elements = [
            TitleElement(f"{L['bill_to']}:", level=2),
            ParagraphElement(random.choice(self.names)),
            ParagraphElement(random.choice(self.addresses)),
        ]
        blocks.append(DocumentBlock(block_type="bill_to", elements=bill_elements))

        # Items block
        item_elements = [
            TitleElement(f"{L['items']}:", level=2),
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
            ParagraphElement(f"{L['subtotal']}: {subtotal:,}"),
            ParagraphElement(f"{L['tax']} (10%): {tax:,}"),
            TitleElement(f"{L['total']}: {total:,}", level=2),
        ]
        blocks.append(DocumentBlock(block_type="total", elements=total_elements))

        # Footer
        footer_elements = [
            ParagraphElement(L["thank_you"]),
        ]
        blocks.append(DocumentBlock(block_type="footer", elements=footer_elements))

        date = self._random_date()
        page_num = str(random.randint(1, 10))

        return blocks, page_num, date

    def _generate_receipt(self, style: DocumentStyle) -> Tuple[List[DocumentBlock], str, str]:
        blocks = []
        L = self.labels

        # Store info
        header_elements = [
            TitleElement(L["receipt"], random.choice(self.companies)),
            ParagraphElement(f"{L['date']}: {self._random_date()}"),
            ParagraphElement(f"{L['receipt_no']}: RCP-{random.randint(10000, 99999)}"),
        ]
        blocks.append(DocumentBlock(block_type="header", elements=header_elements))

        # Items
        item_elements = [
            TitleElement(f"{L['purchased_items']}:", level=2),
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
            TitleElement(f"{L['total']}: {total:,}", level=2),
            ParagraphElement(f"{L['payment']}: {L['cash']}"),
        ]
        blocks.append(DocumentBlock(block_type="total", elements=total_elements))

        # Footer
        footer_elements = [
            ParagraphElement(L["return_policy"]),
        ]
        blocks.append(DocumentBlock(block_type="footer", elements=footer_elements))

        date = self._random_date()
        page_num = "1"

        return blocks, page_num, date

    def _generate_form(self, style: DocumentStyle) -> Tuple[List[DocumentBlock], str, str]:
        blocks = []
        L = self.labels

        header_elements = [
            TitleElement(L["form"], f"{random.choice(self.companies)}"),
        ]
        blocks.append(DocumentBlock(block_type="header", elements=header_elements))

        # Form fields
        form_elements = [
            TitleElement(L["personal_info"], level=2),
            ListElement([
                f"{L['name']}: {random.choice(self.names)}",
                f"{L['department']}: {random.choice(self.departments)}",
                f"{L['date']}: {self._random_date()}",
            ], ordered=True),
        ]
        blocks.append(DocumentBlock(block_type="form", elements=form_elements))

        # Checkboxes
        check_elements = [
            TitleElement(L["options"], level=2),
            ListElement([
                f"[ ] {L['option']} 1",
                f"[x] {L['option']} 2",
                f"[ ] {L['option']} 3",
            ], ordered=False),
        ]
        blocks.append(DocumentBlock(block_type="checkboxes", elements=check_elements))

        # Signature
        sig_elements = [
            ParagraphElement(""),
            ParagraphElement("_" * 30),
            ParagraphElement(f"{L['signature']} / {L['date']}"),
        ]
        blocks.append(DocumentBlock(block_type="signature", elements=sig_elements))

        date = self._random_date()
        page_num = "1"

        return blocks, page_num, date

    def _generate_letter(self, style: DocumentStyle) -> Tuple[List[DocumentBlock], str, str]:
        blocks = []
        L = self.labels
        recipient_name = random.choice(self.names)

        # Header
        header_elements = [
            TitleElement(L["letter"], f"{random.choice(self.companies)}"),
            ParagraphElement(f"{L['date']}: {self._random_date()}"),
            ParagraphElement(f"{L['to']}: {recipient_name}"),
        ]
        blocks.append(DocumentBlock(block_type="header", elements=header_elements))

        # Subject
        subject_elements = [
            ParagraphElement(f"{L['subject']}: {L['request_info']}"),
        ]
        blocks.append(DocumentBlock(block_type="subject", elements=subject_elements))

        # Body
        body_text = f"{L['dear']} {recipient_name},\n\n{L['letter_body']}"
        body_elements = [ParagraphElement(body_text)]
        blocks.append(DocumentBlock(block_type="body", elements=body_elements))

        # Closing
        closing_elements = [
            ParagraphElement(L["regards"]),
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
        L = self.labels
        month = self._random_date().split('-')[1]

        # Title
        title_elements = [
            TitleElement(L["report"], f"{month} {L['month_summary']}"),
        ]
        blocks.append(DocumentBlock(block_type="title", elements=title_elements))

        # Summary
        summary_elements = [
            TitleElement(L["exec_summary"], level=2),
            ParagraphElement(L["summary_text"]),
        ]
        blocks.append(DocumentBlock(block_type="summary", elements=summary_elements))

        # Highlights
        achieved_pct = random.randint(90, 120)
        launched_count = random.randint(1, 3)
        satisfaction_pct = random.randint(85, 99)
        highlights = [
            f"• {L['achieved'].format(achieved_pct)}",
            f"• {L['launched'].format(launched_count)}",
            f"• {L['satisfaction'].format(satisfaction_pct)}",
        ]
        highlights_elements = [
            TitleElement(L["highlights"], level=2),
            ListElement(highlights, ordered=False),
        ]
        blocks.append(DocumentBlock(block_type="highlights", elements=highlights_elements))

        # Conclusion
        conclusion_elements = [
            TitleElement(L["next_steps"], level=2),
            ListElement([
                f"1. {L['review_items']}",
                f"2. {L['schedule_meetings']}",
                f"3. {L['prepare_plans']}",
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
