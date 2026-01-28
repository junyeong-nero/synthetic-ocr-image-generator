import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from generator.base import BaseGenerator

logger = logging.getLogger(__name__)


class BorderStyle(Enum):
    SOLID = "solid"
    NONE = "none"


class CellAlignment(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class TableCell:
    text: str
    colspan: int = 1
    rowspan: int = 1
    is_header: bool = False


@dataclass
class TableStyle:
    border_style: BorderStyle = BorderStyle.SOLID
    border_color: Tuple[int, int, int] = (0, 0, 0)
    border_width: int = 1
    cell_padding: int = 8
    header_bg_color: Tuple[int, int, int] = (220, 220, 220)
    cell_bg_color: Tuple[int, int, int] = (255, 255, 255)
    alt_row_color: Optional[Tuple[int, int, int]] = None
    text_color: Tuple[int, int, int] = (0, 0, 0)
    header_text_color: Tuple[int, int, int] = (0, 0, 0)
    alignment: CellAlignment = CellAlignment.CENTER


@dataclass
class Table:
    cells: List[List[TableCell]]
    style: TableStyle = field(default_factory=TableStyle)
    cell_bounding_boxes: List[List[Tuple[int, int, int, int]]] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        return len(self.cells)

    @property
    def num_cols(self) -> int:
        return max(len(row) for row in self.cells) if self.cells else 0

    def to_html(self) -> str:
        html_parts = ["<table>"]
        for row_idx, row in enumerate(self.cells):
            html_parts.append("  <tr>")
            for cell in row:
                tag = "th" if cell.is_header else "td"
                attrs = []
                if cell.colspan > 1:
                    attrs.append(f'colspan="{cell.colspan}"')
                if cell.rowspan > 1:
                    attrs.append(f'rowspan="{cell.rowspan}"')
                attr_str = " " + " ".join(attrs) if attrs else ""
                html_parts.append(f"    <{tag}{attr_str}>{cell.text}</{tag}>")
            html_parts.append("  </tr>")
        html_parts.append("</table>")
        return "\n".join(html_parts)

    def to_json(self) -> Dict[str, Any]:
        """Convert table to JSON structure with cell positions for TSR evaluation."""
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "style": {
                "border_style": self.style.border_style.value,
                "border_color": self.style.border_color,
                "border_width": self.style.border_width,
                "cell_padding": self.style.cell_padding,
                "header_bg_color": self.style.header_bg_color,
                "cell_bg_color": self.style.cell_bg_color,
                "alt_row_color": self.style.alt_row_color,
                "text_color": self.style.text_color,
                "header_text_color": self.style.header_text_color,
                "alignment": self.style.alignment.value,
            },
            "cells": [
                {
                    "row": row_idx,
                    "col": col_idx,
                    "text": cell.text,
                    "colspan": cell.colspan,
                    "rowspan": cell.rowspan,
                    "is_header": cell.is_header,
                    "bounding_box": self.cell_bounding_boxes[row_idx][col_idx] if row_idx < len(self.cell_bounding_boxes) and col_idx < len(self.cell_bounding_boxes[row_idx]) else None,
                }
                for row_idx, row in enumerate(self.cells)
                for col_idx, cell in enumerate(row)
            ],
            "structure": [
                {
                    "row": i,
                    "num_cells": len(row),
                }
                for i, row in enumerate(self.cells)
            ],
        }


class TableRenderer:

    def __init__(self, font_path: str, font_size: int = 14):
        self.font_size = font_size
        try:
            self.font = ImageFont.truetype(font_path, font_size)
            self.header_font = ImageFont.truetype(font_path, font_size)
        except IOError:
            logger.warning(f"Font '{font_path}' not found. Using default.")
            self.font = ImageFont.load_default()
            self.header_font = self.font

    def render(self, table: Table) -> Image.Image:
        col_widths, row_heights = self._calculate_dimensions(table)

        total_width = sum(col_widths) + table.style.border_width
        total_height = sum(row_heights) + table.style.border_width

        img = Image.new("RGB", (total_width, total_height), table.style.cell_bg_color)
        draw = ImageDraw.Draw(img)

        self._draw_cells(draw, table, col_widths, row_heights)

        if table.style.border_style == BorderStyle.SOLID:
            self._draw_borders(draw, table, col_widths, row_heights)

        return img

    def _calculate_dimensions(self, table: Table) -> Tuple[List[int], List[int]]:
        padding = table.style.cell_padding
        min_cell_width = 40
        min_cell_height = self.font_size + padding * 2

        col_widths = [min_cell_width] * table.num_cols
        row_heights = [min_cell_height] * table.num_rows

        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)

        for row_idx, row in enumerate(table.cells):
            col_idx = 0
            for cell in row:
                font = self.header_font if cell.is_header else self.font
                bbox = dummy_draw.textbbox((0, 0), cell.text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                cell_width = text_width + padding * 2
                cell_height = text_height + padding * 2

                if cell.colspan == 1:
                    col_widths[col_idx] = max(col_widths[col_idx], cell_width)
                if cell.rowspan == 1:
                    row_heights[row_idx] = max(row_heights[row_idx], cell_height)

                col_idx += cell.colspan

        return col_widths, row_heights

    def _draw_cells(
        self,
        draw: ImageDraw.ImageDraw,
        table: Table,
        col_widths: List[int],
        row_heights: List[int],
    ):
        style = table.style
        y = 0
        table.cell_bounding_boxes = []

        for row_idx, row in enumerate(table.cells):
            x = 0
            col_idx = 0
            row_boxes = []

            for cell in row:
                cell_width = sum(col_widths[col_idx : col_idx + cell.colspan])
                cell_height = sum(row_heights[row_idx : row_idx + cell.rowspan])

                if cell.is_header:
                    bg_color = style.header_bg_color
                elif style.alt_row_color and row_idx % 2 == 1:
                    bg_color = style.alt_row_color
                else:
                    bg_color = style.cell_bg_color

                draw.rectangle(
                    [x, y, x + cell_width, y + cell_height],
                    fill=bg_color,
                )

                font = self.header_font if cell.is_header else self.font
                text_color = style.header_text_color if cell.is_header else style.text_color

                bbox = draw.textbbox((0, 0), cell.text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                if style.alignment == CellAlignment.LEFT:
                    text_x = x + style.cell_padding
                elif style.alignment == CellAlignment.RIGHT:
                    text_x = x + cell_width - text_width - style.cell_padding
                else:
                    text_x = x + (cell_width - text_width) // 2

                text_y = y + (cell_height - text_height) // 2

                draw.text((text_x, text_y), cell.text, font=font, fill=text_color)

                # Store bounding box for JSON ground truth
                row_boxes.append((x, y, x + cell_width, y + cell_height))

                x += cell_width
                col_idx += cell.colspan

            table.cell_bounding_boxes.append(row_boxes)
            y += row_heights[row_idx]

    def _draw_borders(
        self,
        draw: ImageDraw.ImageDraw,
        table: Table,
        col_widths: List[int],
        row_heights: List[int],
    ):
        style = table.style
        total_width = sum(col_widths)
        total_height = sum(row_heights)

        y = 0
        for row_height in row_heights:
            draw.line([(0, y), (total_width, y)], fill=style.border_color, width=style.border_width)
            y += row_height
        draw.line([(0, y), (total_width, y)], fill=style.border_color, width=style.border_width)

        x = 0
        for col_width in col_widths:
            draw.line([(x, 0), (x, total_height)], fill=style.border_color, width=style.border_width)
            x += col_width
        draw.line([(x, 0), (x, total_height)], fill=style.border_color, width=style.border_width)


class TableDataGenerator:

    TEMPLATES = {
        "invoice": {
            "headers": ["Item", "Qty", "Price", "Total"],
            "row_generator": "_generate_invoice_row",
        },
        "schedule": {
            "headers": ["Time", "Mon", "Tue", "Wed", "Thu", "Fri"],
            "row_generator": "_generate_schedule_row",
        },
        "product": {
            "headers": ["Product", "Category", "Price", "Stock"],
            "row_generator": "_generate_product_row",
        },
        "contact": {
            "headers": ["Name", "Phone", "Email"],
            "row_generator": "_generate_contact_row",
        },
    }

    KOREAN_HEADERS = {
        "invoice": ["품목", "수량", "가격", "합계"],
        "schedule": ["시간", "월", "화", "수", "목", "금"],
        "product": ["제품", "분류", "가격", "재고"],
        "contact": ["이름", "전화", "이메일"],
    }

    ENGLISH_HEADERS = {
        "invoice": ["Item", "Qty", "Price", "Total"],
        "schedule": ["Time", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "product": ["Product", "Category", "Price", "Stock"],
        "contact": ["Name", "Phone", "Email"],
    }

    JAPANESE_HEADERS = {
        "invoice": ["品目", "数量", "価格", "合計"],
        "schedule": ["時間", "月", "火", "水", "木", "金"],
        "product": ["製品", "カテゴリ", "価格", "在庫"],
        "contact": ["名前", "電話", "メール"],
    }

    HINDI_HEADERS = {
        "invoice": ["वस्तु", "मात्रा", "मूल्य", "कुल"],
        "schedule": ["समय", "सोम", "मंगल", "बुध", "गुरु", "शुक्र"],
        "product": ["उत्पाद", "श्रेणी", "मूल्य", "स्टॉक"],
        "contact": ["नाम", "फोन", "ईमेल"],
    }

    KOREAN_ITEMS = ["사과", "바나나", "우유", "빵", "커피", "라면", "김치", "두부", "계란", "생수"]
    KOREAN_CATEGORIES = ["식품", "음료", "생활용품", "전자제품", "의류"]
    KOREAN_NAMES = ["김민수", "이영희", "박지영", "최동훈", "정수연", "강태호", "윤서연", "임재현"]
    KOREAN_SUBJECTS = ["국어", "수학", "영어", "과학", "사회", "체육", "음악", "미술"]

    ENGLISH_ITEMS = ["Apple", "Banana", "Milk", "Bread", "Coffee", "Pasta", "Cheese", "Eggs", "Water", "Juice"]
    ENGLISH_CATEGORIES = ["Food", "Beverage", "Household", "Electronics", "Clothing"]
    ENGLISH_NAMES = ["John Smith", "Jane Doe", "Mike Johnson", "Sarah Wilson", "Tom Brown"]
    ENGLISH_SUBJECTS = ["Math", "English", "Science", "History", "Art", "PE", "Music"]

    JAPANESE_ITEMS = ["りんご", "バナナ", "牛乳", "パン", "コーヒー", "ラーメン", "寿司", "豆腐", "卵", "水"]
    JAPANESE_CATEGORIES = ["食品", "飲料", "日用品", "電子機器", "衣料品"]
    JAPANESE_NAMES = ["田中太郎", "鈴木花子", "山田一郎", "佐藤美咲", "伊藤健太", "渡辺和子", "高橋誠", "中村優子"]
    JAPANESE_SUBJECTS = ["国語", "数学", "英語", "理科", "社会", "体育", "音楽", "美術"]

    HINDI_ITEMS = ["सेब", "केला", "दूध", "रोटी", "कॉफी", "चावल", "दाल", "पनीर", "अंडा", "पानी"]
    HINDI_CATEGORIES = ["खाद्य", "पेय", "घरेलू", "इलेक्ट्रॉनिक्स", "कपड़े"]
    HINDI_NAMES = ["राहुल शर्मा", "प्रिया गुप्ता", "अमित कुमार", "सुनीता देवी", "विकास सिंह", "अनीता वर्मा", "राजेश पटेल", "नेहा अग्रवाल"]
    HINDI_SUBJECTS = ["हिंदी", "गणित", "अंग्रेजी", "विज्ञान", "सामाजिक", "शारीरिक", "संगीत", "कला"]

    def __init__(self, lang: str = "ko"):
        self.lang = lang
        if lang == "ko":
            self.items = self.KOREAN_ITEMS
            self.categories = self.KOREAN_CATEGORIES
            self.names = self.KOREAN_NAMES
            self.subjects = self.KOREAN_SUBJECTS
            self.headers = self.KOREAN_HEADERS
        elif lang == "ja":
            self.items = self.JAPANESE_ITEMS
            self.categories = self.JAPANESE_CATEGORIES
            self.names = self.JAPANESE_NAMES
            self.subjects = self.JAPANESE_SUBJECTS
            self.headers = self.JAPANESE_HEADERS
        elif lang == "hi":
            self.items = self.HINDI_ITEMS
            self.categories = self.HINDI_CATEGORIES
            self.names = self.HINDI_NAMES
            self.subjects = self.HINDI_SUBJECTS
            self.headers = self.HINDI_HEADERS
        else:
            self.items = self.ENGLISH_ITEMS
            self.categories = self.ENGLISH_CATEGORIES
            self.names = self.ENGLISH_NAMES
            self.subjects = self.ENGLISH_SUBJECTS
            self.headers = self.ENGLISH_HEADERS

    def generate_table(
        self,
        template: Optional[str] = None,
        num_rows: int = 5,
        num_cols: int = 4,
    ) -> Table:
        if template and template in self.TEMPLATES:
            return self._generate_from_template(template, num_rows)
        return self._generate_random_table(num_rows, num_cols)

    def _generate_from_template(self, template: str, num_rows: int) -> Table:
        config = self.TEMPLATES[template]
        row_gen = getattr(self, config["row_generator"])
        headers = self.headers.get(template, config["headers"])

        cells = []
        header_row = [TableCell(text=h, is_header=True) for h in headers]
        cells.append(header_row)

        for _ in range(num_rows):
            row_data = row_gen()
            cells.append([TableCell(text=str(d)) for d in row_data])

        return Table(cells=cells, style=self._random_style())

    def _generate_invoice_row(self) -> List[Any]:
        item = random.choice(self.items)
        qty = random.randint(1, 10)
        price = random.randint(1, 50) * 1000
        total = qty * price
        return [item, str(qty), f"{price:,}", f"{total:,}"]

    def _generate_schedule_row(self) -> List[Any]:
        hour = random.randint(9, 17)
        time = f"{hour:02d}:00"
        subjects = [random.choice(self.subjects) if random.random() > 0.3 else "-" for _ in range(5)]
        return [time] + subjects

    def _generate_product_row(self) -> List[Any]:
        product = random.choice(self.items)
        category = random.choice(self.categories)
        price = random.randint(1, 100) * 1000
        stock = random.randint(0, 500)
        return [product, category, f"{price:,}", str(stock)]

    def _generate_contact_row(self) -> List[Any]:
        name = random.choice(self.names)
        phone = f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        return [name, phone, email]

    def _generate_random_table(self, num_rows: int, num_cols: int) -> Table:
        cells = []

        header_row = [
            TableCell(text=f"Col {i+1}", is_header=True)
            for i in range(num_cols)
        ]
        cells.append(header_row)

        for row_idx in range(num_rows):
            row = []
            for col_idx in range(num_cols):
                if random.random() < 0.3:
                    text = str(random.randint(1, 1000))
                elif random.random() < 0.5:
                    text = random.choice(self.items)
                else:
                    text = random.choice(self.names)
                row.append(TableCell(text=text))
            cells.append(row)

        return Table(cells=cells, style=self._random_style())

    def _random_style(self) -> TableStyle:
        styles = [
            TableStyle(),
            TableStyle(
                header_bg_color=(70, 130, 180),
                header_text_color=(255, 255, 255),
                alt_row_color=(245, 245, 245),
            ),
            TableStyle(
                border_style=BorderStyle.SOLID,
                header_bg_color=(34, 139, 34),
                header_text_color=(255, 255, 255),
                cell_bg_color=(255, 255, 255),
            ),
            TableStyle(
                header_bg_color=(128, 0, 128),
                header_text_color=(255, 255, 255),
                alt_row_color=(250, 240, 255),
            ),
        ]
        return random.choice(styles)


class TableGenerator(BaseGenerator):

    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str = "ko",
    ):
        super().__init__(output_dir, font_dir, lang)
        self.data_generator = TableDataGenerator(lang)

    def generate(
        self,
        num_images: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        self.template = kwargs.get("template")
        self.row_range = kwargs.get("row_range", (3, 8))
        self.col_range = kwargs.get("col_range", (3, 6))
        self.font_size_range = kwargs.get("font_size_range", (12, 18))

        templates = list(TableDataGenerator.TEMPLATES.keys()) if self.template is None else [self.template]
        self.templates = templates

        metadata = []
        for idx in tqdm(range(num_images), desc="Generating table images"):
            image, meta = self.generate_single()

            filename = f"table_{idx:05d}.png"
            self.save_image(image, filename)
            meta["file_name"] = str(self.output_dir / filename)

            metadata.append(meta)

        return metadata

    def generate_single(self, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        if not hasattr(self, "templates"):
             template = kwargs.get("template")
             self.templates = list(TableDataGenerator.TEMPLATES.keys()) if template is None else [template]
             self.row_range = kwargs.get("row_range", (3, 8))
             self.col_range = kwargs.get("col_range", (3, 6))
             self.font_size_range = kwargs.get("font_size_range", (12, 18))

        selected_template = random.choice(self.templates)
        num_rows = random.randint(*self.row_range)
        num_cols = random.randint(*self.col_range)
        font_size = random.randint(*self.font_size_range)

        table = self.data_generator.generate_table(
            template=selected_template,
            num_rows=num_rows,
            num_cols=num_cols,
        )

        font_path = random.choice(self.font_paths)
        renderer = TableRenderer(font_path, font_size)
        image = renderer.render(table)

        html_gt = table.to_html()
        json_gt = table.to_json()

        metadata = {
            "html": html_gt,
            "json": json_gt,
            "template": selected_template,
            "num_rows": table.num_rows,
            "num_cols": table.num_cols,
            "font_size": font_size,
        }
        return image, metadata
