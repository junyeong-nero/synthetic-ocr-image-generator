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

        for row_idx, row in enumerate(table.cells):
            x = 0
            col_idx = 0

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

                x += cell_width
                col_idx += cell.colspan

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

    KOREAN_ITEMS = ["사과", "바나나", "우유", "빵", "커피", "라면", "김치", "두부", "계란", "생수"]
    KOREAN_CATEGORIES = ["식품", "음료", "생활용품", "전자제품", "의류"]
    KOREAN_NAMES = ["김민수", "이영희", "박지영", "최동훈", "정수연", "강태호", "윤서연", "임재현"]
    KOREAN_SUBJECTS = ["국어", "수학", "영어", "과학", "사회", "체육", "음악", "미술"]

    ENGLISH_ITEMS = ["Apple", "Banana", "Milk", "Bread", "Coffee", "Pasta", "Cheese", "Eggs", "Water", "Juice"]
    ENGLISH_CATEGORIES = ["Food", "Beverage", "Household", "Electronics", "Clothing"]
    ENGLISH_NAMES = ["John Smith", "Jane Doe", "Mike Johnson", "Sarah Wilson", "Tom Brown"]
    ENGLISH_SUBJECTS = ["Math", "English", "Science", "History", "Art", "PE", "Music"]

    def __init__(self, lang: str = "ko"):
        self.lang = lang
        if lang == "ko":
            self.items = self.KOREAN_ITEMS
            self.categories = self.KOREAN_CATEGORIES
            self.names = self.KOREAN_NAMES
            self.subjects = self.KOREAN_SUBJECTS
        else:
            self.items = self.ENGLISH_ITEMS
            self.categories = self.ENGLISH_CATEGORIES
            self.names = self.ENGLISH_NAMES
            self.subjects = self.ENGLISH_SUBJECTS

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
        headers = config["headers"]
        row_gen = getattr(self, config["row_generator"])

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
        template: Optional[str] = None,
        row_range: Tuple[int, int] = (3, 8),
        col_range: Tuple[int, int] = (3, 6),
        font_size_range: Tuple[int, int] = (12, 18),
    ) -> List[Dict[str, Any]]:
        templates = list(TableDataGenerator.TEMPLATES.keys()) if template is None else [template]

        metadata = []
        for idx in tqdm(range(num_images), desc="Generating table images"):
            selected_template = random.choice(templates) if template is None else template
            num_rows = random.randint(*row_range)
            num_cols = random.randint(*col_range)
            font_size = random.randint(*font_size_range)

            table = self.data_generator.generate_table(
                template=selected_template,
                num_rows=num_rows,
                num_cols=num_cols,
            )

            font_path = random.choice(self.font_paths)
            renderer = TableRenderer(font_path, font_size)
            image = renderer.render(table)

            filename = f"table_{idx:05d}.png"
            self.save_image(image, filename)

            html_gt = table.to_html()

            metadata.append({
                "file_name": str(self.output_dir / filename),
                "html": html_gt,
                "template": selected_template,
                "num_rows": table.num_rows,
                "num_cols": table.num_cols,
                "font_size": font_size,
            })

        return metadata
