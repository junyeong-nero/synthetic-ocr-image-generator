import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from generator.base import BaseGenerator
from generator.data_provider import DataProvider

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
    page_color: Tuple[int, int, int] = (247, 244, 236)
    page_margin: int = 40
    title_font_scale: float = 1.15
    add_page_noise: bool = True
    add_scan_blur: bool = True
    add_scan_lines: bool = True


@dataclass
class Table:
    cells: List[List[TableCell]]
    style: TableStyle = field(default_factory=TableStyle)
    cell_bounding_boxes: List[List[Tuple[int, int, int, int]]] = field(default_factory=list)
    title: str = ""

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

        table_width = sum(col_widths) + table.style.border_width
        table_height = sum(row_heights) + table.style.border_width

        table_img = Image.new("RGB", (table_width, table_height), table.style.cell_bg_color)
        table_draw = ImageDraw.Draw(table_img)

        self._draw_cells(table_draw, table, col_widths, row_heights)

        if table.style.border_style == BorderStyle.SOLID:
            self._draw_borders(table_draw, table, col_widths, row_heights)

        title_height = 0
        title_font = self.font
        if table.title:
            title_font_size = max(12, int(self.font_size * table.style.title_font_scale))
            try:
                title_font = ImageFont.truetype(self.font.path, title_font_size)
            except (IOError, AttributeError):
                title_font = self.font
            title_height = title_font_size + 12

        page_margin = table.style.page_margin
        page_width = table_width + page_margin * 2
        page_height = table_height + page_margin * 2 + title_height

        img = Image.new("RGB", (page_width, page_height), table.style.page_color)
        draw = ImageDraw.Draw(img)

        if table.style.add_page_noise:
            self._draw_page_noise(draw, page_width, page_height)

        table_x = page_margin
        table_y = page_margin + title_height

        if table.title:
            draw.text(
                (page_margin, page_margin // 2),
                table.title,
                font=title_font,
                fill=(55, 55, 55),
            )

        shadow = Image.new("RGBA", (table_width + 4, table_height + 4), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rectangle([2, 2, table_width + 1, table_height + 1], fill=(0, 0, 0, 30))
        img.paste(shadow, (table_x, table_y), shadow)
        img.paste(table_img, (table_x, table_y))

        self._offset_bounding_boxes(table, table_x, table_y)

        if table.style.add_scan_lines:
            self._draw_scan_lines(draw, page_width, page_height)

        if table.style.add_scan_blur:
            blur_radius = random.uniform(0.15, 0.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(random.uniform(0.95, 1.08))

        return img

    def _draw_page_noise(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        num_points = int(width * height * 0.0015)
        for _ in range(num_points):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            gray = random.randint(180, 235)
            draw.point((x, y), fill=(gray, gray, gray))

    def _draw_scan_lines(self, draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
        spacing = random.randint(3, 6)
        line_color = random.randint(215, 235)
        for y in range(0, height, spacing):
            draw.line([(0, y), (width, y)], fill=(line_color, line_color, line_color), width=1)

    def _offset_bounding_boxes(self, table: Table, dx: int, dy: int) -> None:
        shifted: List[List[Tuple[int, int, int, int]]] = []
        for row in table.cell_bounding_boxes:
            shifted_row = []
            for x1, y1, x2, y2 in row:
                shifted_row.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
            shifted.append(shifted_row)
        table.cell_bounding_boxes = shifted

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
                text_width = int(bbox[2] - bbox[0])
                text_height = int(bbox[3] - bbox[1])

                cell_width = int(text_width + padding * 2)
                cell_height = int(text_height + padding * 2)

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
    """Generates table data using DataProvider for variety."""

    TEMPLATES = {
        "invoice": {
            "row_generator": "_generate_invoice_row",
        },
        "schedule": {
            "row_generator": "_generate_schedule_row",
        },
        "product": {
            "row_generator": "_generate_product_row",
        },
        "contact": {
            "row_generator": "_generate_contact_row",
        },
    }

    def __init__(self, lang: str = "ko", data_provider: Optional[DataProvider] = None):
        self.lang = lang
        self.data = data_provider or DataProvider(lang=lang)

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
        headers = self.data.headers(template)

        cells = []
        header_row = [TableCell(text=h, is_header=True) for h in headers]
        cells.append(header_row)

        for _ in range(num_rows):
            row_data = row_gen()
            cells.append([TableCell(text=str(d)) for d in row_data])

        return Table(cells=cells, style=self._random_style(), title=self._title_for_template(template))

    def _generate_invoice_row(self) -> List[Any]:
        item = self.data.item()
        qty = self.data.quantity(1, 10)
        price = self.data.random_price(1000, 50000, 1000)
        total = qty * price
        return [item, str(qty), f"{price:,}", f"{total:,}"]

    def _generate_schedule_row(self) -> List[Any]:
        hour = random.randint(9, 17)
        time = f"{hour:02d}:00"
        subjects = [self.data.subject() if random.random() > 0.3 else "-" for _ in range(5)]
        return [time] + subjects

    def _generate_product_row(self) -> List[Any]:
        product = self.data.item()
        category = self.data.category()
        price = self.data.random_price(1000, 100000, 1000)
        stock = self.data.random_int(0, 500)
        return [product, category, f"{price:,}", str(stock)]

    def _generate_contact_row(self) -> List[Any]:
        name = self.data.name()
        phone = self.data.phone_number()
        email = self.data.email()
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
                    text = str(self.data.random_int(1, 1000))
                elif random.random() < 0.5:
                    text = self.data.item()
                else:
                    text = self.data.name()
                row.append(TableCell(text=text))
            cells.append(row)

        return Table(cells=cells, style=self._random_style(), title=self.data.title())

    def _title_for_template(self, template: str) -> str:
        if self.lang == "ko":
            titles = {
                "invoice": "거래 명세표",
                "schedule": "일정표",
                "product": "상품 목록",
                "contact": "연락처 목록",
            }
        else:
            titles = {
                "invoice": "Invoice Table",
                "schedule": "Schedule Table",
                "product": "Product List",
                "contact": "Contact List",
            }
        return titles.get(template, self.data.title())

    def _random_style(self) -> TableStyle:
        styles = [
            TableStyle(
                border_color=(80, 80, 80),
                header_bg_color=(226, 226, 219),
                cell_bg_color=(249, 248, 243),
                alt_row_color=(243, 242, 237),
                text_color=(35, 35, 35),
                header_text_color=(30, 30, 30),
                alignment=CellAlignment.LEFT,
                page_color=(247, 244, 236),
            ),
            TableStyle(
                border_color=(70, 70, 70),
                border_width=1,
                header_bg_color=(214, 214, 209),
                cell_bg_color=(247, 246, 241),
                alt_row_color=(240, 239, 233),
                text_color=(30, 30, 30),
                header_text_color=(25, 25, 25),
                alignment=CellAlignment.LEFT,
                page_color=(245, 242, 234),
            ),
            TableStyle(
                border_color=(90, 90, 90),
                border_width=2,
                header_bg_color=(220, 220, 214),
                cell_bg_color=(251, 250, 246),
                alt_row_color=(246, 245, 240),
                text_color=(40, 40, 40),
                header_text_color=(30, 30, 30),
                alignment=CellAlignment.LEFT,
                page_color=(248, 245, 238),
                add_scan_blur=False,
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
