"""KIE (Key Information Extraction) Generator Module.

This module generates synthetic document images for KIE benchmarks,
following SROIE, CORD, and FUNSD dataset styles.

Supported document types:
- Receipt: Store receipts with items, prices, totals
- Invoice: Business invoices with line items
- Form: Key-value pair forms
- Business Card: Contact information cards
"""

import random
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from generator.base import BaseGenerator

logger = logging.getLogger(__name__)


class KIEDocumentType(Enum):
    """KIE document types."""
    RECEIPT = "receipt"
    INVOICE = "invoice"
    FORM = "form"
    BUSINESS_CARD = "business_card"


@dataclass
class KIEField:
    """A single key-value field in KIE document."""
    key: str
    value: str
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    category: str = "other"


@dataclass
class KIELineItem:
    """A line item (product/service) in receipt/invoice."""
    name: str
    quantity: int
    unit_price: int
    total_price: int
    bounding_box: Optional[Tuple[int, int, int, int]] = None


@dataclass
class KIEDocument:
    """KIE document with extracted fields."""
    doc_type: KIEDocumentType
    fields: List[KIEField]
    line_items: List[KIELineItem] = field(default_factory=list)
    raw_text: str = ""

    def to_ground_truth(self) -> Dict[str, Any]:
        """Convert to ground truth format for KIE evaluation."""
        entities = {}
        for f in self.fields:
            entities[f.key] = {
                "value": f.value,
                "category": f.category,
                "bounding_box": f.bounding_box,
            }

        items = [
            {
                "name": item.name,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "total_price": item.total_price,
                "bounding_box": item.bounding_box,
            }
            for item in self.line_items
        ]

        return {
            "document_type": self.doc_type.value,
            "entities": entities,
            "line_items": items,
            "raw_text": self.raw_text,
        }


@dataclass
class KIEStyle:
    """Styling options for KIE documents."""
    width: int = 400
    height: int = 600
    margin: int = 20
    line_spacing: int = 8
    
    background_color: Tuple[int, int, int] = (255, 255, 255)
    text_color: Tuple[int, int, int] = (0, 0, 0)
    header_color: Tuple[int, int, int] = (0, 0, 0)
    
    title_font_size: int = 18
    header_font_size: int = 14
    body_font_size: int = 12
    small_font_size: int = 10
    
    add_noise: bool = True
    add_blur: bool = False
    add_rotation: float = 0.0
    paper_texture: bool = False


class KIEDataGenerator:
    """Generates KIE document data for various languages."""

    # Korean data
    KOREAN_STORE_NAMES = [
        "GS25 강남점", "CU 서초역점", "세븐일레븐 역삼점", "이마트24 삼성점",
        "스타벅스 테헤란로점", "투썸플레이스 강남역점", "이디야커피 선릉점",
        "맥도날드 강남DT점", "버거킹 역삼점", "롯데리아 삼성역점",
        "올리브영 강남본점", "다이소 서초점", "미니스톱 역삼점",
    ]
    KOREAN_COMPANY_NAMES = [
        "삼성전자(주)", "(주)LG전자", "현대자동차(주)", "SK하이닉스(주)",
        "(주)네이버", "카카오(주)", "(주)쿠팡", "배달의민족(주)",
        "토스뱅크(주)", "(주)당근마켓",
    ]
    KOREAN_ADDRESSES = [
        "서울시 강남구 테헤란로 152", "서울시 서초구 서초대로 398",
        "서울시 송파구 올림픽로 300", "경기도 성남시 분당구 판교역로 235",
        "서울시 영등포구 여의대로 108", "서울시 중구 세종대로 110",
        "부산시 해운대구 센텀중앙로 79", "대구시 수성구 동대구로 489",
    ]
    KOREAN_PRODUCT_NAMES = [
        "아메리카노", "카페라떼", "녹차라떼", "바닐라라떼", "카라멜마키아또",
        "삼각김밥", "컵라면", "도시락", "샌드위치", "김밥",
        "우유", "요거트", "주스", "생수", "탄산수",
        "초콜릿", "과자", "빵", "아이스크림", "껌",
    ]
    KOREAN_NAMES = [
        "김민수", "이영희", "박지영", "최동훈", "정수연",
        "강태호", "윤서연", "임재현", "한미영", "오준서",
    ]
    KOREAN_DEPARTMENTS = ["영업부", "마케팅부", "개발부", "인사부", "재무부", "기획부"]
    KOREAN_POSITIONS = ["대리", "과장", "차장", "부장", "이사", "사원"]

    # English data
    ENGLISH_STORE_NAMES = [
        "Starbucks Coffee", "McDonald's", "Burger King", "Subway",
        "7-Eleven", "Walgreens", "CVS Pharmacy", "Target",
        "Walmart", "Costco", "Whole Foods", "Trader Joe's",
    ]
    ENGLISH_COMPANY_NAMES = [
        "Apple Inc.", "Google LLC", "Microsoft Corporation", "Amazon.com Inc.",
        "Meta Platforms Inc.", "Tesla Inc.", "NVIDIA Corporation",
        "Salesforce Inc.", "Adobe Inc.", "Netflix Inc.",
    ]
    ENGLISH_ADDRESSES = [
        "123 Main Street, New York, NY 10001",
        "456 Oak Avenue, Los Angeles, CA 90001",
        "789 Pine Road, Chicago, IL 60601",
        "321 Elm Street, Houston, TX 77001",
        "654 Maple Drive, Phoenix, AZ 85001",
        "987 Cedar Lane, Philadelphia, PA 19101",
    ]
    ENGLISH_PRODUCT_NAMES = [
        "Americano", "Latte", "Cappuccino", "Mocha", "Espresso",
        "Sandwich", "Salad", "Burger", "Pizza Slice", "Hot Dog",
        "Water", "Juice", "Soda", "Coffee", "Tea",
        "Chips", "Cookies", "Candy", "Gum", "Chocolate",
    ]
    ENGLISH_NAMES = [
        "John Smith", "Jane Doe", "Michael Johnson", "Emily Davis",
        "Robert Wilson", "Sarah Brown", "David Lee", "Jennifer Taylor",
    ]
    ENGLISH_DEPARTMENTS = ["Sales", "Marketing", "Engineering", "HR", "Finance", "Operations"]
    ENGLISH_POSITIONS = ["Manager", "Director", "VP", "Associate", "Senior", "Lead"]

    # Japanese data
    JAPANESE_STORE_NAMES = [
        "セブンイレブン渋谷店", "ローソン新宿店", "ファミリーマート池袋店", "ミニストップ品川店",
        "スターバックス表参道店", "ドトールコーヒー銀座店", "タリーズコーヒー六本木店",
        "マクドナルド渋谷センター街店", "モスバーガー新宿西口店", "吉野家秋葉原店",
        "マツモトキヨシ原宿店", "ダイソー渋谷店", "ドン・キホーテ新宿店",
    ]
    JAPANESE_COMPANY_NAMES = [
        "トヨタ自動車株式会社", "ソニーグループ株式会社", "任天堂株式会社", "パナソニック株式会社",
        "ソフトバンクグループ株式会社", "楽天グループ株式会社", "株式会社日立製作所",
        "本田技研工業株式会社", "株式会社ファーストリテイリング", "株式会社メルカリ",
    ]
    JAPANESE_ADDRESSES = [
        "東京都渋谷区神宮前1-2-3", "東京都新宿区西新宿4-5-6",
        "東京都港区六本木7-8-9", "大阪府大阪市北区梅田10-11-12",
        "神奈川県横浜市西区みなとみらい13-14-15", "愛知県名古屋市中区栄16-17-18",
        "福岡県福岡市博多区博多駅前19-20-21", "京都府京都市下京区四条通22-23-24",
    ]
    JAPANESE_PRODUCT_NAMES = [
        "アメリカーノ", "カフェラテ", "抹茶ラテ", "バニララテ", "キャラメルマキアート",
        "おにぎり", "カップ麺", "弁当", "サンドイッチ", "おでん",
        "牛乳", "ヨーグルト", "ジュース", "ミネラルウォーター", "炭酸水",
        "チョコレート", "お菓子", "パン", "アイスクリーム", "ガム",
    ]
    JAPANESE_NAMES = [
        "田中太郎", "鈴木花子", "山田一郎", "佐藤美咲", "伊藤健太",
        "渡辺和子", "高橋誠", "中村優子", "小林大輔", "加藤恵",
    ]
    JAPANESE_DEPARTMENTS = ["営業部", "マーケティング部", "開発部", "人事部", "財務部", "企画部"]
    JAPANESE_POSITIONS = ["係長", "課長", "次長", "部長", "取締役", "社員"]

    # Hindi data
    HINDI_STORE_NAMES = [
        "रिलायंस फ्रेश दिल्ली", "बिग बाजार मुंबई", "डी-मार्ट बैंगलोर", "स्पेंसर्स कोलकाता",
        "कैफे कॉफी डे कनॉट प्लेस", "बरिस्ता जुहू", "स्टारबक्स गुड़गांव",
        "मैकडॉनल्ड्स साकेत", "डोमिनोज पिज्जा अंधेरी", "केएफसी नोएडा",
        "अपोलो फार्मेसी चेन्नई", "मेडप्लस हैदराबाद", "नेटमेड्स पुणे",
    ]
    HINDI_COMPANY_NAMES = [
        "टाटा कंसल्टेंसी सर्विसेज लिमिटेड", "रिलायंस इंडस्ट्रीज लिमिटेड", "इंफोसिस लिमिटेड",
        "विप्रो लिमिटेड", "एचडीएफसी बैंक लिमिटेड", "आईसीआईसीआई बैंक लिमिटेड",
        "भारती एयरटेल लिमिटेड", "हिंदुस्तान यूनिलीवर लिमिटेड", "आईटीसी लिमिटेड", "महिंद्रा एंड महिंद्रा लिमिटेड",
    ]
    HINDI_ADDRESSES = [
        "123 महात्मा गांधी रोड, नई दिल्ली 110001", "456 जुहू बीच रोड, मुंबई 400049",
        "789 एमजी रोड, बैंगलोर 560001", "321 पार्क स्ट्रीट, कोलकाता 700016",
        "654 अन्ना सलाई, चेन्नई 600002", "987 बंजारा हिल्स, हैदराबाद 500034",
        "147 शिवाजीनगर, पुणे 411005", "258 सी जी रोड, अहमदाबाद 380009",
    ]
    HINDI_PRODUCT_NAMES = [
        "अमेरिकानो", "कैफे लाते", "ग्रीन टी लाते", "वनीला लाते", "कारमेल मैकियाटो",
        "समोसा", "कप नूडल्स", "थाली", "सैंडविच", "पराठा",
        "दूध", "दही", "जूस", "पानी", "लस्सी",
        "चॉकलेट", "बिस्कुट", "रोटी", "आइसक्रीम", "च्युइंग गम",
    ]
    HINDI_NAMES = [
        "राहुल शर्मा", "प्रिया गुप्ता", "अमित कुमार", "सुनीता देवी", "विकास सिंह",
        "अनीता वर्मा", "राजेश पटेल", "नेहा अग्रवाल", "संजय मिश्रा", "पूजा राय",
    ]
    HINDI_DEPARTMENTS = ["बिक्री विभाग", "मार्केटिंग विभाग", "विकास विभाग", "मानव संसाधन विभाग", "वित्त विभाग", "योजना विभाग"]
    HINDI_POSITIONS = ["सहायक प्रबंधक", "प्रबंधक", "वरिष्ठ प्रबंधक", "उप निदेशक", "निदेशक", "कार्यकारी"]

    def __init__(self, lang: str = "ko"):
        self.lang = lang
        if lang == "ko":
            self.store_names = self.KOREAN_STORE_NAMES
            self.company_names = self.KOREAN_COMPANY_NAMES
            self.addresses = self.KOREAN_ADDRESSES
            self.product_names = self.KOREAN_PRODUCT_NAMES
            self.names = self.KOREAN_NAMES
            self.departments = self.KOREAN_DEPARTMENTS
            self.positions = self.KOREAN_POSITIONS
            self.currency = "원"
            self.currency_format = "{:,}원"
        elif lang == "ja":
            self.store_names = self.JAPANESE_STORE_NAMES
            self.company_names = self.JAPANESE_COMPANY_NAMES
            self.addresses = self.JAPANESE_ADDRESSES
            self.product_names = self.JAPANESE_PRODUCT_NAMES
            self.names = self.JAPANESE_NAMES
            self.departments = self.JAPANESE_DEPARTMENTS
            self.positions = self.JAPANESE_POSITIONS
            self.currency = "円"
            self.currency_format = "{:,}円"
        elif lang == "hi":
            self.store_names = self.HINDI_STORE_NAMES
            self.company_names = self.HINDI_COMPANY_NAMES
            self.addresses = self.HINDI_ADDRESSES
            self.product_names = self.HINDI_PRODUCT_NAMES
            self.names = self.HINDI_NAMES
            self.departments = self.HINDI_DEPARTMENTS
            self.positions = self.HINDI_POSITIONS
            self.currency = "₹"
            self.currency_format = "₹{:,.2f}"
        else:
            self.store_names = self.ENGLISH_STORE_NAMES
            self.company_names = self.ENGLISH_COMPANY_NAMES
            self.addresses = self.ENGLISH_ADDRESSES
            self.product_names = self.ENGLISH_PRODUCT_NAMES
            self.names = self.ENGLISH_NAMES
            self.departments = self.ENGLISH_DEPARTMENTS
            self.positions = self.ENGLISH_POSITIONS
            self.currency = "$"
            self.currency_format = "${:,.2f}"

    def generate_document(
        self,
        doc_type: KIEDocumentType,
        style: Optional[KIEStyle] = None,
    ) -> KIEDocument:
        """Generate a KIE document of the specified type."""
        if style is None:
            style = self._random_style()

        generators = {
            KIEDocumentType.RECEIPT: self._generate_receipt,
            KIEDocumentType.INVOICE: self._generate_invoice,
            KIEDocumentType.FORM: self._generate_form,
            KIEDocumentType.BUSINESS_CARD: self._generate_business_card,
        }

        return generators[doc_type](style)

    def _generate_receipt(self, style: KIEStyle) -> KIEDocument:
        """Generate receipt document (SROIE/CORD style)."""
        fields = []

        # Store name
        store_name = random.choice(self.store_names)
        fields.append(KIEField(
            key="company",
            value=store_name,
            category="header",
        ))

        # Address
        address = random.choice(self.addresses)
        fields.append(KIEField(
            key="address",
            value=address,
            category="header",
        ))

        # Date and time
        date = self._random_date()
        time = self._random_time()
        fields.append(KIEField(key="date", value=date, category="header"))
        fields.append(KIEField(key="time", value=time, category="header"))

        # Receipt number
        receipt_no = f"R{random.randint(100000, 999999)}"
        fields.append(KIEField(
            key="receipt_number",
            value=receipt_no,
            category="header",
        ))

        # Line items
        line_items = []
        num_items = random.randint(2, 6)
        subtotal = 0

        for _ in range(num_items):
            product = random.choice(self.product_names)
            qty = random.randint(1, 3)
            if self.lang == "ko":
                unit_price = random.randint(1, 15) * 1000
            else:
                unit_price = random.randint(1, 20) * 100  # cents
            total = qty * unit_price
            subtotal += total

            line_items.append(KIELineItem(
                name=product,
                quantity=qty,
                unit_price=unit_price,
                total_price=total,
            ))

        # Tax
        if self.lang == "ko":
            tax = int(subtotal * 0.1)
        else:
            tax = int(subtotal * 0.08)

        total = subtotal + tax

        fields.append(KIEField(
            key="subtotal",
            value=self._format_currency(subtotal),
            category="total",
        ))
        fields.append(KIEField(
            key="tax",
            value=self._format_currency(tax),
            category="total",
        ))
        fields.append(KIEField(
            key="total",
            value=self._format_currency(total),
            category="total",
        ))

        # Payment method
        payment_methods = (
            ["현금", "카드", "삼성페이", "카카오페이", "네이버페이"]
            if self.lang == "ko"
            else ["Cash", "Credit Card", "Apple Pay", "Google Pay"]
        )
        fields.append(KIEField(
            key="payment_method",
            value=random.choice(payment_methods),
            category="payment",
        ))

        return KIEDocument(
            doc_type=KIEDocumentType.RECEIPT,
            fields=fields,
            line_items=line_items,
        )

    def _generate_invoice(self, style: KIEStyle) -> KIEDocument:
        """Generate invoice document."""
        fields = []

        # Company info
        company = random.choice(self.company_names)
        fields.append(KIEField(key="company", value=company, category="header"))
        fields.append(KIEField(
            key="company_address",
            value=random.choice(self.addresses),
            category="header",
        ))

        # Invoice number and date
        invoice_no = f"INV-{random.randint(2024, 2025)}-{random.randint(10000, 99999)}"
        fields.append(KIEField(
            key="invoice_number",
            value=invoice_no,
            category="header",
        ))
        fields.append(KIEField(
            key="invoice_date",
            value=self._random_date(),
            category="header",
        ))
        fields.append(KIEField(
            key="due_date",
            value=self._random_date(offset_days=30),
            category="header",
        ))

        # Bill to
        customer_name = random.choice(self.names)
        fields.append(KIEField(
            key="customer_name",
            value=customer_name,
            category="customer",
        ))
        fields.append(KIEField(
            key="customer_address",
            value=random.choice(self.addresses),
            category="customer",
        ))

        # Line items
        line_items = []
        num_items = random.randint(2, 5)
        subtotal = 0

        for _ in range(num_items):
            product = random.choice(self.product_names)
            qty = random.randint(1, 10)
            if self.lang == "ko":
                unit_price = random.randint(10, 500) * 1000
            else:
                unit_price = random.randint(10, 500) * 100
            total = qty * unit_price
            subtotal += total

            line_items.append(KIELineItem(
                name=product,
                quantity=qty,
                unit_price=unit_price,
                total_price=total,
            ))

        # Totals
        if self.lang == "ko":
            tax = int(subtotal * 0.1)
        else:
            tax = int(subtotal * 0.08)
        total = subtotal + tax

        fields.append(KIEField(
            key="subtotal",
            value=self._format_currency(subtotal),
            category="total",
        ))
        fields.append(KIEField(
            key="tax",
            value=self._format_currency(tax),
            category="total",
        ))
        fields.append(KIEField(
            key="total",
            value=self._format_currency(total),
            category="total",
        ))

        return KIEDocument(
            doc_type=KIEDocumentType.INVOICE,
            fields=fields,
            line_items=line_items,
        )

    def _generate_form(self, style: KIEStyle) -> KIEDocument:
        """Generate form document (FUNSD style)."""
        fields = []

        # Form title
        form_titles = (
            ["신청서", "등록서", "요청서", "확인서", "동의서"]
            if self.lang == "ko"
            else ["Application Form", "Registration Form", "Request Form", "Consent Form"]
        )
        fields.append(KIEField(
            key="form_title",
            value=random.choice(form_titles),
            category="header",
        ))

        # Form number
        form_no = f"F-{random.randint(2024, 2025)}-{random.randint(1000, 9999)}"
        fields.append(KIEField(key="form_number", value=form_no, category="header"))

        # Date
        fields.append(KIEField(
            key="date",
            value=self._random_date(),
            category="header",
        ))

        # Personal information
        name = random.choice(self.names)
        fields.append(KIEField(key="name", value=name, category="personal"))

        # Phone number
        if self.lang == "ko":
            phone = f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
        else:
            phone = f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
        fields.append(KIEField(key="phone", value=phone, category="personal"))

        # Email
        email_name = name.lower().replace(" ", ".").replace(".", "")
        if self.lang == "ko":
            email = f"{email_name}@example.co.kr"
        else:
            email = f"{email_name}@example.com"
        fields.append(KIEField(key="email", value=email, category="personal"))

        # Address
        fields.append(KIEField(
            key="address",
            value=random.choice(self.addresses),
            category="personal",
        ))

        # Department (if applicable)
        if random.random() > 0.5:
            fields.append(KIEField(
                key="department",
                value=random.choice(self.departments),
                category="organization",
            ))

        # Signature fields
        fields.append(KIEField(
            key="signature_date",
            value=self._random_date(),
            category="signature",
        ))

        return KIEDocument(
            doc_type=KIEDocumentType.FORM,
            fields=fields,
        )

    def _generate_business_card(self, style: KIEStyle) -> KIEDocument:
        """Generate business card document."""
        fields = []

        # Name
        name = random.choice(self.names)
        fields.append(KIEField(key="name", value=name, category="personal"))

        # Position and department
        position = random.choice(self.positions)
        department = random.choice(self.departments)
        fields.append(KIEField(key="position", value=position, category="organization"))
        fields.append(KIEField(key="department", value=department, category="organization"))

        # Company
        company = random.choice(self.company_names)
        fields.append(KIEField(key="company", value=company, category="organization"))

        # Phone
        if self.lang == "ko":
            phone = f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
            office_phone = f"02-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        else:
            phone = f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
            office_phone = f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
        fields.append(KIEField(key="mobile", value=phone, category="contact"))
        fields.append(KIEField(key="phone", value=office_phone, category="contact"))

        # Email
        email_name = name.lower().replace(" ", ".").replace(".", "")
        company_domain = company.lower().split()[0].replace(",", "").replace(".", "")
        if self.lang == "ko":
            email = f"{email_name}@{company_domain}.co.kr"
        else:
            email = f"{email_name}@{company_domain}.com"
        fields.append(KIEField(key="email", value=email, category="contact"))

        # Address
        fields.append(KIEField(
            key="address",
            value=random.choice(self.addresses),
            category="contact",
        ))

        # Website (optional)
        if random.random() > 0.5:
            website = f"www.{company_domain}.{'co.kr' if self.lang == 'ko' else 'com'}"
            fields.append(KIEField(key="website", value=website, category="contact"))

        return KIEDocument(
            doc_type=KIEDocumentType.BUSINESS_CARD,
            fields=fields,
        )

    def _random_date(self, offset_days: int = 0) -> str:
        """Generate random date string."""
        import datetime
        base = datetime.date(2024, random.randint(1, 12), random.randint(1, 28))
        if offset_days:
            base += datetime.timedelta(days=offset_days)
        return base.strftime("%Y-%m-%d")

    def _random_time(self) -> str:
        """Generate random time string."""
        hour = random.randint(8, 22)
        minute = random.randint(0, 59)
        return f"{hour:02d}:{minute:02d}"

    def _format_currency(self, amount: int) -> str:
        """Format currency based on language."""
        if self.lang == "ko":
            return f"{amount:,}원"
        else:
            return f"${amount / 100:,.2f}"

    def _random_style(self) -> KIEStyle:
        """Generate random document style."""
        styles = [
            KIEStyle(
                background_color=(255, 255, 255),
                add_noise=True,
            ),
            KIEStyle(
                background_color=(252, 252, 248),
                add_noise=True,
                paper_texture=True,
            ),
            KIEStyle(
                background_color=(255, 255, 250),
                add_noise=True,
                add_blur=True,
            ),
            KIEStyle(
                background_color=(248, 248, 255),
                add_noise=False,
            ),
        ]
        return random.choice(styles)


class KIERenderer:
    """Renders KIE documents to images with bounding box annotations."""

    def __init__(self, font_path: str, style: KIEStyle):
        self.style = style
        self.font_path = font_path

        try:
            self.title_font = ImageFont.truetype(font_path, style.title_font_size)
            self.header_font = ImageFont.truetype(font_path, style.header_font_size)
            self.body_font = ImageFont.truetype(font_path, style.body_font_size)
            self.small_font = ImageFont.truetype(font_path, style.small_font_size)
        except IOError:
            logger.warning(f"Font '{font_path}' not found. Using default.")
            self.title_font = ImageFont.load_default()
            self.header_font = self.title_font
            self.body_font = self.title_font
            self.small_font = self.title_font

    def render(self, document: KIEDocument) -> Tuple[Image.Image, str]:
        """Render document to image and return raw text."""
        renderers = {
            KIEDocumentType.RECEIPT: self._render_receipt,
            KIEDocumentType.INVOICE: self._render_invoice,
            KIEDocumentType.FORM: self._render_form,
            KIEDocumentType.BUSINESS_CARD: self._render_business_card,
        }

        img, raw_text = renderers[document.doc_type](document)
        document.raw_text = raw_text

        # Apply effects
        img = self._apply_effects(img)

        return img, raw_text

    def _render_receipt(self, document: KIEDocument) -> Tuple[Image.Image, str]:
        """Render receipt-style document."""
        style = self.style
        width = style.width
        height = style.height

        img = Image.new("RGB", (width, height), style.background_color)
        draw = ImageDraw.Draw(img)

        raw_lines = []
        y = style.margin
        center_x = width // 2

        # Get fields as dict for easy access
        fields_dict = {f.key: f for f in document.fields}

        # Store name (centered, large)
        if "company" in fields_dict:
            f = fields_dict["company"]
            bbox = draw.textbbox((0, 0), f.value, font=self.title_font)
            text_width = bbox[2] - bbox[0]
            x = center_x - text_width // 2
            draw.text((x, y), f.value, font=self.title_font, fill=style.header_color)
            f.bounding_box = (x, y, x + text_width, y + bbox[3] - bbox[1])
            raw_lines.append(f.value)
            y += bbox[3] - bbox[1] + style.line_spacing

        # Address (centered)
        if "address" in fields_dict:
            f = fields_dict["address"]
            bbox = draw.textbbox((0, 0), f.value, font=self.small_font)
            text_width = bbox[2] - bbox[0]
            x = center_x - text_width // 2
            draw.text((x, y), f.value, font=self.small_font, fill=style.text_color)
            f.bounding_box = (x, y, x + text_width, y + bbox[3] - bbox[1])
            raw_lines.append(f.value)
            y += bbox[3] - bbox[1] + style.line_spacing

        # Separator line
        y += 5
        draw.line([(style.margin, y), (width - style.margin, y)], fill=(200, 200, 200))
        y += 10

        # Date and time
        date_time_line = ""
        if "date" in fields_dict:
            f = fields_dict["date"]
            date_time_line += f.value
        if "time" in fields_dict:
            f = fields_dict["time"]
            date_time_line += f" {f.value}"
        if date_time_line:
            draw.text((style.margin, y), date_time_line, font=self.small_font, fill=style.text_color)
            raw_lines.append(date_time_line)
            y += self.small_font.size + style.line_spacing

        # Receipt number
        if "receipt_number" in fields_dict:
            f = fields_dict["receipt_number"]
            text = f"No: {f.value}"
            bbox = draw.textbbox((0, 0), text, font=self.small_font)
            draw.text((style.margin, y), text, font=self.small_font, fill=style.text_color)
            f.bounding_box = (style.margin, y, style.margin + bbox[2] - bbox[0], y + bbox[3] - bbox[1])
            raw_lines.append(text)
            y += bbox[3] - bbox[1] + style.line_spacing

        # Separator
        y += 5
        draw.line([(style.margin, y), (width - style.margin, y)], fill=(200, 200, 200))
        y += 10

        # Line items
        for item in document.line_items:
            # Item name and quantity
            item_text = f"{item.name} x{item.quantity}"
            price_text = self._format_price(item.total_price)

            bbox = draw.textbbox((0, 0), item_text, font=self.body_font)
            draw.text((style.margin, y), item_text, font=self.body_font, fill=style.text_color)

            price_bbox = draw.textbbox((0, 0), price_text, font=self.body_font)
            price_x = width - style.margin - (price_bbox[2] - price_bbox[0])
            draw.text((price_x, y), price_text, font=self.body_font, fill=style.text_color)

            item.bounding_box = (style.margin, y, width - style.margin, y + bbox[3] - bbox[1])
            raw_lines.append(f"{item_text}  {price_text}")
            y += bbox[3] - bbox[1] + style.line_spacing

        # Separator before totals
        y += 5
        draw.line([(style.margin, y), (width - style.margin, y)], fill=(200, 200, 200))
        y += 10

        # Totals
        for key in ["subtotal", "tax", "total"]:
            if key in fields_dict:
                f = fields_dict[key]
                label = {"subtotal": "Subtotal", "tax": "Tax", "total": "TOTAL"}
                if document.fields[0].value and any(
                    ord(c) >= 0xAC00 for c in document.fields[0].value
                ):
                    label = {"subtotal": "소계", "tax": "부가세", "total": "합계"}

                label_text = f"{label[key]}:"
                value_text = f.value

                font = self.header_font if key == "total" else self.body_font

                bbox = draw.textbbox((0, 0), label_text, font=font)
                draw.text((style.margin, y), label_text, font=font, fill=style.text_color)

                value_bbox = draw.textbbox((0, 0), value_text, font=font)
                value_x = width - style.margin - (value_bbox[2] - value_bbox[0])
                draw.text((value_x, y), value_text, font=font, fill=style.text_color)

                f.bounding_box = (value_x, y, width - style.margin, y + value_bbox[3] - value_bbox[1])
                raw_lines.append(f"{label_text} {value_text}")
                y += max(bbox[3] - bbox[1], value_bbox[3] - value_bbox[1]) + style.line_spacing

        # Payment method
        if "payment_method" in fields_dict:
            y += 5
            f = fields_dict["payment_method"]
            text = f"Payment: {f.value}" if not any(
                ord(c) >= 0xAC00 for c in f.value
            ) else f"결제: {f.value}"
            bbox = draw.textbbox((0, 0), text, font=self.small_font)
            draw.text((style.margin, y), text, font=self.small_font, fill=style.text_color)
            f.bounding_box = (style.margin, y, style.margin + bbox[2] - bbox[0], y + bbox[3] - bbox[1])
            raw_lines.append(text)

        # Crop to content
        img = self._crop_to_content(img, y + style.margin * 2)

        return img, "\n".join(raw_lines)

    def _render_invoice(self, document: KIEDocument) -> Tuple[Image.Image, str]:
        """Render invoice-style document."""
        style = self.style
        width = max(style.width, 500)
        height = max(style.height, 700)

        img = Image.new("RGB", (width, height), style.background_color)
        draw = ImageDraw.Draw(img)

        raw_lines = []
        y = style.margin
        fields_dict = {f.key: f for f in document.fields}

        # Company header
        if "company" in fields_dict:
            f = fields_dict["company"]
            bbox = draw.textbbox((0, 0), f.value, font=self.title_font)
            draw.text((style.margin, y), f.value, font=self.title_font, fill=style.header_color)
            f.bounding_box = (style.margin, y, style.margin + bbox[2] - bbox[0], y + bbox[3] - bbox[1])
            raw_lines.append(f.value)
            y += bbox[3] - bbox[1] + style.line_spacing

        if "company_address" in fields_dict:
            f = fields_dict["company_address"]
            bbox = draw.textbbox((0, 0), f.value, font=self.small_font)
            draw.text((style.margin, y), f.value, font=self.small_font, fill=style.text_color)
            f.bounding_box = (style.margin, y, style.margin + bbox[2] - bbox[0], y + bbox[3] - bbox[1])
            raw_lines.append(f.value)
            y += bbox[3] - bbox[1] + style.line_spacing * 2

        # Invoice title
        invoice_title = "INVOICE"
        bbox = draw.textbbox((0, 0), invoice_title, font=self.title_font)
        title_x = width - style.margin - (bbox[2] - bbox[0])
        draw.text((title_x, style.margin), invoice_title, font=self.title_font, fill=style.header_color)
        raw_lines.append(invoice_title)

        # Invoice details (right side)
        detail_y = style.margin + bbox[3] - bbox[1] + 10
        for key in ["invoice_number", "invoice_date", "due_date"]:
            if key in fields_dict:
                f = fields_dict[key]
                label = key.replace("_", " ").title() + ":"
                text = f"{label} {f.value}"
                bbox = draw.textbbox((0, 0), text, font=self.small_font)
                text_x = width - style.margin - (bbox[2] - bbox[0])
                draw.text((text_x, detail_y), text, font=self.small_font, fill=style.text_color)
                f.bounding_box = (text_x, detail_y, width - style.margin, detail_y + bbox[3] - bbox[1])
                raw_lines.append(text)
                detail_y += bbox[3] - bbox[1] + 5

        y = max(y, detail_y) + 20

        # Bill to section
        bill_to_label = "Bill To:" if self.style.background_color else "청구처:"
        draw.text((style.margin, y), bill_to_label, font=self.header_font, fill=style.text_color)
        raw_lines.append(bill_to_label)
        y += self.header_font.size + 5

        for key in ["customer_name", "customer_address"]:
            if key in fields_dict:
                f = fields_dict[key]
                bbox = draw.textbbox((0, 0), f.value, font=self.body_font)
                draw.text((style.margin + 10, y), f.value, font=self.body_font, fill=style.text_color)
                f.bounding_box = (style.margin + 10, y, style.margin + 10 + bbox[2] - bbox[0], y + bbox[3] - bbox[1])
                raw_lines.append(f.value)
                y += bbox[3] - bbox[1] + 5

        y += 20

        # Items table header
        col_widths = [200, 60, 100, 100]
        headers = ["Item", "Qty", "Unit Price", "Total"]
        x = style.margin
        for i, header in enumerate(headers):
            draw.text((x, y), header, font=self.header_font, fill=style.text_color)
            x += col_widths[i]
        raw_lines.append("  ".join(headers))
        y += self.header_font.size + 5

        # Header line
        draw.line([(style.margin, y), (width - style.margin, y)], fill=(100, 100, 100))
        y += 10

        # Items
        for item in document.line_items:
            x = style.margin
            values = [
                item.name,
                str(item.quantity),
                self._format_price(item.unit_price),
                self._format_price(item.total_price),
            ]
            for i, val in enumerate(values):
                bbox = draw.textbbox((0, 0), val, font=self.body_font)
                draw.text((x, y), val, font=self.body_font, fill=style.text_color)
                x += col_widths[i]

            item.bounding_box = (style.margin, y, width - style.margin, y + self.body_font.size)
            raw_lines.append("  ".join(values))
            y += self.body_font.size + style.line_spacing

        y += 10
        draw.line([(style.margin, y), (width - style.margin, y)], fill=(100, 100, 100))
        y += 15

        # Totals (right-aligned)
        for key in ["subtotal", "tax", "total"]:
            if key in fields_dict:
                f = fields_dict[key]
                label = key.title() + ":"
                font = self.header_font if key == "total" else self.body_font

                label_bbox = draw.textbbox((0, 0), label, font=font)
                value_bbox = draw.textbbox((0, 0), f.value, font=font)

                label_x = width - style.margin - 150
                value_x = width - style.margin - (value_bbox[2] - value_bbox[0])

                draw.text((label_x, y), label, font=font, fill=style.text_color)
                draw.text((value_x, y), f.value, font=font, fill=style.text_color)

                f.bounding_box = (value_x, y, width - style.margin, y + value_bbox[3] - value_bbox[1])
                raw_lines.append(f"{label} {f.value}")
                y += max(label_bbox[3] - label_bbox[1], value_bbox[3] - value_bbox[1]) + style.line_spacing

        img = self._crop_to_content(img, y + style.margin)

        return img, "\n".join(raw_lines)

    def _render_form(self, document: KIEDocument) -> Tuple[Image.Image, str]:
        """Render form-style document (FUNSD style)."""
        style = self.style
        width = max(style.width, 450)
        height = max(style.height, 600)

        img = Image.new("RGB", (width, height), style.background_color)
        draw = ImageDraw.Draw(img)

        raw_lines = []
        y = style.margin
        fields_dict = {f.key: f for f in document.fields}

        # Form title
        if "form_title" in fields_dict:
            f = fields_dict["form_title"]
            bbox = draw.textbbox((0, 0), f.value, font=self.title_font)
            center_x = (width - (bbox[2] - bbox[0])) // 2
            draw.text((center_x, y), f.value, font=self.title_font, fill=style.header_color)
            f.bounding_box = (center_x, y, center_x + bbox[2] - bbox[0], y + bbox[3] - bbox[1])
            raw_lines.append(f.value)
            y += bbox[3] - bbox[1] + style.line_spacing * 2

        # Form number and date
        info_line = ""
        if "form_number" in fields_dict:
            info_line += f"No: {fields_dict['form_number'].value}"
        if "date" in fields_dict:
            info_line += f"  Date: {fields_dict['date'].value}"
        if info_line:
            draw.text((style.margin, y), info_line, font=self.small_font, fill=style.text_color)
            raw_lines.append(info_line)
            y += self.small_font.size + style.line_spacing * 2

        # Separator
        draw.line([(style.margin, y), (width - style.margin, y)], fill=(200, 200, 200))
        y += 15

        # Form fields with boxes
        form_fields = ["name", "phone", "email", "address", "department"]
        labels = {
            "name": "Name" if self.style.background_color == (255, 255, 255) else "이름",
            "phone": "Phone" if self.style.background_color == (255, 255, 255) else "전화번호",
            "email": "Email" if self.style.background_color == (255, 255, 255) else "이메일",
            "address": "Address" if self.style.background_color == (255, 255, 255) else "주소",
            "department": "Department" if self.style.background_color == (255, 255, 255) else "부서",
        }

        # Check language from field values
        is_korean = any(
            any(ord(c) >= 0xAC00 for c in f.value)
            for f in document.fields
            if f.value
        )
        if is_korean:
            labels = {
                "name": "이름",
                "phone": "전화번호",
                "email": "이메일",
                "address": "주소",
                "department": "부서",
            }

        for field_key in form_fields:
            if field_key in fields_dict:
                f = fields_dict[field_key]
                label = labels.get(field_key, field_key.title())

                # Label
                label_text = f"{label}:"
                label_bbox = draw.textbbox((0, 0), label_text, font=self.body_font)
                draw.text((style.margin, y), label_text, font=self.body_font, fill=style.text_color)

                # Value box
                value_x = style.margin + 100
                box_width = width - value_x - style.margin
                box_height = self.body_font.size + 10

                # Draw box
                draw.rectangle(
                    [value_x, y - 2, value_x + box_width, y + box_height],
                    outline=(180, 180, 180),
                    width=1,
                )

                # Value text
                draw.text((value_x + 5, y + 2), f.value, font=self.body_font, fill=style.text_color)
                f.bounding_box = (value_x + 5, y + 2, value_x + box_width - 5, y + box_height - 2)

                raw_lines.append(f"{label_text} {f.value}")
                y += box_height + style.line_spacing

        # Signature area
        y += 30
        sig_label = "Signature:" if not is_korean else "서명:"
        draw.text((style.margin, y), sig_label, font=self.body_font, fill=style.text_color)
        raw_lines.append(sig_label)
        y += self.body_font.size + 10

        # Signature line
        draw.line([(style.margin, y + 30), (style.margin + 200, y + 30)], fill=(100, 100, 100))

        if "signature_date" in fields_dict:
            f = fields_dict["signature_date"]
            date_label = "Date:" if not is_korean else "날짜:"
            date_text = f"{date_label} {f.value}"
            draw.text((style.margin, y + 40), date_text, font=self.small_font, fill=style.text_color)
            raw_lines.append(date_text)
            y += 60

        img = self._crop_to_content(img, y + style.margin)

        return img, "\n".join(raw_lines)

    def _render_business_card(self, document: KIEDocument) -> Tuple[Image.Image, str]:
        """Render business card document."""
        style = self.style
        width = 400
        height = 250

        img = Image.new("RGB", (width, height), style.background_color)
        draw = ImageDraw.Draw(img)

        raw_lines = []
        fields_dict = {f.key: f for f in document.fields}

        # Company name (top)
        y = style.margin
        if "company" in fields_dict:
            f = fields_dict["company"]
            bbox = draw.textbbox((0, 0), f.value, font=self.header_font)
            draw.text((style.margin, y), f.value, font=self.header_font, fill=style.header_color)
            f.bounding_box = (style.margin, y, style.margin + bbox[2] - bbox[0], y + bbox[3] - bbox[1])
            raw_lines.append(f.value)
            y += bbox[3] - bbox[1] + style.line_spacing * 2

        # Name (large)
        if "name" in fields_dict:
            f = fields_dict["name"]
            bbox = draw.textbbox((0, 0), f.value, font=self.title_font)
            draw.text((style.margin, y), f.value, font=self.title_font, fill=style.text_color)
            f.bounding_box = (style.margin, y, style.margin + bbox[2] - bbox[0], y + bbox[3] - bbox[1])
            raw_lines.append(f.value)
            y += bbox[3] - bbox[1] + 5

        # Position and department
        title_parts = []
        if "position" in fields_dict:
            title_parts.append(fields_dict["position"].value)
        if "department" in fields_dict:
            title_parts.append(fields_dict["department"].value)
        if title_parts:
            title_text = " | ".join(title_parts)
            bbox = draw.textbbox((0, 0), title_text, font=self.body_font)
            draw.text((style.margin, y), title_text, font=self.body_font, fill=(100, 100, 100))
            raw_lines.append(title_text)
            y += bbox[3] - bbox[1] + style.line_spacing * 2

        # Contact info
        contact_fields = ["mobile", "phone", "email", "address", "website"]
        icons = {"mobile": "M:", "phone": "T:", "email": "E:", "address": "A:", "website": "W:"}

        for field_key in contact_fields:
            if field_key in fields_dict:
                f = fields_dict[field_key]
                icon = icons.get(field_key, "")
                text = f"{icon} {f.value}"
                bbox = draw.textbbox((0, 0), text, font=self.small_font)
                draw.text((style.margin, y), text, font=self.small_font, fill=style.text_color)
                f.bounding_box = (style.margin, y, style.margin + bbox[2] - bbox[0], y + bbox[3] - bbox[1])
                raw_lines.append(text)
                y += bbox[3] - bbox[1] + 3

        return img, "\n".join(raw_lines)

    def _format_price(self, amount: int) -> str:
        """Format price based on detected language."""
        # Check if Korean by looking at style
        # This is a simplified check
        if hasattr(self, '_is_korean') and self._is_korean:
            return f"{amount:,}원"
        else:
            return f"${amount / 100:,.2f}"

    def _crop_to_content(self, img: Image.Image, content_height: int) -> Image.Image:
        """Crop image to content height."""
        width = img.width
        new_height = min(img.height, max(content_height, 100))
        return img.crop((0, 0, width, new_height))

    def _apply_effects(self, img: Image.Image) -> Image.Image:
        """Apply visual effects to simulate real documents."""
        style = self.style

        # Rotation
        if style.add_rotation and style.add_rotation != 0:
            angle = random.uniform(-style.add_rotation, style.add_rotation)
            img = img.rotate(angle, expand=True, fillcolor=style.background_color)

        # Noise
        if style.add_noise:
            img = self._add_noise(img)

        # Blur
        if style.add_blur:
            blur_radius = random.uniform(0.3, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise to image."""
        width, height = img.size
        noise = Image.new("RGB", (width, height))
        noise_draw = ImageDraw.Draw(noise)

        num_dots = int(width * height * 0.002)
        for _ in range(num_dots):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            gray = random.randint(180, 255)
            noise_draw.point((x, y), fill=(gray, gray, gray))

        return Image.blend(img, noise, 0.03)


class KIEGenerator(BaseGenerator):
    """Main generator for KIE document images."""

    def __init__(
        self,
        output_dir: str,
        font_dir: str,
        lang: str = "ko",
    ):
        super().__init__(output_dir, font_dir, lang)
        self.data_generator = KIEDataGenerator(lang)

    def generate(
        self,
        num_images: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate KIE document images.

        Args:
            num_images: Number of images to generate.
            **kwargs: Additional parameters like doc_type, add_noise, add_blur, add_rotation.

        Returns:
            List of metadata dictionaries.
        """
        self.doc_type = kwargs.get("doc_type")
        self.add_noise = kwargs.get("add_noise", True)
        self.add_blur = kwargs.get("add_blur", False)
        self.add_rotation = kwargs.get("add_rotation", 0.0)

        if self.doc_type:
            try:
                selected_types = [KIEDocumentType(self.doc_type)]
            except ValueError:
                logger.warning(f"Unknown doc_type '{self.doc_type}', using all types")
                selected_types = list(KIEDocumentType)
        else:
            selected_types = list(KIEDocumentType)
        
        self.selected_types = selected_types

        metadata = []
        for idx in tqdm(range(num_images), desc="Generating KIE images"):
            image, meta = self.generate_single()

            # Save image
            filename = f"kie_{idx:05d}.png"
            self.save_image(image, filename)
            meta["file_name"] = str(self.output_dir / filename)

            metadata.append(meta)

        return metadata

    def generate_single(self, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        if not hasattr(self, "selected_types"):
             doc_type = kwargs.get("doc_type")
             if doc_type:
                try:
                    selected_types = [KIEDocumentType(doc_type)]
                except ValueError:
                    selected_types = list(KIEDocumentType)
             else:
                selected_types = list(KIEDocumentType)
             self.selected_types = selected_types
             self.add_noise = kwargs.get("add_noise", True)
             self.add_blur = kwargs.get("add_blur", False)
             self.add_rotation = kwargs.get("add_rotation", 0.0)

        current_type = random.choice(self.selected_types)

        # Generate style
        style = self.data_generator._random_style()
        style.add_noise = self.add_noise
        style.add_blur = self.add_blur
        style.add_rotation = self.add_rotation

        # Generate document data
        document = self.data_generator.generate_document(current_type, style)

        # Render document
        font_path = random.choice(self.font_paths)
        renderer = KIERenderer(font_path, style)

        # Set language hint for price formatting
        renderer._is_korean = self.lang == "ko"

        image, raw_text = renderer.render(document)

        # Build ground truth
        gt = document.to_ground_truth()

        metadata = {
            "format": "kie",
            "document_type": current_type.value,
            "ground_truth": gt,
            "entities": {f.key: f.value for f in document.fields},
            "raw_text": raw_text,
            "add_noise": self.add_noise,
            "add_blur": self.add_blur,
            "add_rotation": self.add_rotation,
        }
        return image, metadata

