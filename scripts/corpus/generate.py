#!/usr/bin/env python3
"""
LLM-based Corpus Generation Script.

Generates diverse synthetic data for OCR image generation using LLM APIs.
Supports OpenAI and Anthropic APIs.

Usage:
    # Generate all categories for Korean
    uv run python scripts/corpus/generate.py --lang ko --count 1000

    # Generate specific category
    uv run python scripts/corpus/generate.py --lang ko --category product_names --count 500

    # Use Anthropic instead of OpenAI
    uv run python scripts/corpus/generate.py --lang ko --provider anthropic --count 1000

Environment variables:
    OPENAI_API_KEY: OpenAI API key
    ANTHROPIC_API_KEY: Anthropic API key
"""

import argparse
import asyncio
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from env_utils import load_env_file

load_env_file()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default output directory
CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "corpus"

LANGUAGE_LABELS: Dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mr": "Marathi",
    "nl": "Dutch",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

# Category definitions with prompts for each language
CATEGORIES = {
    "product_names": {
        "description": "Product/item names for receipts and invoices",
        "prompts": {
            "ko": """한국의 편의점, 마트, 카페에서 판매하는 상품명 {count}개를 생성해주세요.
다양한 카테고리를 포함해주세요: 음료, 과자, 라면, 도시락, 유제품, 생활용품, 커피, 빵 등.
실제로 존재할 법한 자연스러운 상품명을 만들어주세요.
각 상품명은 한 줄에 하나씩, 다른 설명 없이 상품명만 출력해주세요.""",
            "en": """Generate {count} product names sold in convenience stores, supermarkets, and cafes.
Include various categories: beverages, snacks, instant food, dairy, household items, coffee, bakery, etc.
Create realistic, natural-sounding product names.
Output only the product names, one per line, without any other explanation.""",
            "ja": """日本のコンビニ、スーパー、カフェで販売されている商品名を{count}個生成してください。
様々なカテゴリを含めてください：飲料、お菓子、インスタント食品、乳製品、日用品、コーヒー、パンなど。
実際に存在しそうな自然な商品名を作成してください。
商品名のみを1行に1つずつ、他の説明なしで出力してください。""",
            "hi": """भारत के सुविधा स्टोर, सुपरमार्केट और कैफे में बेचे जाने वाले {count} उत्पाद नाम बनाएं।
विभिन्न श्रेणियां शामिल करें: पेय पदार्थ, स्नैक्स, इंस्टेंट फूड, डेयरी, घरेलू सामान, कॉफी, बेकरी आदि।
वास्तविक, प्राकृतिक लगने वाले उत्पाद नाम बनाएं।
केवल उत्पाद नाम, प्रति पंक्ति एक, बिना किसी अन्य स्पष्टीकरण के आउटपुट करें।""",
        },
    },
    "store_names": {
        "description": "Store/shop names for receipts",
        "prompts": {
            "ko": """한국의 편의점, 카페, 음식점, 마트 등의 상호명 {count}개를 생성해주세요.
형식: [브랜드/업종] + [지역명] + [점]
예시: GS25 강남역점, 스타벅스 홍대입구점, 맥도날드 신촌점
다양한 지역과 업종을 포함해주세요.
각 상호명은 한 줄에 하나씩, 다른 설명 없이 상호명만 출력해주세요.""",
            "en": """Generate {count} store/shop names for convenience stores, cafes, restaurants, and supermarkets.
Format: [Brand/Business] + [Location]
Examples: Starbucks Downtown, McDonald's Main Street, 7-Eleven Central Station
Include various locations and business types.
Output only the store names, one per line, without any other explanation.""",
            "ja": """日本のコンビニ、カフェ、レストラン、スーパーなどの店名を{count}個生成してください。
形式：[ブランド/業種] + [地域名] + [店]
例：セブンイレブン渋谷店、スターバックス新宿西口店、マクドナルド池袋東口店
様々な地域と業種を含めてください。
店名のみを1行に1つずつ、他の説明なしで出力してください。""",
            "hi": """भारत के सुविधा स्टोर, कैफे, रेस्तरां और सुपरमार्केट के {count} स्टोर/दुकान के नाम बनाएं।
प्रारूप: [ब्रांड/व्यवसाय] + [स्थान]
उदाहरण: स्टारबक्स कनॉट प्लेस, मैकडॉनल्ड्स साकेत, बिग बाजार नोएडा
विभिन्न स्थानों और व्यवसाय प्रकारों को शामिल करें।
केवल स्टोर नाम, प्रति पंक्ति एक, बिना किसी अन्य स्पष्टीकरण के आउटपुट करें।""",
        },
    },
    "company_names": {
        "description": "Company/business names for invoices and documents",
        "prompts": {
            "ko": """한국 기업의 상호명/회사명 {count}개를 생성해주세요.
다양한 업종을 포함해주세요: IT, 제조, 유통, 금융, 서비스, 건설, 의료 등.
형식: 회사명 + (주), (주)회사명, 회사명 주식회사 등 다양하게.
실제로 존재할 법한 자연스러운 회사명을 만들어주세요.
각 회사명은 한 줄에 하나씩, 다른 설명 없이 회사명만 출력해주세요.""",
            "en": """Generate {count} company/business names.
Include various industries: IT, manufacturing, retail, finance, services, construction, healthcare, etc.
Format: Company Name Inc., Company Name LLC, Company Name Corporation, etc.
Create realistic, natural-sounding company names.
Output only the company names, one per line, without any other explanation.""",
            "ja": """日本の企業名/会社名を{count}個生成してください。
様々な業種を含めてください：IT、製造、小売、金融、サービス、建設、医療など。
形式：株式会社○○、○○株式会社、(株)○○など様々に。
実際に存在しそうな自然な会社名を作成してください。
会社名のみを1行に1つずつ、他の説明なしで出力してください。""",
            "hi": """{count} कंपनी/व्यवसाय नाम बनाएं।
विभिन्न उद्योगों को शामिल करें: आईटी, विनिर्माण, खुदरा, वित्त, सेवाएं, निर्माण, स्वास्थ्य सेवा आदि।
प्रारूप: कंपनी नाम प्राइवेट लिमिटेड, कंपनी नाम लिमिटेड, आदि।
वास्तविक, प्राकृतिक लगने वाले कंपनी नाम बनाएं।
केवल कंपनी नाम, प्रति पंक्ति एक, बिना किसी अन्य स्पष्टीकरण के आउटपुट करें।""",
        },
    },
    "person_names": {
        "description": "Person names for forms and documents",
        "prompts": {
            "ko": """한국인 이름 {count}개를 생성해주세요.
다양한 성씨와 이름을 포함해주세요.
남성과 여성 이름을 균형있게 포함해주세요.
실제로 존재할 법한 자연스러운 이름을 만들어주세요.
각 이름은 한 줄에 하나씩, 다른 설명 없이 이름만 출력해주세요.""",
            "en": """Generate {count} person names.
Include diverse first and last names from various backgrounds.
Include both male and female names in balance.
Create realistic, natural-sounding names.
Output only the names, one per line, without any other explanation.""",
            "ja": """日本人の名前を{count}個生成してください。
様々な姓と名を含めてください。
男性と女性の名前をバランスよく含めてください。
実際に存在しそうな自然な名前を作成してください。
名前のみを1行に1つずつ、他の説明なしで出力してください。""",
            "hi": """{count} व्यक्ति के नाम बनाएं।
विभिन्न पृष्ठभूमि से विविध प्रथम और अंतिम नाम शामिल करें।
पुरुष और महिला दोनों नामों को संतुलित रूप से शामिल करें।
वास्तविक, प्राकृतिक लगने वाले नाम बनाएं।
केवल नाम, प्रति पंक्ति एक, बिना किसी अन्य स्पष्टीकरण के आउटपुट करें।""",
        },
    },
    "addresses": {
        "description": "Street addresses for forms and documents",
        "prompts": {
            "ko": """한국의 실제 같은 주소 {count}개를 생성해주세요.
형식: 시/도 + 구/군 + 도로명 + 번지 (+ 상세주소)
예시: 서울특별시 강남구 테헤란로 152, 경기도 성남시 분당구 판교역로 235
다양한 지역을 포함해주세요.
각 주소는 한 줄에 하나씩, 다른 설명 없이 주소만 출력해주세요.""",
            "en": """Generate {count} realistic street addresses.
Format: Street Number + Street Name, City, State ZIP
Examples: 123 Main Street, New York, NY 10001
Include various locations across the country.
Output only the addresses, one per line, without any other explanation.""",
            "ja": """日本の実際のような住所を{count}個生成してください。
形式：都道府県 + 市区町村 + 町名 + 番地
例：東京都渋谷区神宮前1-2-3、大阪府大阪市北区梅田4-5-6
様々な地域を含めてください。
住所のみを1行に1つずつ、他の説明なしで出力してください。""",
            "hi": """{count} वास्तविक जैसे पते बनाएं।
प्रारूप: नंबर + सड़क का नाम, शहर, राज्य पिन कोड
उदाहरण: 123 महात्मा गांधी रोड, नई दिल्ली 110001
देश भर के विभिन्न स्थानों को शामिल करें।
केवल पते, प्रति पंक्ति एक, बिना किसी अन्य स्पष्टीकरण के आउटपुट करें।""",
        },
    },
    "departments": {
        "description": "Department names for business cards and forms",
        "prompts": {
            "ko": """기업의 부서명 {count}개를 생성해주세요.
다양한 부서를 포함해주세요: 영업, 마케팅, 개발, 인사, 재무, 기획, 연구, 생산 등.
실제 기업에서 사용하는 부서명을 만들어주세요.
각 부서명은 한 줄에 하나씩, 다른 설명 없이 부서명만 출력해주세요.""",
            "en": """Generate {count} department names for companies.
Include various departments: Sales, Marketing, Engineering, HR, Finance, Planning, R&D, Operations, etc.
Create department names used in real companies.
Output only the department names, one per line, without any other explanation.""",
            "ja": """企業の部署名を{count}個生成してください。
様々な部署を含めてください：営業、マーケティング、開発、人事、財務、企画、研究、生産など。
実際の企業で使用される部署名を作成してください。
部署名のみを1行に1つずつ、他の説明なしで出力してください。""",
            "hi": """कंपनियों के लिए {count} विभाग के नाम बनाएं।
विभिन्न विभागों को शामिल करें: बिक्री, मार्केटिंग, इंजीनियरिंग, एचआर, वित्त, योजना, आर एंड डी, संचालन आदि।
वास्तविक कंपनियों में उपयोग किए जाने वाले विभाग नाम बनाएं।
केवल विभाग नाम, प्रति पंक्ति एक, बिना किसी अन्य स्पष्टीकरण के आउटपुट करें।""",
        },
    },
    "positions": {
        "description": "Job titles/positions for business cards",
        "prompts": {
            "ko": """기업에서 사용하는 직책/직급명 {count}개를 생성해주세요.
다양한 직급을 포함해주세요: 사원, 대리, 과장, 차장, 부장, 이사, 상무, 전무, 부사장, 사장, CEO 등.
실제 기업에서 사용하는 직책명을 만들어주세요.
각 직책명은 한 줄에 하나씩, 다른 설명 없이 직책명만 출력해주세요.""",
            "en": """Generate {count} job titles/positions used in companies.
Include various levels: Associate, Senior, Lead, Manager, Director, VP, SVP, EVP, President, CEO, etc.
Create job titles used in real companies.
Output only the job titles, one per line, without any other explanation.""",
            "ja": """企業で使用される役職名を{count}個生成してください。
様々な役職を含めてください：社員、係長、課長、次長、部長、取締役、常務、専務、副社長、社長など。
実際の企業で使用される役職名を作成してください。
役職名のみを1行に1つずつ、他の説明なしで出力してください。""",
            "hi": """कंपनियों में उपयोग किए जाने वाले {count} पद/पदवी नाम बनाएं।
विभिन्न स्तरों को शामिल करें: एसोसिएट, सीनियर, लीड, मैनेजर, डायरेक्टर, वीपी, एसवीपी, ईवीपी, प्रेसिडेंट, सीईओ आदि।
वास्तविक कंपनियों में उपयोग किए जाने वाले पद नाम बनाएं।
केवल पद नाम, प्रति पंक्ति एक, बिना किसी अन्य स्पष्टीकरण के आउटपुट करें।""",
        },
    },
    "titles": {
        "description": "Document/article titles for markdown",
        "prompts": {
            "ko": """기술 문서, 블로그, README 등에 사용할 수 있는 제목 {count}개를 생성해주세요.
다양한 주제를 포함해주세요: 프로그래밍, 개발 도구, 프레임워크, 튜토리얼, 가이드 등.
실제 기술 문서에서 볼 수 있는 제목을 만들어주세요.
각 제목은 한 줄에 하나씩, 다른 설명 없이 제목만 출력해주세요.""",
            "en": """Generate {count} titles for technical documents, blogs, and READMEs.
Include various topics: programming, development tools, frameworks, tutorials, guides, etc.
Create titles that could be seen in real technical documents.
Output only the titles, one per line, without any other explanation.""",
            "ja": """技術文書、ブログ、READMEなどに使用できるタイトルを{count}個生成してください。
様々なトピックを含めてください：プログラミング、開発ツール、フレームワーク、チュートリアル、ガイドなど。
実際の技術文書で見られるタイトルを作成してください。
タイトルのみを1行に1つずつ、他の説明なしで出力してください。""",
            "hi": """तकनीकी दस्तावेज़ों, ब्लॉग और README के लिए {count} शीर्षक बनाएं।
विभिन्न विषयों को शामिल करें: प्रोग्रामिंग, विकास उपकरण, फ्रेमवर्क, ट्यूटोरियल, गाइड आदि।
वास्तविक तकनीकी दस्तावेज़ों में देखे जा सकने वाले शीर्षक बनाएं।
केवल शीर्षक, प्रति पंक्ति एक, बिना किसी अन्य स्पष्टीकरण के आउटपुट करें।""",
        },
    },
    "paragraphs": {
        "description": "Descriptive paragraphs for documents",
        "prompts": {
            "ko": """기술 문서, 소개글, 설명문에 사용할 수 있는 문단 {count}개를 생성해주세요.
각 문단은 2-3문장으로 구성해주세요.
다양한 주제를 포함해주세요: 제품 소개, 기능 설명, 사용법 안내 등.
실제 문서에서 볼 수 있는 자연스러운 문단을 만들어주세요.
각 문단은 빈 줄로 구분해주세요.""",
            "en": """Generate {count} paragraphs for technical documents, introductions, and descriptions.
Each paragraph should be 2-3 sentences.
Include various topics: product introductions, feature descriptions, usage guides, etc.
Create natural paragraphs that could be seen in real documents.
Separate each paragraph with a blank line.""",
            "ja": """技術文書、紹介文、説明文に使用できる段落を{count}個生成してください。
各段落は2-3文で構成してください。
様々なトピックを含めてください：製品紹介、機能説明、使用方法案内など。
実際の文書で見られる自然な段落を作成してください。
各段落は空行で区切ってください。""",
            "hi": """तकनीकी दस्तावेज़ों, परिचय और विवरण के लिए {count} पैराग्राफ बनाएं।
प्रत्येक पैराग्राफ 2-3 वाक्यों का होना चाहिए।
विभिन्न विषयों को शामिल करें: उत्पाद परिचय, सुविधा विवरण, उपयोग गाइड आदि।
वास्तविक दस्तावेज़ों में देखे जा सकने वाले प्राकृतिक पैराग्राफ बनाएं।
प्रत्येक पैराग्राफ को एक रिक्त पंक्ति से अलग करें।""",
        },
    },
    "features": {
        "description": "Feature descriptions for README/docs",
        "prompts": {
            "ko": """소프트웨어/서비스의 기능 설명 {count}개를 생성해주세요.
짧고 간결하게 (3-8 단어) 작성해주세요.
예시: "빠른 처리 속도", "간편한 설치", "다양한 플러그인 지원"
실제 제품에서 볼 수 있는 기능 설명을 만들어주세요.
각 기능은 한 줄에 하나씩, 다른 설명 없이 기능명만 출력해주세요.""",
            "en": """Generate {count} feature descriptions for software/services.
Keep them short and concise (3-8 words).
Examples: "Fast processing speed", "Easy installation", "Multiple plugin support"
Create feature descriptions that could be seen in real products.
Output only the features, one per line, without any other explanation.""",
            "ja": """ソフトウェア/サービスの機能説明を{count}個生成してください。
短く簡潔に（3-8語）作成してください。
例：「高速な処理速度」、「簡単なインストール」、「豊富なプラグインサポート」
実際の製品で見られる機能説明を作成してください。
機能のみを1行に1つずつ、他の説明なしで出力してください。""",
            "hi": """सॉफ्टवेयर/सेवाओं के लिए {count} सुविधा विवरण बनाएं।
उन्हें संक्षिप्त रखें (3-8 शब्द)।
उदाहरण: "तेज़ प्रोसेसिंग गति", "आसान इंस्टॉलेशन", "एकाधिक प्लगइन समर्थन"
वास्तविक उत्पादों में देखी जा सकने वाली सुविधा विवरण बनाएं।
केवल सुविधाएं, प्रति पंक्ति एक, बिना किसी अन्य स्पष्टीकरण के आउटपुट करें।""",
        },
    },
}


class LLMProvider:
    """Base class for LLM providers."""

    async def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini"):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.9,
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5"):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    async def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return ""

        text_chunks: List[str] = []
        for block in response.content:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str) and block_text:
                text_chunks.append(block_text)
        return "\n".join(text_chunks)


def get_provider(provider_name: str, model: Optional[str] = None) -> LLMProvider:
    """Get LLM provider instance."""
    if provider_name == "openai":
        return OpenAIProvider(model=model or "gpt-5-mini")
    elif provider_name == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-5")
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def parse_response(response: str, category: str) -> List[str]:
    """Parse LLM response into list of items."""
    lines = response.strip().split("\n")

    # For paragraphs, join by double newline
    if category == "paragraphs":
        paragraphs = []
        current = []
        for line in lines:
            line = line.strip()
            if line:
                current.append(line)
            elif current:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
        return paragraphs

    # For other categories, filter and clean
    items = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and numbered prefixes
        if not line:
            continue
        # Remove common prefixes
        if line[0].isdigit() and (". " in line[:4] or ") " in line[:4]):
            line = line.split(". ", 1)[-1].split(") ", 1)[-1]
        # Remove bullet points
        if line.startswith("- ") or line.startswith("• ") or line.startswith("* "):
            line = line[2:]
        line = line.strip()
        if line:
            items.append(line)

    return items


def resolve_language_label(lang: str, lang_name: Optional[str] = None) -> str:
    if lang_name:
        return lang_name

    normalized = lang.lower().replace("_", "-")
    base_code = normalized.split("-", 1)[0]
    return LANGUAGE_LABELS.get(base_code, lang)


def build_prompt(
    category_info: Dict[str, Any],
    category: str,
    lang: str,
    count: int,
    lang_name: Optional[str] = None,
) -> str:
    prompts = category_info["prompts"]
    typed_prompts = prompts if isinstance(prompts, dict) else {}

    if lang in typed_prompts:
        return str(typed_prompts[lang]).format(count=count)

    fallback_prompt = typed_prompts.get("en") or next(iter(typed_prompts.values()), "")
    language_label = resolve_language_label(lang, lang_name)

    return (
        "You are generating OCR corpus entries.\n"
        f"Category: {category}\n"
        f"Target language: {language_label} (code: {lang})\n"
        f"Generate exactly {count} unique items.\n"
        "Adapt names, addresses, and wording to the target language and locale naturally.\n"
        "Return plain text only, one item per line, with no numbering or commentary.\n\n"
        "Category guidance:\n"
        f"{fallback_prompt.format(count=count)}"
    )


async def generate_category(
    provider: LLMProvider,
    category: str,
    lang: str,
    count: int,
    batch_size: int = 100,
    lang_name: Optional[str] = None,
) -> List[str]:
    """Generate items for a category using LLM."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    category_info = CATEGORIES[category]

    all_items = []
    remaining = count

    while remaining > 0:
        batch = min(batch_size, remaining)
        prompt = build_prompt(category_info, category, lang, batch, lang_name)

        logger.info(f"Generating {batch} items for {category} ({lang})...")

        try:
            response = await provider.generate(prompt)
            items = parse_response(response, category)

            # Deduplicate within batch
            items = list(dict.fromkeys(items))
            all_items.extend(items)

            logger.info(f"  Got {len(items)} items (total: {len(all_items)})")
            remaining -= batch

        except Exception as e:
            logger.error(f"Error generating {category}: {e}")
            break

    # Final deduplication
    all_items = list(dict.fromkeys(all_items))
    return all_items[:count]


def save_corpus(items: List[str], category: str, lang: str, output_dir: Path):
    """Save generated items to corpus file."""
    lang_dir = output_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    output_file = lang_dir / f"{category}.txt"

    # Load existing items if file exists
    existing = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing = set(line.strip() for line in f if line.strip())

    # Merge and deduplicate
    all_items = list(existing | set(items))
    random.shuffle(all_items)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(item + "\n")

    logger.info(f"Saved {len(all_items)} items to {output_file}")
    return len(all_items)


async def main():
    parser = argparse.ArgumentParser(description="Generate corpus data using LLM")
    parser.add_argument(
        "--lang",
        type=str,
        default="ko",
        help="Language code to generate data for (for example: ko, en, ja, hi, fr, de, es)",
    )
    parser.add_argument(
        "--lang-name",
        type=str,
        default=None,
        help="Optional language name hint used for unsupported or custom language codes",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=list(CATEGORIES.keys()),
        help="Specific category to generate (default: all)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of items to generate per category",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: gpt-5-mini for OpenAI, claude-sonnet-4-5 for Anthropic)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CORPUS_DIR),
        help="Output directory for corpus files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for generation (items per API call)",
    )

    args = parser.parse_args()

    # Get provider
    try:
        provider = get_provider(args.provider, args.model)
    except ImportError as e:
        logger.error(str(e))
        return 1

    output_dir = Path(args.output_dir)
    categories = [args.category] if args.category else list(CATEGORIES.keys())

    total_generated = 0
    for category in categories:
        items = await generate_category(
            provider,
            category,
            args.lang,
            args.count,
            args.batch_size,
            args.lang_name,
        )

        if items:
            count = save_corpus(items, category, args.lang, output_dir)
            total_generated += count

    logger.info(f"Total items generated: {total_generated}")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
