"""
Data provider module that combines external corpus, Faker, and hardcoded data.

This module provides a unified interface for generating various types of data
(names, addresses, products, etc.) with the following priority:
1. External corpus files (LLM-generated, large-scale data)
2. Faker library (dynamic generation)
3. Curated hardcoded data (fallback)

For 100k+ image generation, use external corpus to minimize duplicates.
Generate corpus with: uv run main.py corpus generate
"""

import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from faker import Faker

from generator.faker_locales import base_lang_code, normalize_lang_code, resolve_faker_locale
from generator.language_data import ENGLISH_DATA, LANGUAGE_DATA

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "corpus"


class DataProvider:
    """
    Unified data provider that combines external corpus, Faker, and hardcoded data.

    Data source priority:
    1. External corpus files (for large-scale generation with minimal duplicates)
    2. Faker library (dynamic generation for names, addresses, etc.)
    3. Curated hardcoded data (fallback for domain-specific content)

    For 100k+ image generation, generate corpus first with:
        uv run main.py corpus generate --lang ko --count 10000
    """

    # Mapping from data type to corpus filename
    CORPUS_FILES = {
        "product_names": "product_names.txt",
        "store_names": "store_names.txt",
        "company_names": "company_names.txt",
        "person_names": "person_names.txt",
        "addresses": "addresses.txt",
        "departments": "departments.txt",
        "positions": "positions.txt",
        "titles": "titles.txt",
        "paragraphs": "paragraphs.txt",
        "features": "features.txt",
        "requirements": "requirements.txt",
        "install_commands": "install_commands.txt",
        "usage_commands": "usage_commands.txt",
        "config_lines": "config_lines.txt",
        "api_endpoints": "api_endpoints.txt",
    }

    def __init__(
        self,
        lang: str = "ko",
        mix_ratio: float = 0.7,
        seed: Optional[int] = None,
        corpus_dir: Optional[Path] = None,
        use_corpus: bool = True,
    ):
        """
        Initialize the data provider.

        Args:
            lang: Language code (e.g. ko, en, ja, hi, fr, de, es, zh-CN)
            mix_ratio: Probability of using Faker vs hardcoded (0.0 to 1.0)
                       Only used when corpus is not available
            seed: Random seed for reproducibility
            corpus_dir: Directory containing corpus files (default: data/corpus)
            use_corpus: Whether to use external corpus files when available
        """
        self.lang = lang
        self.normalized_lang = normalize_lang_code(lang)
        self.base_lang = base_lang_code(lang)
        self.mix_ratio = mix_ratio
        self.use_corpus = use_corpus

        self.corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR
        self._corpus_lang_candidates = []
        for candidate in [lang, self.normalized_lang, self.base_lang]:
            if candidate and candidate not in self._corpus_lang_candidates:
                self._corpus_lang_candidates.append(candidate)
        self._lang_corpus_dirs = [
            self.corpus_dir / candidate for candidate in self._corpus_lang_candidates
        ]

        locale = resolve_faker_locale(lang)
        self.faker = Faker(locale)

        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

        self._data = (
            LANGUAGE_DATA.get(self.normalized_lang)
            or LANGUAGE_DATA.get(self.base_lang)
            or ENGLISH_DATA
        )

        # Cache for loaded corpus data
        self._corpus_cache: Dict[str, List[str]] = {}
        self._corpus_indices: Dict[str, int] = {}  # Track position for sequential access

        # Load corpus files
        if use_corpus:
            self._load_all_corpus()

    def _load_corpus_file(self, data_type: str) -> List[str]:
        """Load a single corpus file."""
        if data_type not in self.CORPUS_FILES:
            return []

        items: List[str] = []
        seen = set()

        for lang_dir in self._lang_corpus_dirs:
            filepath = lang_dir / self.CORPUS_FILES[data_type]
            if not filepath.exists():
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        value = line.strip()
                        if not value or value in seen:
                            continue
                        seen.add(value)
                        items.append(value)
                logger.debug(f"Loaded {len(items)} items from {filepath}")
            except Exception as e:
                logger.warning(f"Failed to load corpus {filepath}: {e}")

        if items:
            random.shuffle(items)
        return items

    def _load_all_corpus(self):
        """Load all available corpus files."""
        for data_type in self.CORPUS_FILES:
            items = self._load_corpus_file(data_type)
            if items:
                self._corpus_cache[data_type] = items
                self._corpus_indices[data_type] = 0

        if self._corpus_cache:
            logger.info(
                f"Loaded corpus for {self.lang}: "
                f"{', '.join(f'{k}({len(v)})' for k, v in self._corpus_cache.items())}"
            )

    def _get_from_corpus(self, data_type: str) -> Optional[str]:
        """Get an item from corpus, cycling through to minimize duplicates."""
        if data_type not in self._corpus_cache:
            return None

        items = self._corpus_cache[data_type]
        if not items:
            return None

        # Get current index and advance
        idx = self._corpus_indices.get(data_type, 0)
        item = items[idx]

        # Advance index, wrap around when reaching end
        self._corpus_indices[data_type] = (idx + 1) % len(items)

        # Reshuffle when we've gone through all items
        if self._corpus_indices[data_type] == 0:
            random.shuffle(items)

        return item

    def _use_faker(self) -> bool:
        """Determine whether to use Faker based on mix_ratio."""
        return random.random() < self.mix_ratio

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        normalized = " ".join(str(text).split())
        if not normalized:
            return []

        parts = re.split(r"(?<=[.!?。！？])\s+", normalized)
        sentences = [part.strip() for part in parts if part.strip()]
        return sentences or [normalized]

    @staticmethod
    def _trim_sentence_fragment(text: str, max_words: int, max_chars: int) -> str:
        normalized = " ".join(str(text).split()).strip()
        if not normalized:
            return ""

        normalized = normalized.rstrip(".!?。！？")
        words = normalized.split()
        if words:
            normalized = " ".join(words[:max_words])
        if len(normalized) > max_chars:
            normalized = normalized[:max_chars].rstrip()
        return normalized.strip(" -_:,;")

    def _paragraph_header_candidates(self, count: int) -> List[str]:
        paragraphs = self._corpus_cache.get("paragraphs", [])
        if not paragraphs:
            return []

        candidates: List[str] = []
        seen: set[str] = set()

        for paragraph in paragraphs:
            for sentence in self._split_sentences(paragraph):
                words = self._trim_sentence_fragment(sentence, max_words=8, max_chars=64).split()
                if not words:
                    continue

                for start in range(0, len(words), 2):
                    fragment = " ".join(words[start : start + 2]).strip()
                    lowered = fragment.lower()
                    if not fragment or lowered in seen:
                        continue
                    seen.add(lowered)
                    candidates.append(fragment)
                    if len(candidates) >= count:
                        return candidates

        return candidates

    def has_corpus(self, data_type: str) -> bool:
        """Check if corpus is available for a data type."""
        return data_type in self._corpus_cache and len(self._corpus_cache[data_type]) > 0

    def corpus_size(self, data_type: str) -> int:
        """Get the size of corpus for a data type."""
        return len(self._corpus_cache.get(data_type, []))

    # ==================== Names ====================

    def name(self) -> str:
        """Generate a person's name."""
        # Try corpus first
        if corpus_item := self._get_from_corpus("person_names"):
            return corpus_item
        # Fallback to Faker
        return self.faker.name()

    def first_name(self) -> str:
        """Generate a first name."""
        return self.faker.first_name()

    def last_name(self) -> str:
        """Generate a last name."""
        return self.faker.last_name()

    def names(self, count: int = 1) -> List[str]:
        """Generate multiple names."""
        return [self.name() for _ in range(count)]

    # ==================== Addresses ====================

    def address(self) -> str:
        """Generate an address."""
        # Try corpus first
        if corpus_item := self._get_from_corpus("addresses"):
            return corpus_item
        # Fallback to Faker
        return self.faker.address().replace("\n", ", ")

    def city(self) -> str:
        """Generate a city name."""
        return self.faker.city()

    def street_address(self) -> str:
        """Generate a street address."""
        return self.faker.street_address()

    def postcode(self) -> str:
        """Generate a postal code."""
        return self.faker.postcode()

    # ==================== Companies ====================

    def company(self) -> str:
        """Generate a company name."""
        # Try corpus first
        if corpus_item := self._get_from_corpus("company_names"):
            return corpus_item
        # Fallback to Faker
        return self.faker.company()

    def company_suffix(self) -> str:
        """Generate a company suffix (Inc., LLC, etc.)."""
        return self.faker.company_suffix()

    # ==================== Contact Info ====================

    def phone_number(self) -> str:
        """Generate a phone number."""
        return self.faker.phone_number()

    def email(self) -> str:
        """Generate an email address."""
        return self.faker.email()

    # ==================== Dates and Times ====================

    def date(self, pattern: str = "%Y-%m-%d") -> str:
        """Generate a date string."""
        return self.faker.date(pattern=pattern)

    def time(self, pattern: str = "%H:%M") -> str:
        """Generate a time string."""
        return self.faker.time(pattern=pattern)

    def datetime(self) -> str:
        """Generate a datetime string."""
        return self.faker.date_time().strftime("%Y-%m-%d %H:%M:%S")

    # ==================== Domain-Specific (Hardcoded) ====================

    def item(self) -> str:
        """Get a random item/product name (grocery, food, etc.)."""
        return random.choice(self._data.items)

    def items(self, count: int = 1) -> List[str]:
        """Get multiple random items."""
        return random.choices(self._data.items, k=count)

    def category(self) -> str:
        """Get a random category."""
        return random.choice(self._data.categories)

    def categories(self, count: int = 1) -> List[str]:
        """Get multiple random categories."""
        return random.choices(self._data.categories, k=count)

    def subject(self) -> str:
        """Get a random subject (school subject)."""
        return random.choice(self._data.subjects)

    def subjects(self, count: int = 1) -> List[str]:
        """Get multiple random subjects."""
        return random.choices(self._data.subjects, k=count)

    def store_name(self) -> str:
        """Get a random store name."""
        if corpus_item := self._get_from_corpus("store_names"):
            return corpus_item
        return random.choice(self._data.store_names)

    def store_names(self, count: int = 1) -> List[str]:
        """Get multiple random store names."""
        return random.choices(self._data.store_names, k=count)

    def product_name(self) -> str:
        """Get a random product name."""
        if corpus_item := self._get_from_corpus("product_names"):
            return corpus_item
        if self.has_corpus("paragraphs"):
            product = self._trim_sentence_fragment(self.sentence(), max_words=4, max_chars=32)
            if product:
                return product
        return random.choice(self._data.product_names)

    def product_names(self, count: int = 1) -> List[str]:
        """Get multiple random product names."""
        return random.choices(self._data.product_names, k=count)

    def department(self) -> str:
        """Get a random department name."""
        if corpus_item := self._get_from_corpus("departments"):
            return corpus_item
        return random.choice(self._data.departments)

    def departments(self, count: int = 1) -> List[str]:
        """Get multiple random department names."""
        return random.choices(self._data.departments, k=count)

    def position(self) -> str:
        """Get a random job position/title."""
        if corpus_item := self._get_from_corpus("positions"):
            return corpus_item
        return random.choice(self._data.positions)

    def positions(self, count: int = 1) -> List[str]:
        """Get multiple random positions."""
        return random.choices(self._data.positions, k=count)

    def job_title(self) -> str:
        """Get a job title (combines Faker and hardcoded)."""
        if self._use_faker():
            return self.faker.job()
        return f"{self.position()} - {self.department()}"

    # ==================== Table Headers ====================

    def headers(self, template: str, count: int = 4) -> List[str]:
        """Get headers for a specific table template."""
        if self.has_corpus("paragraphs"):
            candidates = self._paragraph_header_candidates(max(count, 4))
            if len(candidates) >= count:
                return candidates[:count]

        default_headers = self._data.headers.get(template, [])
        if default_headers:
            return list(default_headers[:count])

        return [str(index + 1) for index in range(max(0, count))]

    # ==================== Document Content ====================

    def title(self) -> str:
        """Get a random document title."""
        if corpus_item := self._get_from_corpus("titles"):
            return corpus_item
        if self.has_corpus("paragraphs"):
            title = self._trim_sentence_fragment(self.sentence(), max_words=8, max_chars=72)
            if title:
                return title
        return random.choice(self._data.titles)

    def titles(self, count: int = 1) -> List[str]:
        """Get multiple random titles."""
        return random.choices(self._data.titles, k=count)

    def paragraph(self) -> str:
        """Get a random paragraph."""
        if corpus_item := self._get_from_corpus("paragraphs"):
            return corpus_item
        if self._use_faker():
            return self.faker.paragraph(nb_sentences=3)
        return random.choice(self._data.paragraphs)

    def paragraphs(self, count: int = 1) -> List[str]:
        """Get multiple random paragraphs."""
        return [self.paragraph() for _ in range(count)]

    def feature(self) -> str:
        """Get a random feature description."""
        if corpus_item := self._get_from_corpus("features"):
            return corpus_item
        if self.has_corpus("paragraphs"):
            feature = self._trim_sentence_fragment(self.sentence(), max_words=6, max_chars=36)
            if feature:
                return feature
        return random.choice(self._data.features)

    def features(self, count: int = 1) -> List[str]:
        """Get multiple random features."""
        if count <= len(self._data.features):
            return random.sample(self._data.features, count)
        return random.choices(self._data.features, k=count)

    def requirement_line(self) -> str:
        if corpus_item := self._get_from_corpus("requirements"):
            return corpus_item

        runtime = random.choice([
            "Python",
            "Node.js",
            "PostgreSQL",
            "Redis",
            "Docker",
            "Kubernetes",
            "Terraform",
            "OpenSSL",
        ])
        version_major = random.randint(1, 3) if runtime in {"Terraform", "OpenSSL"} else random.randint(3, 20)
        version_minor = random.randint(0, 12)
        return f"{runtime} >= {version_major}.{version_minor}"

    def install_command(self, package_name: Optional[str] = None) -> str:
        if corpus_item := self._get_from_corpus("install_commands"):
            return corpus_item

        if package_name is None:
            words = [w for w in self.title().lower().replace("_", "-").split() if w]
            package_name = "-".join(words[:2]) if words else "sample-app"

        tool = random.choice(["pip", "uv", "npm", "pnpm"])
        if tool == "pip":
            return f"pip install {package_name}"
        if tool == "uv":
            return f"uv add {package_name}"
        if tool == "pnpm":
            return f"pnpm add {package_name}"
        return f"npm install {package_name}"

    def usage_command(self, entrypoint: Optional[str] = None) -> str:
        if corpus_item := self._get_from_corpus("usage_commands"):
            return corpus_item

        command_target = entrypoint or random.choice([
            "main.py",
            "server.py",
            "app.py",
            "src/main.py",
            "index.js",
            "scripts/run.py",
        ])
        if command_target.endswith(".py"):
            return f"python {command_target}"
        if command_target.endswith(".js"):
            return f"node {command_target}"
        return command_target

    def config_line(self) -> str:
        if corpus_item := self._get_from_corpus("config_lines"):
            return corpus_item

        key = random.choice([
            "log_level",
            "timeout_ms",
            "max_workers",
            "retry_count",
            "api_base_url",
            "enable_metrics",
            "cache_ttl_sec",
            "batch_size",
        ])
        value_map = {
            "log_level": random.choice(["DEBUG", "INFO", "WARN", "ERROR"]),
            "timeout_ms": str(random.choice([1000, 3000, 5000, 10000, 30000])),
            "max_workers": str(random.randint(2, 32)),
            "retry_count": str(random.randint(1, 5)),
            "api_base_url": f"https://api.{self.faker.domain_name()}",
            "enable_metrics": random.choice(["true", "false"]),
            "cache_ttl_sec": str(random.choice([60, 300, 900, 3600])),
            "batch_size": str(random.choice([8, 16, 32, 64, 128])),
        }
        value = value_map.get(key, "true")
        return f"{key}: {value}"

    def api_endpoint(self) -> str:
        if corpus_item := self._get_from_corpus("api_endpoints"):
            return corpus_item

        resource = random.choice([
            "users",
            "projects",
            "documents",
            "invoices",
            "workspaces",
            "notifications",
            "metrics",
            "sessions",
            "tasks",
            "exports",
        ])
        return f"/api/v1/{resource}"

    def code_comment(self) -> str:
        """Get a random code comment."""
        return random.choice(self._data.code_comments)

    def code_comments(self, count: int = 1) -> List[str]:
        """Get multiple random code comments."""
        return random.choices(self._data.code_comments, k=count)

    # ==================== Currency ====================

    @property
    def currency(self) -> str:
        """Get the currency symbol for the language."""
        return self._data.currency

    @property
    def currency_format(self) -> str:
        """Get the currency format string."""
        return self._data.currency_format

    def format_currency(self, amount: float) -> str:
        """Format an amount as currency."""
        return self._data.currency_format.format(amount)

    # ==================== Numeric Data ====================

    def random_int(self, min_val: int = 0, max_val: int = 100) -> int:
        """Generate a random integer."""
        return random.randint(min_val, max_val)

    def random_price(self, min_val: int = 1000, max_val: int = 50000, step: int = 1000) -> int:
        """Generate a random price (in smallest currency unit)."""
        return random.randrange(min_val, max_val + 1, step)

    def quantity(self, min_val: int = 1, max_val: int = 10) -> int:
        """Generate a random quantity."""
        return random.randint(min_val, max_val)

    # ==================== Text Generation ====================

    def sentence(self) -> str:
        """Generate a random sentence."""
        if corpus_item := self._get_from_corpus("paragraphs"):
            sentences = self._split_sentences(corpus_item)
            if sentences:
                return random.choice(sentences)

        if self._use_faker():
            return self.faker.sentence()

        paragraph = random.choice(self._data.paragraphs)
        sentences = self._split_sentences(paragraph)
        if sentences:
            return random.choice(sentences)
        return paragraph

    def sentences(self, count: int = 3) -> List[str]:
        """Generate multiple sentences."""
        return [self.sentence() for _ in range(max(0, count))]

    def text(self, max_chars: int = 200) -> str:
        """Generate random text."""
        return self.faker.text(max_nb_chars=max_chars)

    def word(self) -> str:
        """Generate a random word."""
        return self.faker.word()

    def words(self, count: int = 5) -> List[str]:
        """Generate multiple random words."""
        return self.faker.words(nb=count)
