import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from generator.data_provider import DataProvider
from generator.formula_generator import FormularGenerator
from generator.merge_orchestrator import MergeOrchestrator
from generator.table_generator import TableGenerator
from generator.template_catalog import TemplateSpec
from generator.text_generator import TextGenerator

logger = logging.getLogger(__name__)

DEFAULT_FORMULA_SOURCE_WEIGHTS: Dict[str, float] = {
    "dataset": 0.45,
    "random": 0.30,
    "synthetic": 0.25,
}
DEFAULT_BLUEPRINT_MAX_PARAGRAPH_CHARS = 220
DEFAULT_BLUEPRINT_MAX_LINE_CHARS = 72

_MATHTEXT_ALIAS_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\\ge(?![A-Za-z])"), r"\\geq"),
    (re.compile(r"\\le(?![A-Za-z])"), r"\\leq"),
)


def normalize_mathtext_commands(formula: str) -> str:
    normalized = formula
    for pattern, replacement in _MATHTEXT_ALIAS_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


_RAW_HARD_CODED_FORMULA_EXPRESSIONS: Tuple[str, ...] = (
    r"a^2 + b^2 = c^2",
    r"E = mc^2",
    r"(a+b)^2 = a^2 + 2ab + b^2",
    r"(a-b)^2 = a^2 - 2ab + b^2",
    r"a^3 - b^3 = (a-b)(a^2 + ab + b^2)",
    r"a^3 + b^3 = (a+b)(a^2 - ab + b^2)",
    r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
    r"\Delta = b^2 - 4ac",
    r"\frac{n(n+1)}{2}",
    r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}",
    r"\sum_{k=1}^{n} k^2 = \frac{n(n+1)(2n+1)}{6}",
    r"\sum_{k=1}^{n} k^3 = \left(\frac{n(n+1)}{2}\right)^2",
    r"\prod_{k=1}^{n} k = n!",
    r"\binom{n}{k} = \frac{n!}{k!(n-k)!}",
    r"(x+y)^n = \sum_{k=0}^{n} \binom{n}{k}x^{n-k}y^k",
    r"\sum_{k=0}^{n}\binom{n}{k} = 2^n",
    r"\sum_{k=0}^{n}(-1)^k\binom{n}{k} = 0",
    r"\gcd(a,b)\cdot\operatorname{lcm}(a,b)=ab",
    r"|x+y| \le |x| + |y|",
    r"||x|-|y|| \le |x-y|",
    r"\log_a(xy)=\log_a x + \log_a y",
    r"\log_a\left(\frac{x}{y}\right)=\log_a x - \log_a y",
    r"\log_a(x^r)=r\log_a x",
    r"a^{\log_a x}=x",
    r"\frac{1}{1-r}=\sum_{k=0}^{\infty}r^k,\ |r|<1",
    r"e^{i\theta}=\cos\theta+i\sin\theta",
    r"e^{i\pi}+1=0",
    r"|z|=\sqrt{z\bar{z}}",
    r"\operatorname{Re}(z)=\frac{z+\bar{z}}{2}",
    r"\operatorname{Im}(z)=\frac{z-\bar{z}}{2i}",
    r"\sin^2 x + \cos^2 x = 1",
    r"1+\tan^2 x = \sec^2 x",
    r"1+\cot^2 x = \csc^2 x",
    r"\sin(2x)=2\sin x \cos x",
    r"\cos(2x)=\cos^2 x - \sin^2 x",
    r"\sin(x\pm y)=\sin x\cos y \pm \cos x\sin y",
    r"\cos(x\pm y)=\cos x\cos y \mp \sin x\sin y",
    r"\tan(x\pm y)=\frac{\tan x \pm \tan y}{1 \mp \tan x\tan y}",
    r"\arcsin x + \arccos x = \frac{\pi}{2}",
    r"\sin x \approx x - \frac{x^3}{3!}",
    r"\cos x \approx 1 - \frac{x^2}{2!}",
    r"\tan x \approx x + \frac{x^3}{3}",
    r"\sum_{k=0}^{\infty}(-1)^k\frac{x^{2k+1}}{(2k+1)!}=\sin x",
    r"\sum_{k=0}^{\infty}(-1)^k\frac{x^{2k}}{(2k)!}=\cos x",
    r"\sum_{k=1}^{\infty}\frac{\sin(kx)}{k} = \frac{\pi-x}{2},\ 0<x<2\pi",
    r"\lim_{x\to 0}\frac{\sin x}{x}=1",
    r"\lim_{x\to 0}\frac{1-\cos x}{x^2}=\frac{1}{2}",
    r"\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^n=e",
    r"f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}",
    r"\frac{d}{dx}x^n = nx^{n-1}",
    r"\frac{d}{dx}e^x = e^x",
    r"\frac{d}{dx}\ln x = \frac{1}{x}",
    r"\frac{d}{dx}\sin x = \cos x",
    r"\frac{d}{dx}\cos x = -\sin x",
    r"\frac{d}{dx}\tan x = \sec^2 x",
    r"\frac{d}{dx}\arcsin x = \frac{1}{\sqrt{1-x^2}}",
    r"\frac{d}{dx}\arctan x = \frac{1}{1+x^2}",
    r"\frac{d}{dx}(uv)=u'v+uv'",
    r"\frac{d}{dx}\left(\frac{u}{v}\right)=\frac{u'v-uv'}{v^2}",
    r"\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)",
    r"\int x^n\,dx = \frac{x^{n+1}}{n+1}+C,\ n\ne -1",
    r"\int \frac{1}{x}\,dx = \ln|x| + C",
    r"\int e^x\,dx = e^x + C",
    r"\int \sin x\,dx = -\cos x + C",
    r"\int \cos x\,dx = \sin x + C",
    r"\int \sec^2 x\,dx = \tan x + C",
    r"\int \frac{1}{1+x^2}\,dx = \arctan x + C",
    r"\int \frac{1}{\sqrt{1-x^2}}\,dx = \arcsin x + C",
    r"\int_0^1 x^2\,dx = \frac{1}{3}",
    r"\int_0^\pi \sin x\,dx = 2",
    r"\int_0^{\infty} e^{-ax}\,dx = \frac{1}{a},\ a>0",
    r"\int_{-\infty}^{\infty}e^{-x^2}\,dx = \sqrt{\pi}",
    r"\int_0^{\infty} x^{s-1}e^{-x}\,dx = \Gamma(s)",
    r"\Gamma(n+1)=n!",
    r"B(p,q)=\int_0^1 t^{p-1}(1-t)^{q-1}\,dt",
    r"B(p,q)=\frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}",
    r"\int u\,dv = uv - \int v\,du",
    r"\int_a^b f'(x)\,dx = f(b)-f(a)",
    r"\frac{d}{dx}\int_a^x f(t)\,dt = f(x)",
    r"\sum_{n=0}^{\infty}\frac{x^n}{n!}=e^x",
    r"\ln(1+x)=\sum_{n=1}^{\infty}\frac{(-1)^{n+1}x^n}{n},\ |x|<1",
    r"\frac{1}{1-x}=\sum_{n=0}^{\infty}x^n,\ |x|<1",
    r"\arctan x = \sum_{n=0}^{\infty}(-1)^n\frac{x^{2n+1}}{2n+1},\ |x|\le 1",
    r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
    r"\nabla \cdot \mathbf{B} = 0",
    r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}",
    r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0\varepsilon_0\frac{\partial \mathbf{E}}{\partial t}",
    r"F = G\frac{m_1m_2}{r^2}",
    r"pV = nRT",
    r"\Delta G = \Delta H - T\Delta S",
    r"\lambda = \frac{h}{p}",
    r"E_n = -\frac{13.6\,\mathrm{eV}}{n^2}",
    r"\psi(x,t) = Ae^{i(kx-\omega t)}",
    r"i\hbar\frac{\partial}{\partial t}\Psi = \hat{H}\Psi",
    r"\hat{H}\psi = E\psi",
    r"\Delta x\,\Delta p \ge \frac{\hbar}{2}",
    r"\langle x \rangle = \int_{-\infty}^{\infty} x|\psi(x)|^2\,dx",
    r"P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}",
    r"\mathbb{E}[X] = \sum_x x\,p(x)",
    r"\mathrm{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2",
    r"\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]",
    r"\rho_{X,Y}=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}",
    r"\mathcal{N}(x\mid \mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
    r"\operatorname{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}",
    r"H(p)=-\sum_x p(x)\log p(x)",
    r"I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}",
    r"\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2",
    r"\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|",
    r"\hat{\beta}=(X^TX)^{-1}X^Ty",
    r"\sigma(z)=\frac{1}{1+e^{-z}}",
    r"\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}",
    r"\mathcal{L}_{\mathrm{CE}}=-\sum_i y_i\log \hat{y}_i",
    r"\mathcal{L}_{\mathrm{BCE}}=-(y\log p + (1-y)\log(1-p))",
    r"\mathcal{L}_{\operatorname{SFT}}=-\sum_t \log p_\theta(y_t\mid y_{<t},x)",
    r"\mathcal{L}_{\operatorname{DPO}}=-\log \sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)",
    r"\mathcal{L}_{\operatorname{PPO}}=\mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]",
    r"\mathcal{L}_{\operatorname{KD}}=\tau^2\operatorname{KL}(p_t^{(\tau)}\|p_s^{(\tau)})",
    r"\mathcal{L}_{\operatorname{InfoNCE}}=-\log\frac{\exp(q\cdot k^+/\tau)}{\sum_j \exp(q\cdot k_j/\tau)}",
    r"\operatorname{BLEU}=\operatorname{BP}\cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)",
    r"\mathrm{ROUGE\text{-}L}=\frac{(1+\beta^2)RP}{R+\beta^2P}",
    r"\mathrm{F1}=2\cdot\frac{\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}",
    r"\mathrm{IoU}=\frac{|A\cap B|}{|A\cup B|}",
    r"\operatorname{CRLB}(\hat{\theta}) \ge \frac{1}{nI(\theta)}",
    r"\mathrm{AUC}=\int_0^1 \mathrm{TPR}(\mathrm{FPR}^{-1}(u))\,du",
)

HARD_CODED_FORMULA_EXPRESSIONS: Tuple[str, ...] = tuple(
    normalize_mathtext_commands(formula)
    for formula in _RAW_HARD_CODED_FORMULA_EXPRESSIONS
)


class MarkdownDataGenerator:
    """Generates markdown content for various template types."""

    def __init__(self, lang: str = "ko", data_provider: Optional[DataProvider] = None):
        self.lang = lang
        self.data = data_provider or DataProvider(lang=lang)
        self._last_merge_order: List[str] = []
        self.formula_source_mode = "mixed"
        self.formula_source_weights: Dict[str, float] = dict(DEFAULT_FORMULA_SOURCE_WEIGHTS)
        self.formula_dataset_path: Optional[str] = None
        self._formula_dataset: List[str] = []

    @staticmethod
    def _normalize_source_mode(value: Any, fallback: str = "mixed") -> str:
        normalized = str(value or "").strip().lower()
        allowed = {"mixed", "dataset", "random", "synthetic"}
        return normalized if normalized in allowed else fallback

    @staticmethod
    def _normalize_source_weights(
        dataset_weight: Any,
        random_weight: Any,
        synthetic_weight: Any,
        defaults: Dict[str, float],
    ) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for key, raw_value in {
            "dataset": dataset_weight,
            "random": random_weight,
            "synthetic": synthetic_weight,
        }.items():
            default_value = defaults.get(key, 0.0)
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                parsed = default_value
            weights[key] = max(0.0, parsed)

        if sum(weights.values()) <= 0:
            return dict(defaults)
        return weights

    @staticmethod
    def _normalize_formula_text(raw_formula: str) -> str:
        text = str(raw_formula).strip()
        if not text:
            return ""

        if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
            text = text[2:-2].strip()
        elif text.startswith("$") and text.endswith("$") and len(text) >= 2:
            text = text[1:-1].strip()

        text = normalize_mathtext_commands(text)
        return " ".join(text.split())

    @classmethod
    def _extract_formula_candidate(cls, payload: Any) -> str:
        if isinstance(payload, str):
            return cls._normalize_formula_text(payload)
        if isinstance(payload, dict):
            for key in ("formula", "equation", "latex", "math", "content", "text"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    normalized = cls._normalize_formula_text(value)
                    if normalized:
                        return normalized
        return ""

    @classmethod
    def _load_formula_dataset_entries(cls, dataset_path: Optional[str]) -> List[str]:
        if not dataset_path:
            return []

        path = Path(dataset_path).expanduser()
        if not path.exists() or not path.is_file():
            logger.warning("Formula dataset file not found: %s", dataset_path)
            return []

        formulas: List[str] = []
        suffix = path.suffix.lower()

        def push_formula(candidate: Any) -> None:
            normalized = cls._extract_formula_candidate(candidate)
            if normalized:
                formulas.append(normalized)

        try:
            if suffix == ".json":
                with open(path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
                if isinstance(payload, list):
                    for item in payload:
                        push_formula(item)
                elif isinstance(payload, dict):
                    list_like = None
                    for key in ("formulas", "equations", "items", "records", "data"):
                        value = payload.get(key)
                        if isinstance(value, list):
                            list_like = value
                            break
                    if list_like is not None:
                        for item in list_like:
                            push_formula(item)
                    else:
                        push_formula(payload)
            elif suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as file:
                    for line in file:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            push_formula(json.loads(stripped))
                        except json.JSONDecodeError:
                            push_formula(stripped)
            elif suffix in {".csv", ".tsv"}:
                delimiter = "\t" if suffix == ".tsv" else ","
                with open(path, "r", encoding="utf-8", newline="") as file:
                    reader = csv.DictReader(file, delimiter=delimiter)
                    formula_key = None
                    if reader.fieldnames:
                        lowered = {name.lower(): name for name in reader.fieldnames if name}
                        for key in ("formula", "equation", "latex", "math", "content", "text"):
                            if key in lowered:
                                formula_key = lowered[key]
                                break
                        if formula_key is None:
                            formula_key = next((name for name in reader.fieldnames if name), None)

                    if formula_key:
                        for row in reader:
                            push_formula(row.get(formula_key, ""))
            else:
                with open(path, "r", encoding="utf-8") as file:
                    for line in file:
                        push_formula(line)
        except Exception as exc:
            logger.warning("Failed to load formula dataset '%s': %s", dataset_path, exc)
            return []

        deduplicated: List[str] = []
        seen: set[str] = set()
        for formula in formulas:
            if formula not in seen:
                seen.add(formula)
                deduplicated.append(formula)

        return deduplicated

    def configure_content_sources(
        self,
        *,
        formula_source_mode: str = "mixed",
        formula_dataset_path: Optional[str] = None,
        formula_dataset_weight: float = DEFAULT_FORMULA_SOURCE_WEIGHTS["dataset"],
        formula_random_weight: float = DEFAULT_FORMULA_SOURCE_WEIGHTS["random"],
        formula_synthetic_weight: float = DEFAULT_FORMULA_SOURCE_WEIGHTS["synthetic"],
    ) -> None:
        self.formula_source_mode = self._normalize_source_mode(formula_source_mode)
        self.formula_source_weights = self._normalize_source_weights(
            formula_dataset_weight,
            formula_random_weight,
            formula_synthetic_weight,
            DEFAULT_FORMULA_SOURCE_WEIGHTS,
        )

        normalized_formula_path = str(Path(formula_dataset_path).expanduser()) if formula_dataset_path else None

        if normalized_formula_path != self.formula_dataset_path:
            self.formula_dataset_path = normalized_formula_path
            self._formula_dataset = self._load_formula_dataset_entries(self.formula_dataset_path)

    def pop_merge_order(self) -> List[str]:
        merge_order = list(self._last_merge_order)
        self._last_merge_order = []
        return merge_order

    @staticmethod
    def _select_source(
        mode: str,
        weights: Dict[str, float],
        available_sources: List[str],
        fallback: str,
    ) -> str:
        if not available_sources:
            return fallback

        if mode != "mixed":
            if mode in available_sources:
                return mode
            if fallback in available_sources:
                return fallback
            return available_sources[0]

        candidates = [source for source in available_sources if weights.get(source, 0.0) > 0]
        if not candidates:
            return random.choice(available_sources)

        source_weights = [weights.get(source, 0.0) for source in candidates]
        return random.choices(candidates, weights=source_weights, k=1)[0]

    def generate_markdown(
        self,
        template_id: str = "default",
        template_spec: Optional[TemplateSpec] = None,
    ) -> str:
        _ = template_id
        self._last_merge_order = []
        section_blueprint = (
            template_spec.blueprint
            if template_spec and isinstance(template_spec.blueprint, dict)
            else {}
        )
        return self._generate_from_sections(section_blueprint)

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        normalized = " ".join(text.split())
        if max_chars <= 0 or len(normalized) <= max_chars:
            return normalized
        clipped = normalized[:max_chars].rstrip()
        split_at = clipped.rfind(" ")
        if split_at >= int(max_chars * 0.6):
            clipped = clipped[:split_at].rstrip()
        return clipped.rstrip(" ,;:") + "..."

    @staticmethod
    def _coerce_int_range(value: Any, default_min: int, default_max: int) -> Tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                lower = int(value[0])
                upper = int(value[1])
            except (TypeError, ValueError):
                return default_min, default_max
            if lower > upper:
                lower, upper = upper, lower
            return lower, upper

        if isinstance(value, int):
            return value, value

        return default_min, default_max

    @staticmethod
    def _coerce_positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _generate_formula_expression(self) -> str:
        formula_source = self._select_source(
            self.formula_source_mode,
            self.formula_source_weights,
            self._available_formula_sources(),
            fallback="random",
        )

        expression = ""
        if formula_source == "dataset" and self._formula_dataset:
            expression = random.choice(self._formula_dataset)
        elif formula_source == "synthetic":
            expression = self._build_synthetic_formula_expression()
        else:
            expression = self._build_random_formula_expression()

        expression = self._normalize_formula_text(expression)
        if not expression:
            expression = self._build_random_formula_expression()
        return expression

    def _generate_from_sections(self, blueprint: Dict[str, Any]) -> str:
        text_cfg_raw = blueprint.get("text")
        table_cfg_raw = blueprint.get("table")
        formula_cfg_raw = blueprint.get("formula")
        text_cfg: Dict[str, Any] = text_cfg_raw if isinstance(text_cfg_raw, dict) else {}
        table_cfg: Dict[str, Any] = table_cfg_raw if isinstance(table_cfg_raw, dict) else {}
        formula_cfg: Dict[str, Any] = formula_cfg_raw if isinstance(formula_cfg_raw, dict) else {}

        text_section_range = self._coerce_int_range(text_cfg.get("section_count"), 3, 5)
        table_section_range = self._coerce_int_range(table_cfg.get("section_count"), 1, 2)
        formula_section_range = self._coerce_int_range(formula_cfg.get("section_count"), 1, 2)
        text_max_line_chars = self._coerce_positive_int(
            text_cfg.get("max_line_chars"),
            DEFAULT_BLUEPRINT_MAX_LINE_CHARS,
        )

        row_value = table_cfg.get("rows", table_cfg.get("row_count"))
        col_value = table_cfg.get("columns", table_cfg.get("cols", table_cfg.get("column_count")))
        table_row_range = self._coerce_int_range(row_value, 2, 4)
        table_col_range = self._coerce_int_range(col_value, 3, 5)

        text_count = random.randint(*text_section_range)
        table_count = random.randint(*table_section_range)
        formula_count = random.randint(*formula_section_range)

        text_generator = TextGenerator(
            data=self.data,
            clip_text=self._clip_text,
            max_paragraph_chars=DEFAULT_BLUEPRINT_MAX_PARAGRAPH_CHARS,
            max_line_chars=text_max_line_chars,
        )
        table_generator = TableGenerator(data=self.data, clip_text=self._clip_text)
        formular_generator = FormularGenerator(
            data=self.data,
            clip_text=self._clip_text,
            formula_supplier=self._generate_formula_expression,
        )
        orchestrator = MergeOrchestrator(data=self.data, clip_text=self._clip_text)

        text_sections = text_generator.generate_sections(section_count=text_count)
        table_sections = table_generator.generate_sections(
            section_count=table_count,
            row_range=table_row_range,
            column_range=table_col_range,
        )
        formula_sections = formular_generator.generate_sections(section_count=formula_count)

        markdown_text, merge_order = orchestrator.merge(
            text_sections=text_sections,
            table_sections=table_sections,
            formula_sections=formula_sections,
        )
        self._last_merge_order = merge_order
        return markdown_text

    def _available_formula_sources(self) -> List[str]:
        sources = ["random", "synthetic"]
        if self._formula_dataset:
            sources.append("dataset")
        return sources

    @staticmethod
    def _build_hard_coded_formula_expression() -> str:
        return random.choice(HARD_CODED_FORMULA_EXPRESSIONS)

    @staticmethod
    def _wrap_grouped_expression(expression: str) -> str:
        text = expression.strip()
        if not text:
            return "1"
        if any(token in text for token in (" + ", " - ", r"\cdot", "=")):
            return rf"\left({text}\right)"
        return text

    def _build_formula_terminal(self) -> str:
        symbol = random.choice(["x", "y", "z", "t", "n", "k", r"\theta", r"\lambda"])
        if symbol in {"x", "y", "z"} and random.random() < 0.22:
            symbol = f"{symbol}_{random.randint(1, 4)}"

        if random.random() < 0.15:
            numerator = random.randint(1, 11)
            denominator = random.randint(2, 12)
            return rf"\frac{{{numerator}}}{{{denominator}}}"

        terminals = [symbol, str(random.randint(1, 12)), random.choice([r"\pi", "e", r"\alpha", r"\beta"])]
        return random.choice(terminals)

    def _build_grammar_formula_term(self, depth: int) -> str:
        if depth <= 0:
            return self._build_formula_terminal()

        production = random.choices(
            ["binary", "fraction", "power", "function", "root", "terminal"],
            weights=[0.34, 0.16, 0.16, 0.14, 0.08, 0.12],
            k=1,
        )[0]

        if production == "binary":
            left = self._build_grammar_formula_term(depth - 1)
            right = self._build_grammar_formula_term(depth - 1)
            operator = random.choice([" + ", " - ", r" \cdot "])
            return f"{left}{operator}{right}"

        if production == "fraction":
            numerator = self._build_grammar_formula_term(depth - 1)
            denominator = self._build_grammar_formula_term(max(0, depth - 2))
            if denominator.strip() in {"0", "{0}"}:
                denominator = str(random.randint(1, 9))
            return rf"\frac{{{numerator}}}{{{denominator}}}"

        if production == "power":
            base = self._wrap_grouped_expression(self._build_grammar_formula_term(depth - 1))
            exponent = random.choice([str(random.randint(2, 5)), "n", "k", "2m"])
            return rf"{base}^{{{exponent}}}"

        if production == "function":
            function = random.choice([r"\sin", r"\cos", r"\tan", r"\ln", r"\log"])
            argument = self._build_grammar_formula_term(depth - 1)
            if function == r"\log" and random.random() < 0.4:
                base = random.randint(2, 10)
                return rf"\log_{{{base}}}\left({argument}\right)"
            return rf"{function}\left({argument}\right)"

        if production == "root":
            radicand = self._build_grammar_formula_term(depth - 1)
            return rf"\sqrt{{{radicand}}}"

        return self._build_formula_terminal()

    def _build_grammar_formula_expression(self) -> str:
        depth = random.randint(2, 3)
        branch = random.choices(
            ["equation", "integral", "derivative", "summation", "limit", "probability", "norm"],
            weights=[0.26, 0.18, 0.16, 0.14, 0.10, 0.10, 0.06],
            k=1,
        )[0]

        if branch == "equation":
            left = self._build_grammar_formula_term(depth)
            right = self._build_grammar_formula_term(max(1, depth - 1))
            relation = random.choice(["=", r"\leq", r"\geq"])
            return f"{left} {relation} {right}"

        if branch == "integral":
            variable = random.choice(["x", "t"])
            integrand = self._build_grammar_formula_term(max(1, depth - 1))
            if random.random() < 0.45:
                lower = random.randint(0, 3)
                upper = random.randint(lower + 1, lower + random.randint(2, 8))
                return rf"\int_{{{lower}}}^{{{upper}}} {integrand}\, d{variable}"
            primitive = self._build_grammar_formula_term(max(1, depth - 1))
            return rf"\int {integrand}\, d{variable} = {primitive} + C"

        if branch == "derivative":
            variable = random.choice(["x", "t"])
            target = self._build_grammar_formula_term(max(1, depth - 1))
            inner = self._build_grammar_formula_term(max(1, depth - 1))
            if random.random() < 0.55:
                return rf"\frac{{d}}{{d{variable}}}\left({inner}\right) = {target}"
            return rf"\frac{{\partial}}{{\partial {variable}}}\left({inner}\right) = {target}"

        if branch == "summation":
            index = random.choice(["i", "k", "n"])
            upper = random.randint(4, 16)
            term = self._build_grammar_formula_term(max(1, depth - 1))
            if random.random() < 0.45:
                closed_form = self._build_grammar_formula_term(max(1, depth - 1))
                return rf"\sum_{{{index}=1}}^{{{upper}}} {term} = {closed_form}"
            return rf"\sum_{{{index}=0}}^{{\infty}} {term}"

        if branch == "limit":
            variable = random.choice(["x", "n", "t"])
            approaching = random.choice(["0", "1", "2", r"\infty"])
            expression = self._build_grammar_formula_term(max(1, depth - 1))
            return rf"\lim_{{{variable}\to {approaching}}} {expression}"

        if branch == "probability":
            event_a = random.choice(["A", "B", "C"])
            event_b_candidates = [token for token in ("A", "B", "C") if token != event_a]
            event_b = random.choice(event_b_candidates)
            if random.random() < 0.6:
                return rf"P({event_a}\mid {event_b}) = \frac{{P({event_b}\mid {event_a})P({event_a})}}{{P({event_b})}}"
            variable = random.choice(["x", "z"])
            mean = random.choice([r"\mu", "0"])
            sigma = random.choice([r"\sigma", "1"])
            return (
                rf"f_X({variable}) = \frac{{1}}{{{sigma}\sqrt{{2\pi}}}}"
                rf"\exp\left(-\frac{{({variable}-{mean})^2}}{{2{sigma}^2}}\right)"
            )

        vector_a = random.choice([r"\mathbf{a}", r"\mathbf{x}", r"\mathbf{u}"])
        vector_b = random.choice([r"\mathbf{b}", r"\mathbf{y}", r"\mathbf{v}"])
        return rf"\|{vector_a} + {vector_b}\|_2 \le \|{vector_a}\|_2 + \|{vector_b}\|_2"

    def _build_parametric_synthetic_formula_expression(self) -> str:
        variable = random.choice(["x", "y", "z", "n", "k", "t"])
        coeff_a = random.randint(2, 9)
        coeff_b = random.randint(1, 12)
        coeff_c = random.randint(1, 30)
        power = random.randint(2, 4)
        upper = random.randint(3, 12)
        base = random.randint(2, 9)

        expressions = [
            f"{coeff_a}{variable}^{power} + {coeff_b}{variable} + {coeff_c} = 0",
            f"\\frac{{d}}{{d{variable}}}({coeff_a}{variable}^{power}) = {coeff_a * power}{variable}^{max(1, power - 1)}",
            f"\\sum_{{i=1}}^{upper} i = \\frac{{{upper}({upper}+1)}}{{2}}",
            f"\\int_0^{upper} {variable}^{power} \\, d{variable}",
            f"\\log_{{{base}}}({base}^{variable}) = {variable}",
        ]
        return random.choice(expressions)

    def _build_random_formula_expression(self) -> str:
        variable = random.choice(["x", "y", "n", "t", "k"])
        if random.random() < 0.2:
            return f"f({variable}) = {random.randint(2, 9)}{variable} + {random.randint(1, 20)}"
        return self._build_hard_coded_formula_expression()

    def _build_synthetic_formula_expression(self) -> str:
        branch = random.choices(
            ["grammar", "parametric", "hardcoded"],
            weights=[0.55, 0.30, 0.15],
            k=1,
        )[0]

        if branch == "hardcoded":
            return self._build_hard_coded_formula_expression()
        if branch == "parametric":
            return self._build_parametric_synthetic_formula_expression()
        return self._build_grammar_formula_expression()
