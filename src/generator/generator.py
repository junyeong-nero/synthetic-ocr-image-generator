"""Markdown generator module for synthetic OCR markdown image generation.

This module now supports composable document generation:
- `TextGenerator`: emits text-only markdown sections
- `TableGenerator`: emits markdown table sections
- `FormularGenerator`: emits markdown formula sections
- `MergeOrchestrator`: merges and shuffles sections into one markdown document
"""

import logging
import random
import tempfile
import importlib
import re
import base64
import csv
import json
from io import BytesIO
from collections import Counter, deque
from difflib import SequenceMatcher
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from character_similarity import find_similar_chars
from generator.base import BaseGenerator
from generator.data_provider import DataProvider
from generator.formula_generator import FormularGenerator
from generator.merge_orchestrator import MergeOrchestrator
from generator.table_generator import TableGenerator
from generator.text_generator import TextGenerator
from utils import markdown_to_json_ast, read_json

logger = logging.getLogger(__name__)


DEFAULT_NOVELTY_WINDOW = 80
DEFAULT_NOVELTY_THRESHOLD = 0.95
DEFAULT_NOVELTY_MAX_ATTEMPTS = 4
DEFAULT_BLUEPRINT_MAX_PARAGRAPH_CHARS = 220
A4_MAX_WIDTH_PX = 2480
A4_MAX_HEIGHT_PX = 3508
MAX_RENDER_ASPECT_RATIO = 2.0

_MARKDOWN_IMAGE_PATTERN = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)$")
_MARKDOWN_FORMULA_PATTERN = re.compile(r"^\$\$\s*(?P<formula>.+?)\s*\$\$$")
_SIMPLE_SUPERSCRIPT_PATTERN = re.compile(r"\^(?P<atom>[A-Za-z0-9])")
_SIMPLE_SUBSCRIPT_PATTERN = re.compile(r"_(?P<atom>[A-Za-z0-9])")
_CHAINED_SUPERSCRIPT_PATTERN = re.compile(r"\^\{(?P<first>[^{}]+)\}\^\{(?P<second>[^{}]+)\}")
_CHAINED_SUBSCRIPT_PATTERN = re.compile(r"_\{(?P<first>[^{}]+)\}_\{(?P<second>[^{}]+)\}")
_CHAINED_SUPERSCRIPT_SIMPLE_PATTERN = re.compile(r"\^\{(?P<first>[^{}]+)\}\^(?P<second>[A-Za-z0-9])")
_CHAINED_SUBSCRIPT_SIMPLE_PATTERN = re.compile(r"_\{(?P<first>[^{}]+)\}_(?P<second>[A-Za-z0-9])")
_SUPERSUBSUPER_PATTERN = re.compile(
    r"\^\{(?P<sup>[^{}]+)\}_\{(?P<sub>[^{}]+)\}\^\{(?P<extra_sup>[^{}]+)\}"
)
_SUBSUPSUB_PATTERN = re.compile(
    r"_\{(?P<sub>[^{}]+)\}\^\{(?P<sup>[^{}]+)\}_\{(?P<extra_sub>[^{}]+)\}"
)
_FORMULA_IMAGE_CACHE: Dict[Tuple[str, int, Tuple[int, int, int]], Optional[Image.Image]] = {}
_LATEX_TO_IMAGE_RENDERER: Optional[Any] = None
_LATEX_TO_IMAGE_LOADED = False
DEFAULT_FORMULA_SOURCE_WEIGHTS: Dict[str, float] = {
    "dataset": 0.45,
    "random": 0.30,
    "synthetic": 0.25,
}
_MATHTEXT_ALIAS_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\\ge(?![A-Za-z])"), r"\\geq"),
    (re.compile(r"\\le(?![A-Za-z])"), r"\\leq"),
)


def _normalize_mathtext_commands(formula: str) -> str:
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
    r"f(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2+\cdots",
    r"R_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}",
    r"\nabla f = \left(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y},\frac{\partial f}{\partial z}\right)",
    r"\nabla\cdot \mathbf{F}=\frac{\partial F_x}{\partial x}+\frac{\partial F_y}{\partial y}+\frac{\partial F_z}{\partial z}",
    r"\nabla\times \mathbf{F}=\left(\frac{\partial F_z}{\partial y}-\frac{\partial F_y}{\partial z},\frac{\partial F_x}{\partial z}-\frac{\partial F_z}{\partial x},\frac{\partial F_y}{\partial x}-\frac{\partial F_x}{\partial y}\right)",
    r"\nabla\cdot(\nabla\times\mathbf{F})=0",
    r"\nabla\times(\nabla \phi)=0",
    r"y' + p(x)y = q(x)",
    r"y(x)=e^{-\int p(x)\,dx}\left(\int q(x)e^{\int p(x)\,dx}\,dx + C\right)",
    r"y'' + \omega^2 y = 0",
    r"y(t)=C_1\cos(\omega t)+C_2\sin(\omega t)",
    r"\frac{dy}{dx}=ky",
    r"y=Ce^{kx}",
    r"\frac{d^2x}{dt^2}+\frac{k}{m}x=0",
    r"\frac{\partial u}{\partial t}=\alpha \frac{\partial^2 u}{\partial x^2}",
    r"\frac{\partial^2 u}{\partial t^2}=c^2\frac{\partial^2 u}{\partial x^2}",
    r"i\hbar\frac{\partial \psi}{\partial t}=\hat{H}\psi",
    r"-\frac{\hbar^2}{2m}\nabla^2\psi + V\psi = E\psi",
    r"\mathcal{L}\{f(t)\}(s)=\int_0^\infty e^{-st}f(t)\,dt",
    r"\mathcal{L}\{f'(t)\}=sF(s)-f(0)",
    r"\mathcal{L}^{-1}\left\{\frac{1}{s-a}\right\}=e^{at}",
    r"\mathcal{F}\{f(x)\}(\omega)=\int_{-\infty}^{\infty}f(x)e^{-i\omega x}\,dx",
    r"f(x)=\frac{1}{2\pi}\int_{-\infty}^{\infty}\hat{f}(\omega)e^{i\omega x}\,d\omega",
    r"\int_{-\infty}^{\infty}|f(x)|^2\,dx=\frac{1}{2\pi}\int_{-\infty}^{\infty}|\hat{f}(\omega)|^2\,d\omega",
    r"X_k=\sum_{n=0}^{N-1}x_n e^{-i2\pi kn/N}",
    r"A\mathbf{v}=\lambda \mathbf{v}",
    r"\det(A-\lambda I)=0",
    r"A=Q\Lambda Q^{-1}",
    r"A=U\Sigma V^\top",
    r"\operatorname{rank}(A)+\operatorname{nullity}(A)=n",
    r"\det(AB)=\det(A)\det(B)",
    r"\det(A^\top)=\det(A)",
    r"(AB)^\top = B^\top A^\top",
    r"(A^{-1})^\top = (A^\top)^{-1}",
    r"\operatorname{tr}(AB)=\operatorname{tr}(BA)",
    r"\|Ax\|_2 \le \|A\|_2\|x\|_2",
    r"\langle x,y\rangle \le \|x\|_2\|y\|_2",
    r"\|x+y\|_2 \le \|x\|_2+\|y\|_2",
    r"\sum_{i=1}^{n}\lambda_i = \operatorname{tr}(A)",
    r"\prod_{i=1}^{n}\lambda_i = \det(A)",
    r"\mathbf{a}\cdot\mathbf{b}=\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta",
    r"\|\mathbf{a}\times\mathbf{b}\|=\|\mathbf{a}\|\|\mathbf{b}\|\sin\theta",
    r"\oint_{\partial S}\mathbf{F}\cdot d\mathbf{r}=\iint_{S}(\nabla\times\mathbf{F})\cdot d\mathbf{S}",
    r"\iiint_{V}\nabla\cdot\mathbf{F}\,dV=\iint_{\partial V}\mathbf{F}\cdot d\mathbf{S}",
    r"\oint_{C}\mathbf{E}\cdot d\mathbf{l}=-\frac{d}{dt}\iint_{S}\mathbf{B}\cdot d\mathbf{S}",
    r"\oint_{C}\mathbf{B}\cdot d\mathbf{l}=\mu_0 I_{\text{enc}}+\mu_0\epsilon_0\frac{d}{dt}\iint_S\mathbf{E}\cdot d\mathbf{S}",
    r"\nabla^2\phi = \frac{\partial^2\phi}{\partial x^2}+\frac{\partial^2\phi}{\partial y^2}+\frac{\partial^2\phi}{\partial z^2}",
    r"P(A|B)=\frac{P(B|A)P(A)}{P(B)}",
    r"P(A\cap B)=P(A)P(B)\ \text{(independent)}",
    r"P(A\cup B)=P(A)+P(B)-P(A\cap B)",
    r"\mathbb{E}[X]=\sum_x x\,p(x)",
    r"\mathbb{E}[X]=\int_{-\infty}^{\infty}x f_X(x)\,dx",
    r"\operatorname{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2",
    r"\operatorname{Cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]",
    r"\rho_{X,Y}=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}",
    r"X\sim\operatorname{Bin}(n,p),\ P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}",
    r"X\sim\operatorname{Poisson}(\lambda),\ P(X=k)=e^{-\lambda}\frac{\lambda^k}{k!}",
    r"f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
    r"Z=\frac{X-\mu}{\sigma}",
    r"\chi^2=\sum_{i=1}^{n}\frac{(O_i-E_i)^2}{E_i}",
    r"t=\frac{\bar{X}-\mu}{S/\sqrt{n}}",
    r"F=\frac{S_1^2}{S_2^2}",
    r"\hat{\theta}_{\text{MLE}}=\arg\max_{\theta}\prod_{i=1}^{n}p(x_i|\theta)",
    r"\ell(\theta)=\sum_{i=1}^{n}\log p(x_i|\theta)",
    r"I(\theta)=\mathbb{E}\left[\left(\frac{\partial}{\partial\theta}\log p(X|\theta)\right)^2\right]",
    r"\operatorname{CRLB}(\hat{\theta})\ge \frac{1}{nI(\theta)}",
    r"\operatorname{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}",
    r"H(X)=-\sum_x p(x)\log p(x)",
    r"I(X;Y)=H(X)-H(X|Y)",
    r"\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}",
    r"\sigma(x)=\frac{1}{1+e^{-x}}",
    r"J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2",
    r"\nabla_\theta J(\theta)=\frac{1}{m}X^\top(X\theta-y)",
    r"\theta_{t+1}=\theta_t-\eta\nabla_\theta J(\theta_t)",
    r"\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2",
    r"\operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|",
    r"R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}",
    r"\mathcal{L}_{\operatorname{SFT}}=-\sum_{t=1}^{T}\log p_{\theta}(y_t|y_{<t},x)",
    r"\mathcal{L}_{\operatorname{NLL}}=-\mathbb{E}_{(x,y)}\sum_{t=1}^{T}\log p_{\theta}(y_t|y_{<t},x)",
    r"\mathcal{L}_{\operatorname{CE}}=-\sum_{v=1}^{V}q(v)\log p_{\theta}(v)",
    r"q'(v)=(1-\epsilon)\delta_{v,y}+\frac{\epsilon}{V}",
    r"\mathcal{L}_{\operatorname{LS}}=-\sum_{v=1}^{V}q'(v)\log p_{\theta}(v)",
    r"\mathcal{L}_{\operatorname{KD}}=\tau^2\operatorname{KL}(p_T^{\tau}\|p_S^{\tau})",
    r"\mathcal{L}_{\operatorname{KL}}=\operatorname{KL}(\pi_{\theta}(\cdot|x)\|\pi_{\operatorname{ref}}(\cdot|x))",
    r"\mathcal{L}_{\operatorname{PPO}}=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]",
    r"r_t(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\operatorname{old}}}(a_t|s_t)}",
    r"\mathcal{L}_{\operatorname{RLHF}}=-\mathbb{E}_{y\sim\pi_{\theta}}[r(x,y)-\beta\log\frac{\pi_{\theta}(y|x)}{\pi_{\operatorname{ref}}(y|x)}]",
    r"\mathcal{L}_{\operatorname{DPO}}=-\mathbb{E}_{(x,y_w,y_l)}\log\sigma\left(\beta\left(\log\frac{\pi_{\theta}(y_w|x)}{\pi_{\operatorname{ref}}(y_w|x)}-\log\frac{\pi_{\theta}(y_l|x)}{\pi_{\operatorname{ref}}(y_l|x)}\right)\right)",
    r"\mathcal{L}_{\operatorname{IPO}}=-\mathbb{E}_{(x,y_w,y_l)}\log\sigma\left(\Delta_{\theta}(x,y_w,y_l)-\frac{1}{2\beta}\right)",
    r"\Delta_{\theta}(x,y_w,y_l)=\log\pi_{\theta}(y_w|x)-\log\pi_{\theta}(y_l|x)",
    r"\mathcal{L}_{\operatorname{ORPO}}=\mathcal{L}_{\operatorname{NLL}}+\lambda\log\sigma\left(\log\frac{\pi_{\theta}(y_w|x)}{\pi_{\theta}(y_l|x)}\right)",
    r"\mathcal{L}_{\operatorname{SimPO}}=-\mathbb{E}_{(x,y_w,y_l)}\log\sigma\left(\beta(\log\pi_{\theta}(y_w|x)-\log\pi_{\theta}(y_l|x))-\gamma\right)",
    r"\mathcal{L}_{\operatorname{InfoNCE}}=-\log\frac{\exp(\operatorname{sim}(z_i,z_i^+)/\tau)}{\sum_{j=1}^{N}\exp(\operatorname{sim}(z_i,z_j)/\tau)}",
    r"\mathcal{L}_{\operatorname{Triplet}}=\max(0,d(z,z^+)-d(z,z^-)+m)",
    r"\mathcal{L}_{\operatorname{MoE}}=\alpha N\sum_{e=1}^{E}f_eP_e",
    r"\operatorname{PPL}=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p_{\theta}(y_t|y_{<t},x)\right)",
)
HARD_CODED_FORMULA_EXPRESSIONS: Tuple[str, ...] = tuple(
    _normalize_mathtext_commands(formula)
    for formula in _RAW_HARD_CODED_FORMULA_EXPRESSIONS
)
def parse_markdown_image_line(line: str) -> Optional[Tuple[str, str]]:
    match = _MARKDOWN_IMAGE_PATTERN.match(line.strip())
    if not match:
        return None
    return match.group("alt").strip(), match.group("src").strip()


def parse_markdown_formula_line(line: str) -> Optional[str]:
    match = _MARKDOWN_FORMULA_PATTERN.match(line.strip())
    if not match:
        return None
    return match.group("formula").strip()


def _get_latex_to_image_renderer() -> Optional[Any]:
    global _LATEX_TO_IMAGE_RENDERER, _LATEX_TO_IMAGE_LOADED
    if _LATEX_TO_IMAGE_LOADED:
        return _LATEX_TO_IMAGE_RENDERER

    _LATEX_TO_IMAGE_LOADED = True
    try:
        latex_module = importlib.import_module("latex_to_image")
        renderer_cls = getattr(latex_module, "LaTeXToImg")
        _LATEX_TO_IMAGE_RENDERER = renderer_cls()
    except Exception:
        _LATEX_TO_IMAGE_RENDERER = None
    return _LATEX_TO_IMAGE_RENDERER


def _formula_array_to_rgba(
    formula_array: np.ndarray,
    text_color: Tuple[int, int, int],
) -> Optional[Image.Image]:
    if not isinstance(formula_array, np.ndarray) or formula_array.size == 0:
        return None

    image_data = np.asarray(formula_array)
    if image_data.ndim == 2:
        grayscale = image_data.astype(np.uint8)
    elif image_data.ndim == 3 and image_data.shape[2] >= 3:
        b_channel = image_data[..., 0].astype(np.float32)
        g_channel = image_data[..., 1].astype(np.float32)
        r_channel = image_data[..., 2].astype(np.float32)
        grayscale = np.clip(0.114 * b_channel + 0.587 * g_channel + 0.299 * r_channel, 0, 255).astype(np.uint8)
    else:
        return None

    alpha_black_on_white = (255 - grayscale).astype(np.uint8)
    alpha_white_on_black = grayscale.astype(np.uint8)
    alpha_black_on_white[alpha_black_on_white < 10] = 0
    alpha_white_on_black[alpha_white_on_black < 10] = 0

    def _border_nonzero_ratio(alpha: np.ndarray) -> float:
        border = np.concatenate([alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]])
        if border.size == 0:
            return 1.0
        return float(np.count_nonzero(border)) / float(border.size)

    def _nonzero_ratio(alpha: np.ndarray) -> float:
        total = alpha.size
        if total <= 0:
            return 1.0
        return float(np.count_nonzero(alpha)) / float(total)

    black_on_white_border = _border_nonzero_ratio(alpha_black_on_white)
    white_on_black_border = _border_nonzero_ratio(alpha_white_on_black)
    if white_on_black_border < black_on_white_border:
        alpha = alpha_white_on_black
    elif black_on_white_border < white_on_black_border:
        alpha = alpha_black_on_white
    else:
        alpha = (
            alpha_white_on_black
            if _nonzero_ratio(alpha_white_on_black) < _nonzero_ratio(alpha_black_on_white)
            else alpha_black_on_white
        )

    colored = np.zeros((grayscale.shape[0], grayscale.shape[1], 4), dtype=np.uint8)
    colored[..., 0] = int(text_color[0])
    colored[..., 1] = int(text_color[1])
    colored[..., 2] = int(text_color[2])
    colored[..., 3] = alpha

    image = Image.fromarray(colored, mode="RGBA")
    bbox = image.getbbox()
    if bbox is None:
        return None
    return image.crop(bbox)


def _normalize_chained_scripts(expression: str) -> str:
    normalized = _SIMPLE_SUPERSCRIPT_PATTERN.sub(r"^{\g<atom>}", expression)
    normalized = _SIMPLE_SUBSCRIPT_PATTERN.sub(r"_{\g<atom>}", normalized)

    for _ in range(6):
        updated = _CHAINED_SUPERSCRIPT_PATTERN.sub(r"^{\g<first>^{\g<second>}}", normalized)
        updated = _CHAINED_SUBSCRIPT_PATTERN.sub(r"_{\g<first>_{\g<second>}}", updated)
        updated = _CHAINED_SUPERSCRIPT_SIMPLE_PATTERN.sub(r"^{\g<first>^\g<second>}", updated)
        updated = _CHAINED_SUBSCRIPT_SIMPLE_PATTERN.sub(r"_{\g<first>_\g<second>}", updated)
        updated = _SUPERSUBSUPER_PATTERN.sub(r"^{\g<sup>^{\g<extra_sup>}}_{\g<sub>}", updated)
        updated = _SUBSUPSUB_PATTERN.sub(r"_{\g<sub>_{\g<extra_sub>}}^{\g<sup>}", updated)
        if updated == normalized:
            break
        normalized = updated
    return normalized


def _render_formula_array_with_latex_tools(renderer: Any, expression: str) -> Optional[np.ndarray]:
    latex_engine = getattr(renderer, "latex", None)
    if latex_engine is None:
        return None

    def _compile_formula(math_expression: str) -> Optional[np.ndarray]:
        template = (
            "\\documentclass[12pt]{article}\n"
            "\\usepackage{amsmath,amssymb,amsfonts,mathtools,bm}\n"
            "\\pagestyle{empty}\n"
            "\\begin{document}\n"
            f"${math_expression}$\n"
            "\\end{document}\n"
        )

        work_dir = tempfile.gettempdir()
        tex_file_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".tex",
                prefix="eq-",
                dir=work_dir,
                delete=False,
            ) as handle:
                tex_file_path = handle.name
                handle.write(template)

            command = (
                "xelatex "
                "-interaction nonstopmode "
                "-halt-on-error "
                "-file-line-error "
                f"-output-directory {work_dir} "
                f"{tex_file_path}"
            )
            _, return_code = latex_engine.run_cmd(command)
            pdf_file = Path(tex_file_path).with_suffix(".pdf")
            if return_code != 0 or not pdf_file.exists() or pdf_file.stat().st_size <= 0:
                return None

            image_array = latex_engine.convert_pdf_to_png(pdf_file)
            if image_array is None:
                return None

            cropper = getattr(renderer, "cropper", None)
            if cropper is None:
                return image_array

            try:
                cropped = cropper(image_array)
            except Exception:
                return image_array
            return image_array if cropped is None else cropped
        except Exception:
            return None
        finally:
            if tex_file_path:
                try:
                    latex_engine.clear_files(tex_file_path)
                except Exception:
                    pass

    rendered = _compile_formula(expression)
    if rendered is not None:
        return rendered

    normalized_expression = _normalize_chained_scripts(expression)
    if normalized_expression == expression:
        return None

    return _compile_formula(normalized_expression)


def _render_formula_image(
    formula_text: str,
    font_size: int,
    text_color: Tuple[int, int, int],
) -> Optional[Image.Image]:
    expression = formula_text.strip()
    if not expression:
        return None

    cache_key = (expression, int(font_size), text_color)
    if cache_key in _FORMULA_IMAGE_CACHE:
        cached = _FORMULA_IMAGE_CACHE[cache_key]
        return cached.copy() if cached is not None else None

    renderer = _get_latex_to_image_renderer()
    if renderer is None:
        _FORMULA_IMAGE_CACHE[cache_key] = None
        return None

    rendered_array = _render_formula_array_with_latex_tools(renderer, expression)
    if rendered_array is None:
        _FORMULA_IMAGE_CACHE[cache_key] = None
        return None

    formula_image = _formula_array_to_rgba(rendered_array, text_color)
    if formula_image is None:
        _FORMULA_IMAGE_CACHE[cache_key] = None
        return None

    scale = max(0.35, min(3.0, float(max(6, int(font_size))) / 24.0))
    if abs(scale - 1.0) >= 0.05:
        target_size = (
            max(1, int(formula_image.width * scale)),
            max(1, int(formula_image.height * scale)),
        )
        formula_image = formula_image.resize(target_size, Image.Resampling.LANCZOS)

    _FORMULA_IMAGE_CACHE[cache_key] = formula_image
    return formula_image.copy()


def _image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    if image.mode in {"RGBA", "LA"}:
        image.save(buffer, format="PNG")
    else:
        image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

_DEFAULT_TEMPLATE_DIR = Path("configs") / "generator" / "templates"
_DEFAULT_SECTION_TEMPLATE: Dict[str, Any] = {
    "id": "default",
    "family": "sections",
    "complexity": 2,
    "mode": "sections",
    "blueprint": {
        "text": {"section_count": [3, 5]},
        "table": {"section_count": [1, 2], "rows": [2, 4], "columns": [3, 5]},
        "formula": {"section_count": [1, 2]},
    },
}


def _canonicalize_template_ref(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass
class TemplateSpec:
    template_id: str
    family: str = "sections"
    complexity: int = 2
    weight: float = 1.0
    mode: str = "sections"
    blueprint: Optional[Dict[str, Any]] = None
    aliases: Optional[List[str]] = None
    version: str = "2"
    source: str = "builtin"
    tags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.blueprint is None:
            self.blueprint = {}
        if self.aliases is None:
            self.aliases = []
        if self.tags is None:
            self.tags = []

    def refs(self) -> List[str]:
        refs: List[str] = [_canonicalize_template_ref(self.template_id)]
        for alias in self.aliases or []:
            normalized = _canonicalize_template_ref(alias)
            if normalized and normalized not in refs:
                refs.append(normalized)
        return refs


class TemplateCatalog:
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else _DEFAULT_TEMPLATE_DIR
        self.templates: Dict[str, TemplateSpec] = {}
        self.alias_to_id: Dict[str, str] = {}
        self._loaded = False

    @staticmethod
    def _extract_entries(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            templates = data.get("templates")
            if isinstance(templates, list):
                return [item for item in templates if isinstance(item, dict)]
            if any(key in data for key in ("text", "table", "formula")):
                return [
                    {
                        "id": data.get("id", "default"),
                        "family": data.get("family", "sections"),
                        "complexity": data.get("complexity", 2),
                        "mode": "sections",
                        "version": data.get("version", "2"),
                        "blueprint": {
                            "text": data.get("text", {}),
                            "table": data.get("table", {}),
                            "formula": data.get("formula", {}),
                        },
                    }
                ]
            if "id" in data:
                return [data]
        return []

    @staticmethod
    def _coerce_spec(raw: Dict[str, Any], source: str) -> Optional[TemplateSpec]:
        template_id = _canonicalize_template_ref(str(raw.get("id", "")))
        if not template_id:
            return None

        mode = str(raw.get("mode", "sections")).strip().lower()
        if mode != "sections":
            logger.warning("Unsupported template mode '%s' for '%s'. Coercing to sections.", mode, template_id)
            mode = "sections"

        family = str(raw.get("family") or mode).strip().lower() or mode

        try:
            complexity = int(raw.get("complexity", 2))
        except (TypeError, ValueError):
            complexity = 2
        complexity = max(1, min(5, complexity))

        try:
            weight = float(raw.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        weight = max(0.01, weight)

        aliases_raw = raw.get("aliases", [])
        aliases: List[str] = []
        if isinstance(aliases_raw, list):
            for item in aliases_raw:
                if isinstance(item, str) and item.strip():
                    aliases.append(item.strip())

        tags_raw = raw.get("tags", [])
        tags: List[str] = []
        if isinstance(tags_raw, list):
            for item in tags_raw:
                if isinstance(item, str) and item.strip():
                    tags.append(item.strip())

        blueprint_raw = raw.get("blueprint")
        blueprint: Dict[str, Any] = blueprint_raw if isinstance(blueprint_raw, dict) else {}

        return TemplateSpec(
            template_id=template_id,
            family=family,
            complexity=complexity,
            weight=weight,
            mode=mode,
            blueprint=blueprint,
            aliases=aliases,
            version=str(raw.get("version", "2")),
            source=source,
            tags=tags,
        )

    def load(self) -> None:
        template_by_id: Dict[str, TemplateSpec] = {}

        if self.config_dir.exists():
            import yaml

            for yaml_path in sorted(self.config_dir.glob("*.y*ml")):
                try:
                    with open(yaml_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except Exception as exc:
                    logger.warning("Failed to read template catalog '%s': %s", yaml_path, exc)
                    continue

                for raw in self._extract_entries(data):
                    spec = self._coerce_spec(raw, source=str(yaml_path))
                    if spec is not None:
                        template_by_id[spec.template_id] = spec

        if not template_by_id:
            fallback = self._coerce_spec(dict(_DEFAULT_SECTION_TEMPLATE), source="builtin")
            if fallback is not None:
                template_by_id[fallback.template_id] = fallback

        self.templates = template_by_id
        self.alias_to_id = {}
        for template_id, spec in self.templates.items():
            for ref in spec.refs():
                self.alias_to_id[ref] = template_id

        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def all_specs(self) -> List[TemplateSpec]:
        self._ensure_loaded()
        return [self.templates[key] for key in sorted(self.templates)]

    def get(self, template_ref: str) -> Optional[TemplateSpec]:
        self._ensure_loaded()
        template_id = self.alias_to_id.get(_canonicalize_template_ref(template_ref))
        if not template_id:
            return None
        return self.templates.get(template_id)

    def resolve(
        self,
        template: Optional[str],
        template_family: Optional[str],
        min_complexity: Optional[int],
        max_complexity: Optional[int],
    ) -> List[TemplateSpec]:
        self._ensure_loaded()

        if template:
            resolved = self.get(template)
            if resolved is not None:
                return [resolved]
            logger.warning("Unknown template '%s'; applying filters over full catalog.", template)

        candidates = self.all_specs()
        if template_family:
            family = template_family.strip().lower()
            candidates = [spec for spec in candidates if spec.family == family]
        if min_complexity is not None:
            candidates = [spec for spec in candidates if spec.complexity >= min_complexity]
        if max_complexity is not None:
            candidates = [spec for spec in candidates if spec.complexity <= max_complexity]

        if not candidates:
            logger.warning("Template filters returned no candidates; falling back to full catalog.")
            return self.all_specs()

        return candidates


def parse_coverage_targets(raw: Any) -> Dict[str, float]:
    if raw is None:
        return {}

    parsed: Dict[str, float] = {}

    def put(key: str, value: Any) -> None:
        normalized = key.strip().lower()
        if not normalized:
            return
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return
        parsed[normalized] = max(0.0, min(1.0, ratio))

    if isinstance(raw, dict):
        for key, value in raw.items():
            put(str(key), value)
        return parsed

    items: List[str] = []
    if isinstance(raw, str):
        items.extend(token for token in raw.split(",") if token.strip())
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, str):
                items.extend(token for token in item.split(",") if token.strip())

    for item in items:
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            continue
        put(key, value)

    return parsed


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

        text = _normalize_mathtext_commands(text)
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
        section_blueprint = template_spec.blueprint if template_spec and isinstance(template_spec.blueprint, dict) else {}
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

    @staticmethod
    def _parse_image_line(stripped: str) -> Optional[Tuple[str, str]]:
        return parse_markdown_image_line(stripped)

    @staticmethod
    def _parse_formula_line(stripped: str) -> Optional[str]:
        return parse_markdown_formula_line(stripped)

    @staticmethod
    def _image_placeholder_height(style: MarkdownStyle) -> int:
        return int(max(110, style.body_font_size * 7.0))

    @staticmethod
    def _formula_font_size(style: MarkdownStyle) -> int:
        return int(style.body_font_size * 0.9)

    @staticmethod
    def _resolve_image_asset(
        image_src: str,
        image_assets: Optional[Dict[str, Image.Image]],
    ) -> Optional[Image.Image]:
        if image_assets:
            cached = image_assets.get(image_src)
            if cached is not None:
                return cached

        candidate_path: Optional[Path] = None
        if image_src.startswith("file://"):
            candidate_path = Path(image_src[len("file://") :])
        elif "://" not in image_src:
            candidate_path = Path(image_src)

        if candidate_path is None or not candidate_path.exists() or not candidate_path.is_file():
            return None

        try:
            loaded = Image.open(candidate_path).convert("RGB")
            loaded.load()
            return loaded
        except Exception:
            return None

    @staticmethod
    def _fit_media_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
        safe_max_width = max(1, int(max_width))
        safe_max_height = max(1, int(max_height))
        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        ratio = min(safe_max_width / width, safe_max_height / height)
        if ratio >= 1.0:
            return image.copy()

        new_width = max(1, int(width * ratio))
        new_height = max(1, int(height * ratio))
        if hasattr(Image, "Resampling"):
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image.resize((new_width, new_height), getattr(Image, "LANCZOS", 1))

    def _image_block_height(
        self,
        style: MarkdownStyle,
        image_src: str,
        image_assets: Optional[Dict[str, Image.Image]],
    ) -> int:
        fallback = self._image_placeholder_height(style)
        image_asset = self._resolve_image_asset(image_src, image_assets)
        if image_asset is None:
            return fallback

        available_width = max(30, style.content_width - 12)
        estimated_height = int(available_width * image_asset.height / max(1, image_asset.width))
        max_height = max(fallback, int(style.content_width * 0.85))
        return max(fallback, min(max_height, estimated_height))

    def render(
        self,
        markdown_text: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> Image.Image:
        """Render markdown text to image."""
        lines = markdown_text.split("\n")
        style = self.style

        # First pass: calculate required height
        total_height = style.margin_top + style.margin_bottom
        line_heights = []

        for line in lines:
            height = self._get_line_height(line, image_assets=image_assets)
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
            elif (image_payload := self._parse_image_line(stripped)) is not None:
                current_y = self._draw_image_block(
                    img,
                    draw,
                    image_payload[0],
                    image_payload[1],
                    current_y,
                    style,
                    image_assets,
                )
            elif (formula_text := self._parse_formula_line(stripped)) is not None:
                current_y = self._draw_formula_line(img, draw, formula_text, current_y, style)
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

    def _get_line_height(
        self,
        line: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> int:
        """Calculate height needed for a line."""
        stripped = line.strip()
        base_spacing = int(self.style.line_spacing * self.style.body_font_size)

        if stripped.startswith("# "):
            return int(self.style.h1_font_size * self.style.line_spacing) + 10
        if stripped.startswith("## "):
            return int(self.style.h2_font_size * self.style.line_spacing) + 8
        if stripped.startswith("### "):
            return int(self.style.h3_font_size * self.style.line_spacing) + 6
        if (image_payload := self._parse_image_line(stripped)) is not None:
            return self._image_block_height(self.style, image_payload[1], image_assets) + 14
        if (formula_text := self._parse_formula_line(stripped)) is not None:
            formula_image = _render_formula_image(
                formula_text,
                self._formula_font_size(self.style),
                self.style.code_text_color,
            )
            if formula_image is not None:
                return max(base_spacing + 18, formula_image.height + 20)
            return base_spacing + 18
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

    def _draw_formula_line(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        formula_text: str,
        y: int,
        style: MarkdownStyle,
    ) -> int:
        text = formula_text.strip()
        formula_image = _render_formula_image(
            text,
            self._formula_font_size(style),
            style.code_text_color,
        )

        box_left = style.margin_left
        box_top = y + 1
        box_right = style.margin_left + style.content_width

        if formula_image is not None:
            available_width = max(20, style.content_width - 16)
            available_height = max(20, int(style.content_width * 0.45))
            rendered_formula = self._fit_media_image(formula_image, available_width, available_height)
            box_bottom = box_top + rendered_formula.height + 10

            draw.rectangle(
                [box_left, box_top, box_right, box_bottom],
                fill=style.code_bg_color,
                outline=style.blockquote_border_color,
                width=1,
            )

            formula_x = box_left + max(8, (style.content_width - rendered_formula.width) // 2)
            formula_y = box_top + 5
            canvas.paste(rendered_formula, (formula_x, formula_y), rendered_formula)
            return int(box_bottom + 8)

        x = style.margin_left + 8
        text_y = y + 4
        text_bbox = draw.textbbox((0, 0), text, font=self.code_font)
        text_width = max(1, text_bbox[2] - text_bbox[0])
        x = style.margin_left + max(8, (style.content_width - text_width) // 2)
        bbox = draw.textbbox((x, text_y), text, font=self.code_font)
        box_right = style.margin_left + style.content_width
        box_bottom = bbox[3] + 5

        draw.rectangle(
            [box_left, box_top, box_right, box_bottom],
            fill=style.code_bg_color,
            outline=style.blockquote_border_color,
            width=1,
        )
        draw.text((x, text_y), text, font=self.code_font, fill=style.code_text_color)
        return int(box_bottom + 8)

    def _draw_image_block(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        alt_text: str,
        image_src: str,
        y: int,
        style: MarkdownStyle,
        image_assets: Optional[Dict[str, Image.Image]],
    ) -> int:
        block_height = self._image_block_height(style, image_src, image_assets)
        left = style.margin_left
        top = y + 4
        right = style.margin_left + style.content_width
        bottom = top + block_height

        image_asset = self._resolve_image_asset(image_src, image_assets)
        if image_asset is not None:
            draw.rectangle(
                [left, top, right, bottom],
                fill=(250, 250, 250),
                outline=(175, 175, 175),
                width=1,
            )

            max_width = max(24, style.content_width - 12)
            max_height = max(24, block_height - 12)
            rendered_image = self._fit_media_image(image_asset.convert("RGB"), max_width, max_height)
            paste_x = left + (style.content_width - rendered_image.width) // 2
            paste_y = top + (block_height - rendered_image.height) // 2
            canvas.paste(rendered_image, (paste_x, paste_y))
            return int(bottom + 10)

        draw.rectangle(
            [left, top, right, bottom],
            fill=(245, 245, 245),
            outline=(170, 170, 170),
            width=2,
        )
        draw.line([(left + 8, top + 8), (right - 8, bottom - 8)], fill=(190, 190, 190), width=1)
        draw.line([(left + 8, bottom - 8), (right - 8, top + 8)], fill=(190, 190, 190), width=1)

        label = f"Image: {alt_text}" if alt_text else "Image"
        label_bbox = draw.textbbox((0, 0), label, font=self.body_font)
        label_x = left + max(8, (style.content_width - (label_bbox[2] - label_bbox[0])) // 2)
        label_y = top + max(8, (block_height - (label_bbox[3] - label_bbox[1])) // 2)
        draw.rectangle(
            [label_x - 6, label_y - 3, label_x + (label_bbox[2] - label_bbox[0]) + 6, label_y + (label_bbox[3] - label_bbox[1]) + 3],
            fill=(255, 255, 255),
        )
        draw.text((label_x, label_y), label, font=self.body_font, fill=(90, 90, 90))

        return int(bottom + 10)

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

    def _prepare_component_markdown(
        self,
        markdown_text: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> str:
        prepared_lines: List[str] = []
        for raw_line in markdown_text.splitlines():
            stripped = raw_line.strip()

            image_payload = parse_markdown_image_line(stripped)
            if image_payload is not None:
                alt_text = image_payload[0] or "Image"
                safe_alt = escape(alt_text)
                image_src = image_payload[1]
                image_asset = MarkdownRenderer._resolve_image_asset(image_src, image_assets)
                if image_asset is not None:
                    data_uri = _image_to_data_uri(image_asset)
                    prepared_lines.extend(
                        [
                            '<figure class="md-image-placeholder">',
                            f'  <img class="md-image-rendered" src="{data_uri}" alt="{safe_alt}" />',
                            f"  <figcaption>{safe_alt}</figcaption>",
                            "</figure>",
                        ]
                    )
                else:
                    prepared_lines.extend(
                        [
                            '<figure class="md-image-placeholder">',
                            f'  <div class="md-image-box" aria-label="{safe_alt}"><span>{safe_alt}</span></div>',
                            f"  <figcaption>{safe_alt}</figcaption>",
                            "</figure>",
                        ]
                    )
                continue

            formula_text = parse_markdown_formula_line(stripped)
            if formula_text is not None:
                formula_image = _render_formula_image(
                    formula_text,
                    MarkdownRenderer._formula_font_size(self.style),
                    self.style.code_text_color,
                )
                if formula_image is not None:
                    formula_data_uri = _image_to_data_uri(formula_image)
                    prepared_lines.append(
                        f'<div class="md-formula"><img class="md-formula-img" src="{formula_data_uri}" alt="formula" /></div>'
                    )
                else:
                    prepared_lines.append(f'<div class="md-formula">{escape(formula_text)}</div>')
                continue

            prepared_lines.append(raw_line)

        return "\n".join(prepared_lines)

    def _estimate_viewport_height(self, markdown_text: str) -> int:
        lines = markdown_text.splitlines() or [""]
        body_line_px = int(self.style.body_font_size * self.style.line_spacing)
        chars_per_line = max(18, self.style.content_width // max(self.style.body_font_size - 1, 8))

        wrapped_line_count = 0
        header_bonus = 0
        code_bonus = 0
        table_bonus = 0
        image_bonus = 0
        formula_bonus = 0
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
            if parse_markdown_image_line(line):
                image_bonus += max(120, int(self.style.body_font_size * 8.5))
            formula_text = parse_markdown_formula_line(line)
            if formula_text:
                formula_image = _render_formula_image(
                    formula_text,
                    MarkdownRenderer._formula_font_size(self.style),
                    self.style.code_text_color,
                )
                if formula_image is not None:
                    formula_bonus += formula_image.height + 12
                else:
                    formula_bonus += int(body_line_px * 1.6)

        estimated = (
            self.style.margin_top
            + self.style.margin_bottom
            + wrapped_line_count * body_line_px
            + header_bonus
            + code_bonus
            + table_bonus
            + image_bonus
            + formula_bonus
            + 120
        )
        return max(300, min(9000, int(estimated)))

    def _build_html_document(
        self,
        markdown_text: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> str:
        prepared_markdown = self._prepare_component_markdown(markdown_text, image_assets=image_assets)
        rendered_html = self._coerce_markdown_html(prepared_markdown)
        page_width = self.style.margin_left + self.style.content_width + self.style.margin_right
        css = f"""
@font-face {{
  font-family: 'RenderFont';
  src: url('file://{escape(self.font_path)}') format('truetype');
}}
html, body {{
  margin: 0;
  padding: 0;
  width: {page_width}px;
  background: rgb{self.style.background_color};
}}
*, *::before, *::after {{
  box-sizing: border-box;
}}
.markdown-body {{
  width: {page_width}px;
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
.markdown-body .md-formula {{
  margin: 0 0 12px 0;
  padding: 8px 10px;
  background: rgb{self.style.code_bg_color};
  color: rgb{self.style.code_text_color};
  border: 1px solid rgba(0, 0, 0, 0.2);
  font-family: 'RenderFont', monospace;
  font-size: {self.style.code_font_size}px;
  overflow-wrap: anywhere;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}}
.markdown-body .md-image-placeholder {{
  margin: 0 0 12px 0;
}}
.markdown-body .md-image-box {{
  width: 100%;
  min-height: 130px;
  border: 2px solid rgba(0, 0, 0, 0.28);
  background: rgba(240, 240, 240, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(0, 0, 0, 0.6);
  font-weight: 600;
}}
.markdown-body .md-image-rendered {{
  width: auto;
  max-width: 100%;
  display: block;
  border: 1px solid rgba(0, 0, 0, 0.28);
  object-fit: contain;
  max-height: 520px;
  margin: 0 auto;
}}
.markdown-body .md-image-placeholder figcaption {{
  margin-top: 6px;
  color: rgba(0, 0, 0, 0.65);
  font-size: {max(10, self.style.body_font_size - 1)}px;
}}
.markdown-body .md-formula .md-formula-img {{
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
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

    def render(
        self,
        markdown_text: str,
        image_assets: Optional[Dict[str, Image.Image]] = None,
    ) -> Image.Image:
        try:
            Html2Image = importlib.import_module("html2image").Html2Image
        except ImportError as exc:
            raise RuntimeError(
                "html2image package is required for html->image rendering. "
                "Install with: uv sync --group generate"
            ) from exc

        width = self.style.margin_left + self.style.content_width + self.style.margin_right
        height = self._estimate_viewport_height(markdown_text)
        html_doc = self._build_html_document(markdown_text, image_assets=image_assets)

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
        self.template_catalog = TemplateCatalog()
        self.template_specs: List[TemplateSpec] = self.template_catalog.all_specs()
        self.template_counts: Counter[str] = Counter()
        self.family_counts: Counter[str] = Counter()
        self.coverage_targets: Dict[str, float] = {}
        self.template_family: Optional[str] = None
        self.min_template_complexity: Optional[int] = None
        self.max_template_complexity: Optional[int] = None
        self.template_config_dir: Optional[str] = None
        self.add_noise = True
        self.add_blur = False
        self.noise_ratio = 0.1
        self.blur_ratio = 0.1
        self.similar_char_ratio = 0.08
        self.markdown_renderer = "pil"
        self.style_profile = "balanced"
        self.novelty_window = DEFAULT_NOVELTY_WINDOW
        self.novelty_threshold = DEFAULT_NOVELTY_THRESHOLD
        self.novelty_max_attempts = DEFAULT_NOVELTY_MAX_ATTEMPTS
        self._recent_signatures: deque[str] = deque(maxlen=self.novelty_window)
        self.base_seed: Optional[int] = None
        self.max_render_width = A4_MAX_WIDTH_PX
        self.max_render_height = A4_MAX_HEIGHT_PX
        self.max_render_aspect_ratio = MAX_RENDER_ASPECT_RATIO

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

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_ratio(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, ratio))

    @staticmethod
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

    @staticmethod
    def _normalize_choice(
        value: Any,
        allowed: set[str],
        fallback: str,
        warning_label: str,
    ) -> str:
        normalized = str(value).strip().lower()
        if normalized in allowed:
            return normalized
        logger.warning(
            "Unknown %s '%s'. Falling back to '%s'.",
            warning_label,
            normalized,
            fallback,
        )
        return fallback

    def _resolve_effect_settings(
        self,
        *,
        enabled_key: str,
        ratio_key: str,
        enabled_default: bool,
        ratio_default: float,
        kwargs: dict[str, Any],
    ) -> tuple[bool, float]:
        enabled = self._coerce_bool(kwargs.get(enabled_key), enabled_default)
        ratio = self._coerce_ratio(kwargs.get(ratio_key), ratio_default)
        if enabled_key in kwargs and kwargs.get(enabled_key) is not None:
            ratio = 1.0 if enabled else 0.0
        return enabled, ratio

    def _resolve_template_specs(self, template: Optional[str]) -> List[TemplateSpec]:
        return self.template_catalog.resolve(
            template=template,
            template_family=self.template_family,
            min_complexity=self.min_template_complexity,
            max_complexity=self.max_template_complexity,
        )

    def _configure_template_selection(self, **kwargs) -> None:
        self.template_family = kwargs.get("template_family")
        self.min_template_complexity = self._coerce_optional_int(
            kwargs.get("min_template_complexity")
        )
        self.max_template_complexity = self._coerce_optional_int(
            kwargs.get("max_template_complexity")
        )
        if (
            self.min_template_complexity is not None
            and self.max_template_complexity is not None
            and self.min_template_complexity > self.max_template_complexity
        ):
            self.min_template_complexity, self.max_template_complexity = (
                self.max_template_complexity,
                self.min_template_complexity,
            )

        requested_catalog_dir = kwargs.get("template_config_dir")
        if requested_catalog_dir != self.template_config_dir:
            self.template_catalog = TemplateCatalog(config_dir=requested_catalog_dir)
            self.template_config_dir = requested_catalog_dir

        template = kwargs.get("template")
        self.template_specs = self._resolve_template_specs(template)
        self.coverage_targets = parse_coverage_targets(kwargs.get("coverage_targets"))

    def _configure_rendering(self, **kwargs) -> None:
        self.add_noise, self.noise_ratio = self._resolve_effect_settings(
            enabled_key="add_noise",
            ratio_key="noise_ratio",
            enabled_default=True,
            ratio_default=0.1,
            kwargs=kwargs,
        )
        self.add_blur, self.blur_ratio = self._resolve_effect_settings(
            enabled_key="add_blur",
            ratio_key="blur_ratio",
            enabled_default=False,
            ratio_default=0.1,
            kwargs=kwargs,
        )
        self.similar_char_ratio = float(kwargs.get("similar_char_ratio", 0.08))

        self.markdown_renderer = self._normalize_choice(
            kwargs.get("markdown_renderer", self.markdown_renderer),
            {"pil", "html2image"},
            "pil",
            "markdown renderer",
        )

        self.style_profile = self._normalize_choice(
            kwargs.get("style_profile", self.style_profile),
            {"legacy", "balanced", "aggressive"},
            "balanced",
            "style profile",
        )

    def _configure_novelty(self, **kwargs) -> None:
        self.novelty_window = max(
            5,
            self._coerce_optional_int(kwargs.get("novelty_window")) or self.novelty_window,
        )
        self.novelty_threshold = self._coerce_ratio(
            kwargs.get("novelty_threshold"),
            self.novelty_threshold,
        )
        self.novelty_max_attempts = max(
            1,
            self._coerce_optional_int(kwargs.get("novelty_max_attempts"))
            or self.novelty_max_attempts,
        )
        self._recent_signatures = deque(
            self._recent_signatures,
            maxlen=self.novelty_window,
        )

    def _configure_content_sources(self, **kwargs) -> None:
        self.data_generator.configure_content_sources(
            formula_source_mode=kwargs.get(
                "formula_source_mode",
                self.data_generator.formula_source_mode,
            ),
            formula_dataset_path=kwargs.get("formula_dataset_path", self.data_generator.formula_dataset_path),
            formula_dataset_weight=kwargs.get(
                "formula_dataset_weight",
                self.data_generator.formula_source_weights.get("dataset", DEFAULT_FORMULA_SOURCE_WEIGHTS["dataset"]),
            ),
            formula_random_weight=kwargs.get(
                "formula_random_weight",
                self.data_generator.formula_source_weights.get("random", DEFAULT_FORMULA_SOURCE_WEIGHTS["random"]),
            ),
            formula_synthetic_weight=kwargs.get(
                "formula_synthetic_weight",
                self.data_generator.formula_source_weights.get("synthetic", DEFAULT_FORMULA_SOURCE_WEIGHTS["synthetic"]),
            ),
        )

    def _configure_generation(self, **kwargs) -> None:
        if "seed" in kwargs:
            self.base_seed = self._coerce_optional_int(kwargs.get("seed"))

        self._configure_template_selection(**kwargs)
        self._configure_rendering(**kwargs)
        self._configure_novelty(**kwargs)
        self._configure_content_sources(**kwargs)

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

    def _mutate_text_generator_sections(
        self,
        markdown_text: str,
        ratio: float,
        merge_order: List[str],
    ) -> Tuple[str, int]:
        if not merge_order:
            return markdown_text, 0

        lines = markdown_text.splitlines()
        if not lines:
            return markdown_text, 0

        preface_lines: List[str] = []
        sections: List[str] = []
        current_section: List[str] = []
        in_section = False

        for line in lines:
            if line.startswith("## "):
                if current_section:
                    sections.append("\n".join(current_section).rstrip())
                current_section = [line]
                in_section = True
                continue

            if in_section:
                current_section.append(line)
            else:
                preface_lines.append(line)

        if current_section:
            sections.append("\n".join(current_section).rstrip())

        if len(sections) != len(merge_order):
            return markdown_text, 0

        mutated_sections: List[str] = []
        mutated_count = 0
        for section_text, section_type in zip(sections, merge_order):
            if section_type == "text":
                mutated_section, section_mutations = self._mutate_similar_text(section_text, ratio)
                mutated_sections.append(mutated_section)
                mutated_count += section_mutations
            else:
                mutated_sections.append(section_text)

        prefix = "\n".join(preface_lines).rstrip()
        body = "\n\n".join(mutated_sections).strip()
        if prefix and body:
            return f"{prefix}\n\n{body}", mutated_count
        if body:
            return body, mutated_count
        return prefix, mutated_count

    def _derive_sample_seed(self, sample_index: int, attempt: int) -> Optional[int]:
        if self.base_seed is None:
            return None
        return int(self.base_seed + sample_index * 1009 + attempt * 9176)

    def _seed_for_sample(self, sample_seed: Optional[int]) -> None:
        if sample_seed is None:
            return

        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32 - 1))

        faker = getattr(self.data_generator.data, "faker", None)
        if faker is not None:
            try:
                faker.seed_instance(sample_seed)
            except Exception:
                pass

    def _fit_image_to_a4(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        return image, False

    @staticmethod
    def _structure_signature(markdown_text: str) -> str:
        tokens: List[str] = []
        for raw_line in markdown_text.splitlines():
            line = raw_line.strip()
            if not line:
                tokens.append("blank")
            elif parse_markdown_image_line(line):
                tokens.append("image")
            elif parse_markdown_formula_line(line):
                tokens.append("formula")
            elif line.startswith("# "):
                tokens.append("h1")
            elif line.startswith("## "):
                tokens.append("h2")
            elif line.startswith("### "):
                tokens.append("h3")
            elif line.startswith("```"):
                tokens.append("code")
            elif line.startswith("| ") and line.endswith(" |"):
                tokens.append("table")
            elif line.startswith("- [ ") or line.startswith("- [x"):
                tokens.append("check")
            elif line.startswith("- "):
                tokens.append("ul")
            elif line and line[0].isdigit() and ". " in line:
                tokens.append("ol")
            elif line.startswith("> "):
                tokens.append("quote")
            elif line in {"---", "***"}:
                tokens.append("rule")
            else:
                tokens.append("p")
        return "|".join(tokens)

    def _novelty_score(self, signature: str) -> float:
        if not self._recent_signatures:
            return 0.0
        return max(
            SequenceMatcher(None, signature, existing).ratio()
            for existing in self._recent_signatures
        )

    def _select_template_spec(self) -> Tuple[TemplateSpec, float]:
        if not self.template_specs:
            self.template_specs = self.template_catalog.all_specs()

        total_generated = sum(self.family_counts.values())
        weights: List[float] = []
        for spec in self.template_specs:
            template_seen = self.template_counts.get(spec.template_id, 0)
            family_seen = self.family_counts.get(spec.family, 0)

            diversity_factor = 1.0 / (1.0 + template_seen * 0.45)
            family_balance_factor = 1.0 / (1.0 + family_seen * 0.2)
            coverage_factor = 1.0

            if self.coverage_targets:
                target_ratio = self.coverage_targets.get(spec.family)
                if target_ratio is not None:
                    if total_generated == 0:
                        coverage_factor = 1.5
                    else:
                        observed_ratio = family_seen / total_generated
                        deficit = target_ratio - observed_ratio
                        coverage_factor = max(0.25, 1.0 + deficit * 5.0)

            weights.append(max(0.01, spec.weight * diversity_factor * family_balance_factor * coverage_factor))

        selected_index = random.choices(range(len(self.template_specs)), weights=weights, k=1)[0]
        return self.template_specs[selected_index], weights[selected_index]

    def generate(
        self,
        num_images: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate markdown images."""
        self._configure_generation(**kwargs)
        self.template_counts = Counter()
        self.family_counts = Counter()
        self._recent_signatures = deque(maxlen=self.novelty_window)

        metadata = []
        for idx in tqdm(range(num_images), desc="Generating markdown images"):
            image, meta = self.generate_single(sample_index=idx)

            # Save image
            filename = f"markdown_{idx:05d}.png"
            self.save_image(image, filename)
            meta["file_name"] = str(self.output_dir / filename)

            metadata.append(meta)

        return metadata

    def generate_single(self, sample_index: int = 0, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        if kwargs:
            self._configure_generation(**kwargs)

        available_specs = self.template_specs or self.template_catalog.all_specs()
        if not available_specs:
            raise RuntimeError("Template catalog is empty. Add template specs under configs/generator/templates.")

        selected_template: TemplateSpec = available_specs[0]
        selected_weight = 0.0
        markdown_text = ""
        mutation_count = 0
        signature = ""
        novelty_score = 0.0
        sample_seed: Optional[int] = None
        selection_attempt = 1
        merge_order: List[str] = []

        for attempt in range(self.novelty_max_attempts):
            sample_seed = self._derive_sample_seed(sample_index, attempt)
            self._seed_for_sample(sample_seed)

            selected_template, selected_weight = self._select_template_spec()

            original_markdown = self.data_generator.generate_markdown(
                template_id=selected_template.template_id,
                template_spec=selected_template,
            )
            merge_order = self.data_generator.pop_merge_order()
            markdown_text, mutation_count = self._mutate_text_generator_sections(
                original_markdown,
                self.similar_char_ratio,
                merge_order,
            )
            signature = self._structure_signature(markdown_text)
            novelty_score = self._novelty_score(signature)
            selection_attempt = attempt + 1

            if novelty_score < self.novelty_threshold or attempt == self.novelty_max_attempts - 1:
                break

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
        image, a4_clipped = self._fit_image_to_a4(image)

        self.template_counts[selected_template.template_id] += 1
        self.family_counts[selected_template.family] += 1
        self._recent_signatures.append(signature)

        generated_count = max(1, sum(self.family_counts.values()))
        family_ratio = self.family_counts[selected_template.family] / generated_count

        metadata = {
            "template": selected_template.template_id,
            "template_id": selected_template.template_id,
            "template_family": selected_template.family,
            "template_complexity": selected_template.complexity,
            "template_mode": selected_template.mode,
            "template_version": selected_template.version,
            "template_source": selected_template.source,
            "template_weight": round(selected_weight, 6),
            "GT_markdown": markdown_text,
            "GT_json": markdown_to_json_ast(markdown_text),
            "similar_char_mutations": mutation_count,
            "renderer": self.markdown_renderer,
            "style_profile": self.style_profile,
            "sample_index": sample_index,
            "sample_seed": sample_seed,
            "selection_attempt": selection_attempt,
            "structure_signature": signature,
            "novelty_score": round(novelty_score, 6),
            "family_ratio": round(family_ratio, 6),
            "merge_order": merge_order,
            "a4_scaled": a4_clipped,
            "a4_clipped": a4_clipped,
            "image_width": image.width,
            "image_height": image.height,
        }
        return image, metadata

    @staticmethod
    def _base_styles() -> List[MarkdownStyle]:
        return [
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

    @staticmethod
    def _clamp_color(value: int) -> int:
        return max(0, min(255, value))

    def _jitter_color(self, color: Tuple[int, int, int], span: int) -> Tuple[int, int, int]:
        return (
            self._clamp_color(color[0] + random.randint(-span, span)),
            self._clamp_color(color[1] + random.randint(-span, span)),
            self._clamp_color(color[2] + random.randint(-span, span)),
        )

    def _random_style(self) -> MarkdownStyle:
        """Generate random style variations."""
        selected = random.choice(self._base_styles())

        if self.style_profile == "legacy":
            selected.margin_top += random.randint(-8, 14)
            selected.margin_bottom += random.randint(-8, 14)
            selected.content_width += random.randint(-24, 24)
            selected.line_spacing = max(1.3, min(1.7, selected.line_spacing + random.uniform(-0.08, 0.1)))
            return selected

        if self.style_profile == "balanced":
            selected.margin_top += random.randint(-16, 24)
            selected.margin_bottom += random.randint(-16, 24)
            selected.margin_left += random.randint(-10, 16)
            selected.margin_right += random.randint(-10, 16)
            selected.content_width += random.randint(-64, 72)
            selected.line_spacing = max(1.2, min(1.9, selected.line_spacing + random.uniform(-0.2, 0.25)))
            selected.background_color = self._jitter_color(selected.background_color, 12)
            selected.h1_color = self._jitter_color(selected.h1_color, 16)
            selected.h2_color = self._jitter_color(selected.h2_color, 16)
            selected.text_color = self._jitter_color(selected.text_color, 12)
            selected.link_color = self._jitter_color(selected.link_color, 24)
        else:
            selected.margin_top += random.randint(-24, 36)
            selected.margin_bottom += random.randint(-24, 36)
            selected.margin_left += random.randint(-18, 24)
            selected.margin_right += random.randint(-18, 24)
            selected.content_width += random.randint(-96, 108)
            selected.line_spacing = max(1.15, min(2.0, selected.line_spacing + random.uniform(-0.28, 0.35)))
            selected.body_font_size = max(12, min(18, selected.body_font_size + random.randint(-2, 3)))
            selected.code_font_size = max(10, min(16, selected.code_font_size + random.randint(-1, 3)))
            selected.h1_font_size = max(
                selected.body_font_size + 6,
                min(40, selected.h1_font_size + random.randint(-4, 8)),
            )
            selected.h2_font_size = max(
                selected.body_font_size + 3,
                min(34, selected.h2_font_size + random.randint(-3, 6)),
            )
            selected.h3_font_size = max(
                selected.body_font_size + 1,
                min(28, selected.h3_font_size + random.randint(-2, 5)),
            )

            bg_base = random.randint(220, 255)
            selected.background_color = (
                bg_base,
                self._clamp_color(bg_base + random.randint(-12, 10)),
                self._clamp_color(bg_base + random.randint(-12, 10)),
            )
            selected.text_color = (
                random.randint(18, 70),
                random.randint(18, 70),
                random.randint(18, 70),
            )
            selected.h1_color = (
                random.randint(0, 60),
                random.randint(0, 60),
                random.randint(0, 80),
            )
            selected.h2_color = self._jitter_color(selected.h1_color, 20)
            selected.link_color = (
                random.randint(0, 40),
                random.randint(80, 150),
                random.randint(140, 220),
            )

        selected.margin_top = max(16, selected.margin_top)
        selected.margin_bottom = max(16, selected.margin_bottom)
        selected.margin_left = max(20, selected.margin_left)
        selected.margin_right = max(20, selected.margin_right)
        selected.content_width = max(460, min(720, selected.content_width))
        return selected
