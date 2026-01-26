# VLM/OCR Evaluation Pipeline - Implementation Plan

## Overview

Synthetic OCR 데이터셋을 활용하여 VLM(Vision Language Model) 및 OCR 모델의 성능을 평가하는 파이프라인을 구축합니다.

### 목표
- 다양한 추론 백엔드 지원 (API, vLLM, SGLang, Ollama, Transformers)
- 4가지 포맷 평가 (Sentence, Table, Document, Markdown)
- 표준 OCR 메트릭 계산 (CER, WER, TEDS, Accuracy 등)
- 멀티 모델 비교 및 리포트 생성

### 지원 추론 백엔드

| Backend | 용도 | 예시 모델 |
|---------|------|-----------|
| OpenAI API | 클라우드 API | GPT-4o, GPT-4V |
| Anthropic API | 클라우드 API | Claude 3.5/4 Sonnet |
| Google API | 클라우드 API | Gemini 1.5/2.0 |
| vLLM | 고성능 로컬 추론 | Qwen2-VL, InternVL |
| SGLang | 고성능 로컬 추론 | Qwen2-VL, LLaVA |
| Ollama | 간편한 로컬 추론 | LLaVA, Moondream |
| Transformers | HuggingFace 직접 로드 | Florence-2, Pix2Struct |

---

## Architecture

### 디렉토리 구조

```
src/
├── evaluation/                    # 평가 파이프라인 핵심
│   ├── __init__.py
│   ├── config.py                  # Pydantic 설정 스키마
│   ├── pipeline.py                # 메인 오케스트레이터
│   ├── runner.py                  # 배치 추론 + 체크포인팅
│   ├── report.py                  # 리포트 생성 (JSON/HTML/Markdown)
│   ├── comparator.py              # 멀티 모델 비교
│   └── templates/
│       ├── report.md.j2
│       └── report.html.j2
├── models/
│   ├── __init__.py                # 통합 exports
│   ├── base.py                    # VLMModel 공통 인터페이스
│   ├── registry.py                # 모델 팩토리
│   ├── api/
│   │   ├── __init__.py
│   │   ├── base.py                # API 모델 베이스 (Rate limiting)
│   │   ├── openai_vision.py       # GPT-4o, GPT-4V
│   │   ├── claude_vision.py       # Claude 3.5/4
│   │   └── gemini_vision.py       # Gemini 1.5/2.0
│   └── local/
│       ├── __init__.py
│       ├── transformers.py        # HuggingFace Transformers
│       ├── vllm.py                # vLLM
│       ├── sglang.py              # SGLang
│       └── ollama.py              # Ollama
├── metrics/
│   ├── __init__.py                # 메트릭 exports
│   ├── edit_distance.py           # CER, WER (기존 + 확장)
│   ├── normalization.py           # 텍스트 정규화
│   ├── table_edit_distance.py     # TEDS (기존)
│   └── table_document_metrics.py  # 테이블/문서 메트릭 (기존)
└── evaluate.py                    # CLI 엔트리포인트

tests/
├── evaluation/
│   ├── test_config.py
│   ├── test_pipeline.py
│   ├── test_runner.py
│   └── test_metrics.py
└── models/
    ├── test_api_models.py
    └── test_local_models.py
```

### 데이터 흐름

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  HuggingFace    │────▶│  Evaluation      │────▶│  Report         │
│  Dataset        │     │  Pipeline        │     │  Generator      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  Model Backend   │
                        │  (API/Local)     │
                        └──────────────────┘
```

---

## Implementation Phases

### Phase 1: Core Infrastructure

**목표**: 설정 스키마, 베이스 클래스, 의존성 설정

#### 1.1 Configuration Schema

**File**: `src/evaluation/config.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class FormatType(str, Enum):
    SENTENCE = "sentence"
    TABLE = "table"
    DOCUMENT = "document"
    MARKDOWN = "markdown"

class InferenceBackend(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    SGLANG = "sglang"
    OLLAMA = "ollama"

class ModelConfig(BaseModel):
    """모델 설정"""
    model_id: str
    backend: InferenceBackend
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3
    # API 전용
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    rate_limit_rpm: Optional[int] = None
    # Local 전용
    device: str = "cuda"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1

class EvaluationConfig(BaseModel):
    """평가 파이프라인 설정"""
    dataset_id: str
    subset: str = "default"
    split: str = "test"
    format_type: FormatType
    model: ModelConfig
    batch_size: int = Field(default=1, ge=1)
    max_samples: Optional[int] = None
    output_dir: str = "./evaluation_results"
    prompt: Optional[str] = None
    resume_from_checkpoint: bool = True
```

#### 1.2 Base Model Interface

**File**: `src/models/base.py`

```python
from abc import ABC, abstractmethod
from typing import List
from PIL import Image

class VLMModel(ABC):
    """모든 VLM 모델의 공통 인터페이스"""

    @abstractmethod
    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """동기 추론"""
        pass

    async def run_async(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """비동기 추론 (기본: 동기 래핑)"""
        return self.run(prompts, images)

    def run_batch(self, prompts: List[str], images: List[Image.Image],
                  batch_size: int = 1) -> List[str]:
        """배치 추론"""
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_images = images[i:i+batch_size]
            results.extend(self.run(batch_prompts, batch_images))
        return results
```

#### 1.3 Dependencies Update

**File**: `pyproject.toml`

```toml
[project.optional-dependencies]
eval = [
    "pydantic>=2.0.0",
    "openai>=1.0.0",
    "anthropic>=0.25.0",
    "google-generativeai>=0.5.0",
    "tenacity>=8.0.0",
    "jinja2>=3.0.0",
    "aiohttp>=3.9.0",
    "aiolimiter>=1.1.0",
    "ollama>=0.3.0",
]
vllm = ["vllm>=0.6.0"]
sglang = ["sglang[all]>=0.3.0"]
```

---

### Phase 2: Model Integrations

**목표**: 7가지 추론 백엔드 구현

#### 2.1 API Base Class

**File**: `src/models/api/base.py`

```python
from abc import abstractmethod
from typing import List
from PIL import Image
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

class APIModel(VLMModel):
    """API 기반 모델의 베이스 클래스"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._limiter = AsyncLimiter(
            config.rate_limit_rpm or 60,
            time_period=60
        )

    @abstractmethod
    async def _call_api(self, prompt: str, image: Image.Image) -> str:
        """단일 API 호출 (서브클래스에서 구현)"""
        pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_with_retry(self, prompt: str, image: Image.Image) -> str:
        async with self._limiter:
            return await self._call_api(prompt, image)

    async def run_async(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        tasks = [self._call_with_retry(p, img) for p, img in zip(prompts, images)]
        return await asyncio.gather(*tasks)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        return asyncio.run(self.run_async(prompts, images))
```

#### 2.2 OpenAI Vision

**File**: `src/models/api/openai_vision.py`

```python
from openai import AsyncOpenAI
import base64
from io import BytesIO

class OpenAIVision(APIModel):
    """GPT-4o, GPT-4V 지원"""

    SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
        )

    def _encode_image(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    async def _call_api(self, prompt: str, image: Image.Image) -> str:
        base64_image = self._encode_image(image)
        response = await self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content
```

#### 2.3 Claude Vision

**File**: `src/models/api/claude_vision.py`

```python
from anthropic import AsyncAnthropic
import base64
from io import BytesIO

class ClaudeVision(APIModel):
    """Claude 3.5/4 Sonnet 지원"""

    SUPPORTED_MODELS = ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022"]

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=config.api_key)

    async def _call_api(self, prompt: str, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode()

        response = await self.client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text
```

#### 2.4 Gemini Vision

**File**: `src/models/api/gemini_vision.py`

```python
import google.generativeai as genai

class GeminiVision(APIModel):
    """Gemini 1.5/2.0 지원"""

    SUPPORTED_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"]

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model_id)

    async def _call_api(self, prompt: str, image: Image.Image) -> str:
        response = await self.model.generate_content_async(
            [image, prompt],
            generation_config=genai.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            ),
        )
        return response.text
```

#### 2.5 HuggingFace Transformers

**File**: `src/models/local/transformers.py`

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from typing import List
from PIL import Image

class TransformersVLM(VLMModel):
    """HuggingFace Transformers 기반 VLM"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.model_id,
            torch_dtype=getattr(torch, config.dtype),
            device_map=config.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        results = []
        for prompt, image in zip(prompts, images):
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=self.config.temperature > 0,
                    temperature=self.config.temperature or 1.0,
                )

            # 입력 토큰 제외하고 디코딩
            input_len = inputs.get("input_ids", inputs.get("pixel_values")).shape[-1]
            decoded = self.processor.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            )
            results.append(decoded)
        return results
```

#### 2.6 vLLM

**File**: `src/models/local/vllm.py`

```python
from vllm import LLM, SamplingParams
from typing import List
from PIL import Image

class VLLMModel(VLMModel):
    """vLLM 기반 고성능 추론"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm = LLM(
            model=config.model_id,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )
        self.sampling_params = SamplingParams(
            temperature=config.temperature or 0.0,
            max_tokens=config.max_tokens,
        )

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        # vLLM 멀티모달 입력 형식
        inputs = []
        for prompt, image in zip(prompts, images):
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            })

        outputs = self.llm.generate(inputs, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
```

#### 2.7 SGLang

**File**: `src/models/local/sglang.py`

```python
import sglang as sgl
from sglang import RuntimeEndpoint
from typing import List, Optional
from PIL import Image

class SGLangModel(VLMModel):
    """SGLang 기반 추론"""

    def __init__(self, config: ModelConfig):
        self.config = config

        if config.api_base:
            # 외부 서버 사용
            sgl.set_default_backend(RuntimeEndpoint(config.api_base))
            self.runtime = None
        else:
            # 로컬 런타임 생성
            self.runtime = sgl.Runtime(
                model_path=config.model_id,
                tp_size=config.tensor_parallel_size,
            )
            sgl.set_default_backend(self.runtime)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        @sgl.function
        def vlm_inference(s, prompt: str, image: Image.Image):
            s += sgl.image(image)
            s += sgl.user(prompt)
            s += sgl.assistant(sgl.gen("response", max_tokens=self.config.max_tokens))

        results = []
        for prompt, image in zip(prompts, images):
            state = vlm_inference.run(prompt=prompt, image=image)
            results.append(state["response"])
        return results

    def __del__(self):
        if self.runtime:
            self.runtime.shutdown()
```

#### 2.8 Ollama

**File**: `src/models/local/ollama.py`

```python
import ollama
import base64
from io import BytesIO
from typing import List
from PIL import Image

class OllamaModel(VLMModel):
    """Ollama 기반 로컬 추론"""

    VISION_MODELS = ["llava", "llava-llama3", "bakllava", "moondream", "llava-phi3"]

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = ollama.Client(
            host=config.api_base or "http://localhost:11434"
        )

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        results = []
        for prompt, image in zip(prompts, images):
            response = self.client.chat(
                model=self.config.model_id,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [self._image_to_base64(image)],
                }],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )
            results.append(response["message"]["content"])
        return results
```

#### 2.9 Model Registry

**File**: `src/models/registry.py`

```python
from typing import Type, Dict
from models.base import VLMModel, ModelConfig
from models.api.openai_vision import OpenAIVision
from models.api.claude_vision import ClaudeVision
from models.api.gemini_vision import GeminiVision
from models.local.transformers import TransformersVLM
from models.local.vllm import VLLMModel
from models.local.sglang import SGLangModel
from models.local.ollama import OllamaModel
from evaluation.config import InferenceBackend

MODEL_REGISTRY: Dict[InferenceBackend, Type[VLMModel]] = {
    InferenceBackend.OPENAI: OpenAIVision,
    InferenceBackend.ANTHROPIC: ClaudeVision,
    InferenceBackend.GOOGLE: GeminiVision,
    InferenceBackend.TRANSFORMERS: TransformersVLM,
    InferenceBackend.VLLM: VLLMModel,
    InferenceBackend.SGLANG: SGLangModel,
    InferenceBackend.OLLAMA: OllamaModel,
}

def create_model(config: ModelConfig) -> VLMModel:
    """팩토리 함수로 모델 생성"""
    model_class = MODEL_REGISTRY.get(config.backend)
    if model_class is None:
        raise ValueError(f"Unknown backend: {config.backend}")
    return model_class(config)
```

---

### Phase 3: Enhanced Metrics

**목표**: 텍스트 정규화 및 확장 메트릭

#### 3.1 Text Normalization

**File**: `src/metrics/normalization.py`

```python
import re
import unicodedata
from typing import Callable, Optional

def normalize_text(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    normalize_whitespace: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """OCR 비교를 위한 텍스트 정규화"""
    if text is None:
        return ""

    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    return text

def normalize_for_language(text: str, lang: str) -> str:
    """언어별 정규화"""
    if lang in ("ko", "ja", "zh"):
        # CJK 문자 사이의 공백 제거
        text = re.sub(
            r'(?<=[\u4e00-\u9fff\uac00-\ud7af\u3040-\u309f\u30a0-\u30ff])\s+'
            r'(?=[\u4e00-\u9fff\uac00-\ud7af\u3040-\u309f\u30a0-\u30ff])',
            '', text
        )
    return text

def create_normalizer(
    lowercase: bool = False,
    remove_punctuation: bool = False,
    language: Optional[str] = None,
) -> Callable[[str], str]:
    """정규화 함수 생성"""
    def normalizer(text: str) -> str:
        text = normalize_text(
            text,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation
        )
        if language:
            text = normalize_for_language(text, language)
        return text
    return normalizer
```

#### 3.2 Extended Edit Distance Metrics

**File**: `src/metrics/edit_distance.py` (확장)

```python
from typing import Callable, Optional
from metrics.normalization import normalize_text

def normalized_cer(
    reference: str,
    hypothesis: str,
    normalize_fn: Optional[Callable[[str], str]] = None
) -> float:
    """정규화된 CER"""
    if normalize_fn:
        reference = normalize_fn(reference)
        hypothesis = normalize_fn(hypothesis)
    return cer(reference, hypothesis)

def normalized_wer(
    reference: str,
    hypothesis: str,
    normalize_fn: Optional[Callable[[str], str]] = None
) -> float:
    """정규화된 WER"""
    if normalize_fn:
        reference = normalize_fn(reference)
        hypothesis = normalize_fn(hypothesis)
    return wer(reference, hypothesis)

def accuracy(reference: str, hypothesis: str) -> float:
    """정확도 (Exact Match)"""
    return 1.0 if reference.strip() == hypothesis.strip() else 0.0

def word_accuracy(reference: str, hypothesis: str) -> float:
    """단어 수준 정확도"""
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())
    if not ref_words:
        return 1.0 if not hyp_words else 0.0
    return len(ref_words & hyp_words) / len(ref_words)

def character_accuracy(reference: str, hypothesis: str) -> float:
    """문자 수준 정확도 (1 - CER)"""
    return max(0.0, 1.0 - cer(reference, hypothesis))
```

#### 3.3 Metrics Module Exports

**File**: `src/metrics/__init__.py`

```python
from metrics.edit_distance import (
    cer, wer, levenshtein_distance,
    normalized_cer, normalized_wer,
    accuracy, word_accuracy, character_accuracy,
)
from metrics.normalization import (
    normalize_text, normalize_for_language, create_normalizer,
)
from metrics.table_edit_distance import TEDS
from metrics.table_document_metrics import (
    evaluate_table,
    evaluate_document,
    evaluate_sentence_metrics,
    evaluate_markdown_metrics,
)

__all__ = [
    "cer", "wer", "levenshtein_distance",
    "normalized_cer", "normalized_wer",
    "accuracy", "word_accuracy", "character_accuracy",
    "normalize_text", "normalize_for_language", "create_normalizer",
    "TEDS",
    "evaluate_table", "evaluate_document",
    "evaluate_sentence_metrics", "evaluate_markdown_metrics",
]
```

---

### Phase 4: Evaluation Pipeline Core

**목표**: 배치 러너, 체크포인팅, 메인 파이프라인

#### 4.1 Inference Result Types

**File**: `src/evaluation/types.py`

```python
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, List, Dict
from pathlib import Path
import json

@dataclass
class InferenceResult:
    """단일 추론 결과"""
    index: int
    prediction: str
    ground_truth: Any
    latency_ms: float
    error: Optional[str] = None

@dataclass
class RunnerState:
    """러너 체크포인트 상태"""
    completed: List[int] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)

    def save(self, path: Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'RunnerState':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

@dataclass
class EvaluationOutput:
    """평가 결과"""
    config: Dict[str, Any]
    metrics: Dict[str, float]
    per_sample_results: List[Dict[str, Any]]
    summary: Dict[str, Any]
```

#### 4.2 Batch Inference Runner

**File**: `src/evaluation/runner.py`

```python
from dataclasses import asdict
from typing import List, Any
from pathlib import Path
from PIL import Image
import time
import asyncio
from tqdm import tqdm

from evaluation.config import EvaluationConfig
from evaluation.types import InferenceResult, RunnerState
from models.base import VLMModel

class EvaluationRunner:
    """배치 추론 러너 (체크포인팅 지원)"""

    def __init__(self, config: EvaluationConfig, model: VLMModel):
        self.config = config
        self.model = model
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.state = self._load_or_create_state()

    def _load_or_create_state(self) -> RunnerState:
        if self.config.resume_from_checkpoint and self.checkpoint_path.exists():
            return RunnerState.load(self.checkpoint_path)
        return RunnerState()

    async def run(
        self,
        images: List[Image.Image],
        ground_truths: List[Any],
        prompts: List[str],
    ) -> List[InferenceResult]:
        """배치 추론 실행"""
        results = []

        # 완료된 인덱스 제외
        remaining = [
            i for i in range(len(images))
            if i not in self.state.completed
        ]

        # 배치 처리
        batch_size = self.config.batch_size
        progress = tqdm(range(0, len(remaining), batch_size), desc="Evaluating")

        for batch_start in progress:
            batch_indices = remaining[batch_start:batch_start + batch_size]
            batch_images = [images[i] for i in batch_indices]
            batch_prompts = [prompts[i] for i in batch_indices]
            batch_gts = [ground_truths[i] for i in batch_indices]

            try:
                start_time = time.time()

                # 비동기 지원 여부 확인
                if hasattr(self.model, 'run_async'):
                    predictions = await self.model.run_async(batch_prompts, batch_images)
                else:
                    predictions = self.model.run(batch_prompts, batch_images)

                latency = (time.time() - start_time) * 1000 / len(batch_indices)

                for idx, pred, gt in zip(batch_indices, predictions, batch_gts):
                    result = InferenceResult(
                        index=idx,
                        prediction=pred,
                        ground_truth=gt,
                        latency_ms=latency,
                    )
                    results.append(result)
                    self.state.completed.append(idx)
                    self.state.results.append(asdict(result))

                # 체크포인트 저장
                self.state.save(self.checkpoint_path)

            except Exception as e:
                for idx in batch_indices:
                    result = InferenceResult(
                        index=idx,
                        prediction="",
                        ground_truth=ground_truths[idx],
                        latency_ms=0,
                        error=str(e),
                    )
                    results.append(result)

        return results
```

#### 4.3 Main Evaluation Pipeline

**File**: `src/evaluation/pipeline.py`

```python
import os
import asyncio
from typing import List, Any, Dict, Optional
from pathlib import Path
from dataclasses import asdict
from datasets import load_dataset

from evaluation.config import EvaluationConfig, FormatType
from evaluation.types import EvaluationOutput, InferenceResult
from evaluation.runner import EvaluationRunner
from models.registry import create_model
from metrics import (
    evaluate_sentence_metrics,
    evaluate_table,
    evaluate_document,
    evaluate_markdown_metrics,
)

# 포맷별 기본 프롬프트
DEFAULT_PROMPTS = {
    FormatType.SENTENCE: "Extract all text from the image exactly as shown, including any typos.",
    FormatType.TABLE: "Extract the table from this image. Return as HTML table format.",
    FormatType.DOCUMENT: "Extract all text elements from this document image. Return as JSON.",
    FormatType.MARKDOWN: "Extract the markdown content from this image exactly as shown.",
}

class EvaluationPipeline:
    """평가 파이프라인 오케스트레이터"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = create_model(config.model)
        self.runner = EvaluationRunner(config, self.model)

    def _load_dataset(self):
        """HuggingFace 데이터셋 로드"""
        if self.config.subset == "default":
            return load_dataset(
                self.config.dataset_id,
                split=self.config.split
            )
        return load_dataset(
            self.config.dataset_id,
            name=self.config.subset,
            split=self.config.split,
        )

    def _get_prompt(self) -> str:
        """포맷 타입에 맞는 프롬프트 반환"""
        return self.config.prompt or DEFAULT_PROMPTS[self.config.format_type]

    def _extract_ground_truths(self, dataset) -> List[Any]:
        """포맷 타입에 따른 ground truth 추출"""
        format_type = self.config.format_type

        if format_type == FormatType.SENTENCE:
            return dataset["typo_text"]
        elif format_type == FormatType.TABLE:
            return [{"html": d.get("html", ""), "json": d.get("json", {})}
                    for d in dataset]
        elif format_type == FormatType.DOCUMENT:
            return [{"ground_truth": d.get("ground_truth", {})} for d in dataset]
        elif format_type == FormatType.MARKDOWN:
            return [d.get("markdown", "") for d in dataset]

        raise ValueError(f"Unknown format type: {format_type}")

    def _compute_metrics(self, results: List[InferenceResult]) -> Dict[str, float]:
        """포맷 타입에 따른 메트릭 계산"""
        # 에러 없는 결과만 필터링
        valid_results = [r for r in results if r.error is None]
        predictions = [r.prediction for r in valid_results]
        ground_truths = [r.ground_truth for r in valid_results]

        format_type = self.config.format_type

        if format_type == FormatType.SENTENCE:
            return evaluate_sentence_metrics(predictions, ground_truths)["metrics"]
        elif format_type == FormatType.TABLE:
            return evaluate_table(predictions, ground_truths)["metrics"]
        elif format_type == FormatType.DOCUMENT:
            return evaluate_document(predictions, ground_truths)["metrics"]
        elif format_type == FormatType.MARKDOWN:
            return evaluate_markdown_metrics(predictions, ground_truths)["metrics"]

        raise ValueError(f"Unknown format type: {format_type}")

    async def run(self) -> EvaluationOutput:
        """평가 파이프라인 실행"""
        # 데이터셋 로드
        dataset = self._load_dataset()

        # 샘플 수 제한
        if self.config.max_samples:
            dataset = dataset.select(
                range(min(self.config.max_samples, len(dataset)))
            )

        # 데이터 추출
        images = dataset["image"]
        ground_truths = self._extract_ground_truths(dataset)
        prompts = [self._get_prompt()] * len(images)

        # 추론 실행
        results = await self.runner.run(images, ground_truths, prompts)

        # 메트릭 계산
        metrics = self._compute_metrics(results)

        # 결과 반환
        return EvaluationOutput(
            config=asdict(self.config),
            metrics=metrics,
            per_sample_results=[asdict(r) for r in results],
            summary={
                "total_samples": len(results),
                "successful": len([r for r in results if r.error is None]),
                "failed": len([r for r in results if r.error is not None]),
                "avg_latency_ms": (
                    sum(r.latency_ms for r in results) / len(results)
                    if results else 0
                ),
            },
        )


def evaluate_pipeline(
    dataset_id: str,
    model_id: str,
    backend: str,
    format_type: str = "sentence",
    **kwargs,
) -> EvaluationOutput:
    """평가 파이프라인 편의 함수"""
    from evaluation.config import ModelConfig, InferenceBackend

    model_config = ModelConfig(
        model_id=model_id,
        backend=InferenceBackend(backend),
        api_key=kwargs.pop("api_key", None) or os.environ.get(f"{backend.upper()}_API_KEY"),
        **{k: v for k, v in kwargs.items() if k in ModelConfig.__fields__},
    )

    config = EvaluationConfig(
        dataset_id=dataset_id,
        format_type=FormatType(format_type),
        model=model_config,
        **{k: v for k, v in kwargs.items() if k in EvaluationConfig.__fields__},
    )

    pipeline = EvaluationPipeline(config)
    return asyncio.run(pipeline.run())
```

---

### Phase 5: Reporting and Comparison

**목표**: 리포트 생성 및 멀티 모델 비교

#### 5.1 Report Generator

**File**: `src/evaluation/report.py`

```python
from pathlib import Path
from typing import Optional
import json
from jinja2 import Environment, PackageLoader, select_autoescape
from evaluation.types import EvaluationOutput

class ReportGenerator:
    """평가 결과 리포트 생성"""

    def __init__(self, output: EvaluationOutput):
        self.output = output
        self.env = Environment(
            loader=PackageLoader('evaluation', 'templates'),
            autoescape=select_autoescape(['html']),
        )

    def to_json(self, path: Path) -> Path:
        """JSON 리포트 생성"""
        report = {
            "config": self.output.config,
            "metrics": self.output.metrics,
            "summary": self.output.summary,
            "per_sample_results": self.output.per_sample_results,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path

    def to_markdown(self, path: Path) -> Path:
        """Markdown 리포트 생성"""
        template = self.env.get_template('report.md.j2')
        content = template.render(
            config=self.output.config,
            metrics=self.output.metrics,
            summary=self.output.summary,
        )
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def to_html(self, path: Path) -> Path:
        """HTML 리포트 생성"""
        template = self.env.get_template('report.html.j2')
        # 에러 샘플 추출 (최대 20개)
        error_samples = [
            r for r in self.output.per_sample_results
            if r.get("error") is not None
        ][:20]

        content = template.render(
            config=self.output.config,
            metrics=self.output.metrics,
            summary=self.output.summary,
            error_samples=error_samples,
        )
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def save_all(self, output_dir: Path, prefix: str = "report") -> dict:
        """모든 포맷으로 저장"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "json": self.to_json(output_dir / f"{prefix}.json"),
            "markdown": self.to_markdown(output_dir / f"{prefix}.md"),
            "html": self.to_html(output_dir / f"{prefix}.html"),
        }
```

#### 5.2 Multi-Model Comparator

**File**: `src/evaluation/comparator.py`

```python
from typing import List, Dict, Any
from pathlib import Path
import json

class ModelComparator:
    """멀티 모델 비교"""

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results

    @classmethod
    def from_json_files(cls, paths: List[Path]) -> 'ModelComparator':
        """JSON 리포트 파일들로부터 생성"""
        results = []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        return cls(results)

    def to_dataframe(self):
        """비교 DataFrame 생성"""
        import pandas as pd

        rows = []
        for result in self.results:
            config = result["config"]
            model_config = config.get("model", {})

            row = {
                "model": model_config.get("model_id", "unknown"),
                "backend": model_config.get("backend", "unknown"),
                "format": config.get("format_type", "unknown"),
                "dataset": config.get("dataset_id", "unknown"),
                **result["metrics"],
                "samples": result["summary"]["total_samples"],
                "success_rate": (
                    result["summary"]["successful"] /
                    result["summary"]["total_samples"]
                ),
                "avg_latency_ms": result["summary"]["avg_latency_ms"],
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def to_markdown_table(self) -> str:
        """Markdown 비교 테이블 생성"""
        df = self.to_dataframe()
        return df.to_markdown(index=False)

    def rank_by_metric(self, metric: str, ascending: bool = True) -> List[str]:
        """메트릭 기준 모델 순위"""
        df = self.to_dataframe()
        sorted_df = df.sort_values(metric, ascending=ascending)
        return sorted_df["model"].tolist()

    def save_comparison(self, output_path: Path) -> Path:
        """비교 결과 저장"""
        df = self.to_dataframe()

        # JSON 저장
        df.to_json(output_path.with_suffix('.json'), orient='records', indent=2)

        # Markdown 저장
        with open(output_path.with_suffix('.md'), 'w') as f:
            f.write("# Model Comparison\n\n")
            f.write(df.to_markdown(index=False))

        return output_path
```

#### 5.3 Report Templates

**File**: `src/evaluation/templates/report.md.j2`

```markdown
# OCR Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | {{ config.model.model_id }} |
| Backend | {{ config.model.backend }} |
| Dataset | {{ config.dataset_id }} |
| Subset | {{ config.subset }} |
| Format | {{ config.format_type }} |

## Metrics

| Metric | Value |
|--------|-------|
{% for key, value in metrics.items() %}
| {{ key }} | {{ "%.4f"|format(value) }} |
{% endfor %}

## Execution Summary

- **Total Samples**: {{ summary.total_samples }}
- **Successful**: {{ summary.successful }}
- **Failed**: {{ summary.failed }}
- **Average Latency**: {{ "%.2f"|format(summary.avg_latency_ms) }} ms
```

---

### Phase 6: CLI Integration

**목표**: CLI 엔트리포인트 구현

#### 6.1 CLI Command

**File**: `src/evaluate.py` (업데이트)

```python
import click
import os
from pathlib import Path

@click.command()
@click.option('--model', '-m', required=True, help='Model ID')
@click.option('--backend', '-b', required=True,
              type=click.Choice(['openai', 'anthropic', 'google',
                                'transformers', 'vllm', 'sglang', 'ollama']))
@click.option('--dataset', '-d', required=True, help='HuggingFace dataset ID')
@click.option('--subset', default='default', help='Dataset subset')
@click.option('--format', 'format_type', default='sentence',
              type=click.Choice(['sentence', 'table', 'document', 'markdown']))
@click.option('--batch-size', default=1, help='Batch size')
@click.option('--max-samples', default=None, type=int, help='Max samples to evaluate')
@click.option('--output-dir', default='./evaluation_results', help='Output directory')
@click.option('--api-base', default=None, help='API base URL (for vLLM/SGLang server)')
@click.option('--tensor-parallel', default=1, help='Tensor parallel size (vLLM/SGLang)')
@click.option('--report-format', default='all',
              type=click.Choice(['json', 'markdown', 'html', 'all']))
def evaluate(model, backend, dataset, subset, format_type, batch_size,
             max_samples, output_dir, api_base, tensor_parallel, report_format):
    """VLM/OCR 모델 평가 실행"""
    from evaluation.config import EvaluationConfig, ModelConfig, FormatType, InferenceBackend
    from evaluation.pipeline import EvaluationPipeline
    from evaluation.report import ReportGenerator
    import asyncio

    # API 키 환경변수에서 가져오기
    api_key_map = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
    }
    api_key = os.environ.get(api_key_map.get(backend, ''))

    # 설정 생성
    model_config = ModelConfig(
        model_id=model,
        backend=InferenceBackend(backend),
        api_key=api_key,
        api_base=api_base,
        tensor_parallel_size=tensor_parallel,
    )

    config = EvaluationConfig(
        dataset_id=dataset,
        subset=subset,
        format_type=FormatType(format_type),
        model=model_config,
        batch_size=batch_size,
        max_samples=max_samples,
        output_dir=output_dir,
    )

    # 파이프라인 실행
    click.echo(f"Evaluating {model} on {dataset}...")
    pipeline = EvaluationPipeline(config)
    output = asyncio.run(pipeline.run())

    # 리포트 생성
    generator = ReportGenerator(output)
    output_path = Path(output_dir)

    if report_format == 'all':
        paths = generator.save_all(output_path)
        click.echo(f"Reports saved: {paths}")
    else:
        method = getattr(generator, f"to_{report_format}")
        path = method(output_path / f"report.{report_format}")
        click.echo(f"Report saved: {path}")

    # 결과 요약 출력
    click.echo("\n=== Results ===")
    click.echo(f"Samples: {output.summary['successful']}/{output.summary['total_samples']}")
    click.echo(f"Avg Latency: {output.summary['avg_latency_ms']:.2f}ms")
    click.echo("\nMetrics:")
    for key, value in output.metrics.items():
        click.echo(f"  {key}: {value:.4f}")


@click.command()
@click.argument('report_files', nargs=-1, type=click.Path(exists=True))
@click.option('--output', '-o', default='comparison', help='Output file prefix')
def compare(report_files, output):
    """여러 평가 결과 비교"""
    from evaluation.comparator import ModelComparator

    comparator = ModelComparator.from_json_files([Path(f) for f in report_files])
    comparator.save_comparison(Path(output))

    click.echo("\n=== Model Comparison ===")
    click.echo(comparator.to_markdown_table())


@click.group()
def cli():
    """VLM/OCR Evaluation Pipeline"""
    pass

cli.add_command(evaluate)
cli.add_command(compare)

if __name__ == '__main__':
    cli()
```

---

### Phase 7: Testing

**목표**: 단위/통합 테스트 작성 (80%+ 커버리지)

#### 7.1 Configuration Tests

**File**: `tests/evaluation/test_config.py`

```python
import pytest
from evaluation.config import EvaluationConfig, ModelConfig, FormatType, InferenceBackend

class TestModelConfig:
    def test_default_values(self):
        config = ModelConfig(model_id="gpt-4o", backend=InferenceBackend.OPENAI)
        assert config.temperature == 0.0
        assert config.max_tokens == 4096

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            ModelConfig(model_id="test", backend="invalid")

class TestEvaluationConfig:
    def test_valid_config(self):
        model = ModelConfig(model_id="gpt-4o", backend=InferenceBackend.OPENAI)
        config = EvaluationConfig(
            dataset_id="test/dataset",
            format_type=FormatType.SENTENCE,
            model=model,
        )
        assert config.batch_size == 1
        assert config.split == "test"
```

#### 7.2 Metrics Tests

**File**: `tests/evaluation/test_metrics.py`

```python
import pytest
from metrics import cer, wer, accuracy, normalize_text

class TestCER:
    @pytest.mark.parametrize("ref,hyp,expected", [
        ("hello", "hello", 0.0),
        ("hello", "hallo", 0.2),
        ("", "", 0.0),
        ("abc", "", 1.0),
    ])
    def test_cer_values(self, ref, hyp, expected):
        assert abs(cer(ref, hyp) - expected) < 0.001

class TestWER:
    @pytest.mark.parametrize("ref,hyp,expected", [
        ("hello world", "hello world", 0.0),
        ("hello world", "hello", 0.5),
    ])
    def test_wer_values(self, ref, hyp, expected):
        assert abs(wer(ref, hyp) - expected) < 0.001

class TestNormalization:
    def test_whitespace_normalization(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_lowercase(self):
        assert normalize_text("HELLO", lowercase=True) == "hello"
```

#### 7.3 Model Tests (Mocked)

**File**: `tests/models/test_api_models.py`

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from PIL import Image
from models.api.openai_vision import OpenAIVision
from evaluation.config import ModelConfig, InferenceBackend

@pytest.fixture
def mock_image():
    return Image.new('RGB', (100, 100), color='white')

class TestOpenAIVision:
    @pytest.mark.asyncio
    async def test_run_async_mocked(self, mock_image):
        with patch('openai.AsyncOpenAI') as mock_client:
            # Mock 설정
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "extracted text"
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            config = ModelConfig(
                model_id="gpt-4o",
                backend=InferenceBackend.OPENAI,
                api_key="test-key",
            )
            model = OpenAIVision(config)

            result = await model.run_async(["Extract text"], [mock_image])
            assert result == ["extracted text"]
```

---

## CLI Usage Examples

```bash
# OpenAI GPT-4o
python -m evaluate evaluate \
    --model gpt-4o \
    --backend openai \
    --dataset nero-nlp/synthetic-ocr-korean \
    --format sentence \
    --max-samples 100

# Claude Vision
python -m evaluate evaluate \
    --model claude-sonnet-4-20250514 \
    --backend anthropic \
    --dataset nero-nlp/synthetic-ocr-korean \
    --format table

# vLLM (로컬)
python -m evaluate evaluate \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --backend vllm \
    --dataset nero-nlp/synthetic-ocr-korean \
    --tensor-parallel 2

# SGLang Server
python -m evaluate evaluate \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --backend sglang \
    --api-base http://localhost:30000 \
    --dataset nero-nlp/synthetic-ocr-korean

# Ollama
python -m evaluate evaluate \
    --model llava-llama3 \
    --backend ollama \
    --dataset nero-nlp/synthetic-ocr-korean

# HuggingFace Transformers
python -m evaluate evaluate \
    --model microsoft/Florence-2-large \
    --backend transformers \
    --dataset nero-nlp/synthetic-ocr-korean

# 모델 비교
python -m evaluate compare \
    results/gpt4o/report.json \
    results/claude/report.json \
    results/qwen/report.json \
    --output comparison
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| API Rate Limits | Medium | Rate limiter 구현, 체크포인팅 |
| API Costs | High | max_samples 제한, 비용 추정 기능 |
| Async Complexity | Medium | sync wrapper 제공, 철저한 테스트 |
| Memory (Large Dataset) | Medium | 배치 처리, 스트리밍 데이터셋 |
| Model Output Parsing | Medium | 강건한 파싱, 폴백 로직 |
| Breaking Changes | Low | 하위 호환성 유지, 마이그레이션 가이드 |

---

## Success Criteria

- [ ] 7가지 추론 백엔드 모두 동작
- [ ] 4가지 포맷 타입 평가 지원
- [ ] CER, WER, TEDS, Accuracy 메트릭 계산
- [ ] 체크포인팅으로 중단 후 재시작 가능
- [ ] JSON/Markdown/HTML 리포트 생성
- [ ] 멀티 모델 비교 테이블 생성
- [ ] 80%+ 테스트 커버리지
- [ ] CLI 완전 동작

---

## Dependencies to Add

```toml
[project.optional-dependencies]
eval = [
    "pydantic>=2.0.0",
    "openai>=1.0.0",
    "anthropic>=0.25.0",
    "google-generativeai>=0.5.0",
    "tenacity>=8.0.0",
    "jinja2>=3.0.0",
    "aiohttp>=3.9.0",
    "aiolimiter>=1.1.0",
    "ollama>=0.3.0",
    "pandas>=2.0.0",
    "tabulate>=0.9.0",
]
vllm = ["vllm>=0.6.0"]
sglang = ["sglang[all]>=0.3.0"]
```
