# 설치 가이드

이 문서는 korean-ocr-bench (합성 OCR 이미지 생성기 및 VLM 평가 파이프라인) 설치 방법을 설명합니다.

## 사전 요구 사항

### Python 버전

- **Python 3.11 이상** 필수
- Python 버전 확인: `python --version`

### uv 패키지 매니저

이 프로젝트는 [uv](https://github.com/astral-sh/uv)를 패키지 매니저로 사용합니다.

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# pip으로 설치
pip install uv
```

설치 확인:

```bash
uv --version
```

## 기본 설치

프로젝트 클론 후 기본 의존성 설치:

```bash
git clone <repository-url>
cd korean-ocr-bench

# 기본 의존성만 설치 (이미지 생성 기능)
uv sync
```

기본 설치에는 다음 패키지가 포함됩니다:
- `torch`, `torchvision` - 딥러닝 프레임워크
- `pillow`, `opencv-python`, `scikit-image` - 이미지 처리
- `datasets`, `huggingface-hub` - HuggingFace 데이터셋
- `matplotlib`, `plotly` - 시각화
- `pydantic` - 데이터 검증

## 선택적 의존성

평가 파이프라인과 다양한 백엔드를 사용하려면 추가 의존성을 설치해야 합니다.

### eval - API 클라이언트

OpenAI, Anthropic, Google 등의 API를 통해 VLM을 평가할 때 필요합니다.

```bash
uv sync --extra eval
```

포함 패키지:
- `openai` - OpenAI API (GPT-4V 등)
- `anthropic` - Anthropic API (Claude 등)
- `google-generativeai` - Google API (Gemini 등)
- `tenacity` - 재시도 로직
- `aiohttp` - 비동기 HTTP 클라이언트
- `pandas`, `tabulate` - 결과 분석 및 표시

### transformers - HuggingFace Transformers

로컬에서 HuggingFace 모델을 실행할 때 필요합니다.

```bash
uv sync --extra eval --extra transformers
```

포함 패키지:
- `transformers` - HuggingFace Transformers 라이브러리

**지원 모델 예시:**
- Qwen2-VL
- LLaVA
- InternVL
- 기타 HuggingFace Hub의 VLM 모델

### ollama - Ollama 백엔드

[Ollama](https://ollama.ai/)를 통해 로컬 모델을 실행할 때 필요합니다.

```bash
uv sync --extra eval --extra ollama
```

포함 패키지:
- `ollama` - Ollama Python 클라이언트

**사전 설치 필요:**
```bash
# Ollama 설치 (https://ollama.ai/)
# 모델 다운로드
ollama pull llava
ollama pull llama3.2-vision
```

### vllm - vLLM 백엔드 (Linux CUDA 전용)

고성능 추론을 위한 [vLLM](https://github.com/vllm-project/vllm) 백엔드입니다.

```bash
uv sync --extra eval --extra vllm
```

포함 패키지:
- `vllm` - vLLM 추론 엔진

**요구 사항:**
- Linux 운영체제
- NVIDIA GPU (CUDA 지원)
- CUDA Toolkit

### sglang - SGLang 백엔드 (Linux CUDA 전용)

[SGLang](https://github.com/sgl-project/sglang)을 통한 고성능 추론 백엔드입니다.

```bash
uv sync --extra eval --extra sglang
```

포함 패키지:
- `sglang[all]` - SGLang 전체 패키지

**요구 사항:**
- Linux 운영체제
- NVIDIA GPU (CUDA 지원)
- CUDA Toolkit

## 중요 사항

### 백엔드 충돌 경고

**vllm, sglang, transformers는 의존성 충돌이 있습니다.**

이 백엔드들은 서로 다른 버전의 패키지를 요구하므로, **반드시 별도로 설치**해야 합니다.

```bash
# 잘못된 예시 - 절대 이렇게 하지 마세요!
uv sync --extra vllm --extra sglang --extra transformers  # 충돌 발생!

# 올바른 예시 - 하나씩 별도 설치
uv sync --extra eval --extra transformers  # transformers 백엔드 사용 시
# 또는
uv sync --extra eval --extra vllm         # vLLM 백엔드 사용 시
# 또는
uv sync --extra eval --extra sglang       # SGLang 백엔드 사용 시
```

### 플랫폼 제한

| 백엔드 | macOS | Linux (CPU) | Linux (CUDA) |
|--------|-------|-------------|--------------|
| eval | O | O | O |
| transformers | O | O | O |
| ollama | O | O | O |
| vllm | X | X | O |
| sglang | X | X | O |

- vLLM과 SGLang은 **Linux + CUDA** 환경에서만 동작합니다.
- macOS나 Windows에서는 `transformers`, `ollama`, 또는 API 백엔드를 사용하세요.

## 설치 예시

### 시나리오별 설치 명령어

```bash
# 1. 이미지 생성만 사용 (평가 없음)
uv sync

# 2. API 평가 (OpenAI, Anthropic, Google)
uv sync --extra eval

# 3. 로컬 HuggingFace 모델 평가
uv sync --extra eval --extra transformers

# 4. Ollama 로컬 모델 평가
uv sync --extra eval --extra ollama

# 5. vLLM 고성능 평가 (Linux CUDA)
uv sync --extra eval --extra vllm

# 6. SGLang 고성능 평가 (Linux CUDA)
uv sync --extra eval --extra sglang

# 7. API + Ollama 조합
uv sync --extra eval --extra ollama
```

### 가상 환경 활성화

uv는 자동으로 `.venv` 디렉토리에 가상 환경을 생성합니다.

```bash
# 가상 환경 활성화 (필요시)
source .venv/bin/activate  # Linux/macOS
# 또는
.venv\Scripts\activate     # Windows

# uv run으로 직접 실행 (활성화 불필요)
uv run python main.py
uv run evaluate --help
```

## 문제 해결

### 일반적인 문제

#### 1. Python 버전 오류

```
error: Requires Python >=3.11
```

**해결:**
```bash
# pyenv로 Python 3.11+ 설치
pyenv install 3.11
pyenv local 3.11

# 또는 uv로 Python 설치
uv python install 3.11
```

#### 2. CUDA 관련 오류 (vLLM/SGLang)

```
CUDA not available
```

**해결:**
- NVIDIA 드라이버 설치 확인
- CUDA Toolkit 설치 확인
- `nvidia-smi` 명령으로 GPU 인식 확인

```bash
nvidia-smi  # GPU 상태 확인
nvcc --version  # CUDA 버전 확인
```

#### 3. 의존성 충돌

```
Conflicting dependencies detected
```

**해결:**
```bash
# 기존 환경 제거 후 재설치
rm -rf .venv uv.lock
uv sync --extra eval --extra <원하는_백엔드>
```

#### 4. torch 설치 실패 (macOS Apple Silicon)

```
No matching distribution found for torch
```

**해결:**
```bash
# PyTorch 공식 문서에 따라 설치
# https://pytorch.org/get-started/locally/
```

#### 5. Ollama 연결 실패

```
Connection refused to localhost:11434
```

**해결:**
```bash
# Ollama 서비스 실행 확인
ollama serve  # 서버 시작

# 또는 Ollama 앱 실행 (macOS)
```

### 도움 요청

문제가 해결되지 않으면:
1. GitHub Issues에 문제 보고
2. 오류 메시지 전문 포함
3. 운영체제, Python 버전, CUDA 버전 명시

## 설치 확인

설치가 완료되면 다음 명령으로 확인:

```bash
# 기본 기능 테스트
uv run python -c "from src.generator import ImageGenerator; print('OK')"

# 평가 CLI 확인 (eval 설치 시)
uv run evaluate --help
```
