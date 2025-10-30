# Korean-OCR-bench

한글 OCR(광학 문자 인식) 모델의 성능을 벤치마킹하고 평가하기 위한 도구입니다. 이 프로젝트는 다양한 유형의 질문(Question Types)을 포함하는 벤치마크 데이터셋을 생성하고, 이를 통해 OCR 모델이 실제 시나리오에서 얼마나 잘 작동하는지 측정하는 것을 목표로 합니다.

## 주요 기능:

*   **한국어 벤치마크 데이터셋 생성**: 다양한 복잡성과 스타일을 가진 한국어 텍스트 데이터를 생성하여 OCR 모델 평가에 활용합니다.
*   **유연한 설정**: `config` 디렉토리의 파일을 통해 벤치마크 데이터셋 생성 설정을 조정할 수 있습니다.
*   **모델 성능 평가**: 생성된 데이터셋을 사용하여 OCR 모델의 정확도, 견고성 등을 평가합니다.

## 사용법

### 1. 환경 설정

프로젝트를 실행하기 전에 필요한 의존성을 설치해야 합니다. `uv`를 사용하여 의존성을 관리합니다.

```bash
uv sync
```

### 2. 데이터 생성

`main.py` 스크립트를 사용하여 벤치마크 데이터를 생성할 수 있습니다.

```bash
python main.py --config_path config/config_ko.yaml
```

*   `--config_path`: 사용할 설정 파일의 경로를 지정합니다. `config/config_ko.yaml` (한국어) 또는 `config/config_ja.yaml` (일본어) 중 하나를 선택할 수 있습니다.

### 3. 설정 파일 수정

`config` 디렉토리 내의 YAML 파일을 수정하여 데이터 생성 방식을 세부적으로 제어할 수 있습니다. 예를 들어, 생성할 데이터의 양, 텍스트 스타일, 이미지 배경 등을 설정할 수 있습니다.

## 개발 및 기여

프로젝트에 기여하려면 다음 단계를 따르세요:

1.  저장소를 포크(Fork)합니다.
2.  새로운 브랜치를 생성합니다 (`git checkout -b feature/your-feature-name`).
3.  변경 사항을 커밋합니다 (`git commit -m 'Add some feature'`).
4.  원격 저장소에 푸시합니다 (`git push origin feature/your-feature-name`).
5.  풀 리퀘스트(Pull Request)를 생성합니다.

# Question Types

```
{
    "Key Information Extraction",
    "Handwriting Recognition",
    "Scene Text-centric VQA",
    "Handwritten Mathematical Expression Recognition",
    "Irregular Text Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
    "Artistic Text Recognition",
    "Doc-oriented VQA",
    "Regular Text Recognition",
}
```

# Reference

- HF: [link](https://huggingface.co/datasets/echo840/OCRBench)