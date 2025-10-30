# Korean-OCR-bench

한글 OCR(광학 문자 인식) 모델의 성능을 벤치마킹하고 평가하기 위한 도구입니다. 이 프로젝트는 다양한 유형의 질문(Question Types)을 포함하는 벤치마크 데이터셋을 생성하고, 이를 통해 OCR 모델이 실제 시나리오에서 얼마나 잘 작동하는지 측정하는 것을 목표로 합니다.

## 주요 기능:

*   **한국어 벤치마크 데이터셋 생성**: 다양한 복잡성과 스타일을 가진 한국어 텍스트 데이터를 생성하여 OCR 모델 평가에 활용합니다.
*   **유연한 설정**: `config` 디렉토리의 파일을 통해 벤치마크 데이터셋 생성 설정을 조정할 수 있습니다.
*   **모델 성능 평가**: 생성된 데이터셋을 사용하여 OCR 모델의 정확도, 견고성 등을 평가합니다.

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

- HF: (link)[https://huggingface.co/datasets/echo840/OCRBench]