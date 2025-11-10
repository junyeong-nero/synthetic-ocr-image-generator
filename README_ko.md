# 합성 OCR 이미지 생성기

이 프로젝트는 합성 OCR 이미지 생성기를 제공합니다.

- [Huggingface Dataset](https://huggingface.co/datasets/junyeong-nero/synthetic-ocr-images-korean)

# 사용 방법

## 환경 설정

`uv`를 사용하여 환경을 설정합니다:

```shell
uv sync
```

## 스크립트 실행

합성 OCR 이미지를 생성하려면 `scripts/generate.sh`를 통해 `main.py` 스크립트를 실행합니다:

```python
# scripts/generate.sh
uv run main.py \
    --lang ko \
    --font-path "fonts/NotoSans-VariableFont_wdth,wght.ttf" \
    --repo-id "junyeong-nero/synthetic-ocr-images-korean" \
    --corpus-size 10000 \
    --size 1000 \
    --typo-ratio 0.4
```

### 매개변수:

- `lang`: 텍스트 생성에 사용할 언어를 지정합니다.
- `font-path`: 문자 수준 유사성 계산에 사용되는 글꼴 디렉토리 경로입니다.
- `repo-id`: 업데이트할 Hugging Face 저장소 ID입니다.
- `corpus-size`: [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) 데이터셋에서 가져온 코퍼스에 대해 생성할 문장 수입니다.
- `size`: 데이터셋에 대해 생성할 합성 이미지의 총 개수입니다.
- `typo-ratio`: 생성된 텍스트에 도입할 오타의 비율입니다.

# 평가

평가를 위해 vLLM과 Transformers를 활용했습니다. 그러나 이 프로젝트의 `uv` 환경은 다양한 OCR 모델의 상이한 설정 요구 사항(예: 특정 PyTorch 및 CUDA 버전)으로 인해 직접적인 평가를 지원하지 않습니다.

다른 OCR 모델과의 통합 및 추론에 대한 자세한 내용은 `src/models`를 참조하십시오.

평가 스크립트 예시:

```
uv run src/evaluate.py \
    "allenai/olmOCR-2-7B-1025" \
    "junyeong-nero/synthetic-ocr-images-korean" \
    --target-column "typo_text" \
    --prompt "Extract all text from the image verbatim, including typos, without translation or character modification." \
    --output-dataset-id "junyeong-nero/synthetic-ocr-images-korean-olmOCR-2-7B-1025" \
    --batchsize 8
```

# 결과

DeepSeek-OCR을 사용하려고 시도했지만, 모델이 대상 언어와 일치하지 않는 반복적이고 무의미한 문자(예: "號號號號號...")를 생성했습니다.

아래 결과는 한국어 텍스트로 수행된 평가를 기반으로 합니다.

| 모델                                                             | 평균 CER    | 표준 CER    |
|-------------------------------------------------------------------|------------|------------|
| allenai/olmOCR-2-7B-1025                                          | 0.159544   | 2.159467   |
| Qwen/Qwen3-VL-2B-Instruct                                         | 0.191162   | 2.157042   |
| Qwen/Qwen3-VL-4B-Instruct                                         | 0.259124   | 2.964853   |
| nanonets/Nanonets-OCR2-3B                                         | 0.267985   | 4.309995   |
| Qwen/Qwen3-VL-8B-Instruct                                         | 0.290215   | 4.032342   |
| NCSOFT/VARCO-VISION-2.0-1.7B-OCR                                  | 0.398493   | 0.270318   |
| PaddlePaddle/PaddleOCR-VL                                         | 0.494337   | 8.531293   |
| google/gemma-3-4b-it                                              | 0.997308   | 7.249212   |
| rednote-hilab/dots.ocr                                            | 1.988376   | 15.208363   |
| stepfun-ai/GOT-OCR-2.0-hf                                         | 6.497117   | 16.651408   |

# 향후 연구 방향

* **범위 확장**: 기본적인 텍스트 인식 단계를 넘어, 문서 단위 OCR 및 핵심 정보 추출(KIE)에 대한 증가하는 수요를 해결하는 것을 목표로 한다.
* **대상 데이터 유형**: 보다 복잡하고 다양한 합성 이미지를 생성하는 데 초점을 맞출 예정이다