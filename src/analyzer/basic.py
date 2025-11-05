import sys
import json
import numpy as np
from datasets import load_dataset

# 'src' 경로를 추가하여 'metrics.edit_distance' 모듈을 임포트할 수 있도록 합니다.
sys.path.insert(0, "src")
from metrics.edit_distance import cer


def analysis(dataset_id, split="train"):
    """
    데이터셋을 로드하여 평균 CER과 CER의 표준 편차를 계산합니다.

    Args:
        dataset_id (str): 로드할 데이터셋의 ID.
        split (str): 사용할 데이터셋의 스플릿 (기본값: "train").

    Returns:
        tuple: 평균 CER과 CER의 표준 편차를 담은 튜플.
    """
    print(f"\nAnalyzing dataset: {dataset_id}...")
    dataset = load_dataset(dataset_id, split=split)
    # print(dataset) # 데이터셋 정보 출력이 너무 길어질 수 있으므로 주석 처리합니다. 필요시 활성화하세요.

    typo = dataset["typo_text"]
    original = dataset["original_text"]
    ocr_result = dataset["ocr_result"]

    # 직접 cer 함수를 사용하여 CER 리스트를 다시 계산합니다.
    cer_list = [cer(y_gt, y_pred) for y_gt, y_pred in zip(typo, ocr_result)]

    # 1보다 큰 CER 값을 포함하여 모든 값을 사용합니다.
    cer_list_filtered = [elem for elem in cer_list]

    # 평균 계산
    avg = sum(cer_list_filtered) / len(cer_list_filtered) if cer_list_filtered else 0

    # 표준 편차 계산
    std = np.std(cer_list_filtered) if cer_list_filtered else 0

    num = [(1 if typo[i] == original[i] else 0) for i in range(len(typo))]
    print(f"오타와 원문이 동일한 샘플 수: {sum(num)}")
    print(f"avg CER: {avg:.4f}")  # 소수점 4자리까지 표시
    print(f"std CER: {std:.4f}")  # 소수점 4자리까지 표시

    return avg, std


if __name__ == "__main__":

    names = [
        "rednote-hilab/dots.ocr",
        "nanonets/Nanonets-OCR2-3B",
        # "lightonai/LightOnOCR-1B-1025",
        "allenai/olmOCR-2-7B-1025",
        # "deepseek-ai/DeepSeek-OCR",
        "google/gemma-3-4b-it",
        "stepfun-ai/GOT-OCR-2.0-hf",
        "PaddlePaddle/PaddleOCR-VL",
        "Qwen/Qwen3-VL-2B-Instruct",
        "NCSOFT/VARCO-VISION-2.0-1.7B-OCR",
    ]

    base_url = "junyeong-nero/synthetic-ocr-images-korean-"
    dataset_list = [base_url + name.split("/")[-1] for name in names]
    print("분석할 데이터셋 목록:")
    print(dataset_list)

    results = []
    for dataset_id in dataset_list:
        avg_cer, std_cer = analysis(dataset_id)
        # 결과 리스트에 데이터셋 이름, 평균, 표준편차를 추가합니다.
        results.append({"dataset": dataset_id, "avg_cer": avg_cer, "std_cer": std_cer})

    # --- 결과 테이블 출력 ---
    print("\n\n--- 최종 결과 요약 ---")

    # 헤더 출력
    header = f"| {'Dataset':<65} | {'Avg CER':<10} | {'Std CER':<10} |"
    separator = f"|{'-'*67}|{'-'*12}|{'-'*12}|"
    print(header)
    print(separator)

    # 각 데이터셋의 결과 출력
    for result in results:
        row = f"| {result['dataset']:<65} | {result['avg_cer']:.6f}   | {result['std_cer']:.6f}   |"
        print(row)

    # --- JSON 파일로 결과 저장 ---
    output_filename = "analysis_results.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\n결과가 {output_filename} 파일에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"\nJSON 파일 저장 중 오류가 발생했습니다: {e}")
