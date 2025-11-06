import sys
import json
import numpy as np
from datasets import load_dataset

# 'src'를 경로에 추가하여 'metrics.edit_distance' 모듈을 임포트합니다.
sys.path.insert(0, "src")
from metrics.edit_distance import cer


def analysis(dataset_id, split="train"):
    """
    데이터셋을 로드하고 평균 CER 및 CER의 표준 편차를 계산합니다.

    Args:
        dataset_id (str): 로드할 데이터셋의 ID.
        split (str): 사용할 데이터셋의 분할 (기본값: "train").

    Returns:
        tuple: 평균 CER과 CER의 표준 편차를 포함하는 튜플.
    """
    print(f"\nAnalyzing dataset: {dataset_id}...")
    dataset = load_dataset(dataset_id, split=split)
    # print(dataset) # 데이터셋 정보가 매우 길 수 있으므로 주석 처리합니다.

    typo = dataset["typo_text"]
    original = dataset["original_text"]
    ocr_result = dataset["ocr_result"]

    # cer 함수를 직접 사용하여 CER 목록을 다시 계산합니다.
    cer_list = [cer(y_gt, y_pred) for y_gt, y_pred in zip(typo, ocr_result)]

    # CER이 1보다 큰 값을 포함하여 모든 값을 사용합니다.
    cer_list_filtered = [elem for elem in cer_list]

    # 평균을 계산합니다.
    avg = sum(cer_list_filtered) / len(cer_list_filtered) if cer_list_filtered else 0

    # 표준 편차를 계산합니다.
    std = np.std(cer_list_filtered) if cer_list_filtered else 0

    num = [(1 if typo[i] == original[i] else 0) for i in range(len(typo))]
    print(f"Number of samples where typo and original are identical: {sum(num)}")
    print(f"avg CER: {avg:.4f}")
    print(f"std CER: {std:.4f}")

    return avg, std


def run_all():
    """
    미리 정의된 모델 및 데이터셋 목록에 대한 분석을 실행하고,
    요약 테이블을 출력하며, 결과를 JSON 파일에 저장합니다.
    """
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
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "NCSOFT/VARCO-VISION-2.0-1.7B-OCR",
    ]

    base_url = "junyeong-nero/synthetic-ocr-images-korean-"
    dataset_list = [base_url + name.split("/")[-1] for name in names]
    print("List of datasets to analyze:")
    print(dataset_list)

    results = []
    for name, dataset_id in zip(names, dataset_list):
        avg_cer, std_cer = analysis(dataset_id)
        # 데이터셋 이름, 평균, 표준 편차를 결과 목록에 추가합니다.
        results.append(
            {
                "model": name,
                "dataset": dataset_id,
                "avg_cer": avg_cer,
                "std_cer": std_cer,
            }
        )

    # --- avg_cer을 기준으로 결과를 오름차순으로 정렬 ---
    results.sort(key=lambda x: x["avg_cer"])

    # --- 결과 테이블 출력 ---
    print("\n\n--- Final Results Summary ---")

    # 헤더 출력
    header = f"| {'Model':<65} | {'Avg CER':<10} | {'Std CER':<10} |"
    separator = f"|{'-'*67}|{'-'*12}|{'-'*12}|"
    print(header)
    print(separator)

    # 각 데이터셋의 결과 출력
    for result in results:
        row = f"| {result['model']:<65} | {result['avg_cer']:.6f}   | {result['std_cer']:.6f}   |"
        print(row)

    # --- 결과를 JSON 파일로 저장 ---
    output_filename = "analysis_results.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\nResults successfully saved to {output_filename}.")
    except Exception as e:
        print(f"\nAn error occurred while saving the JSON file: {e}")


if __name__ == "__main__":

    run_all()
