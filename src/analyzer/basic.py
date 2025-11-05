import numpy as np
from datasets import load_dataset


def analysis(dataset_id, split="train"):
    """
    데이터셋을 로드하여 평균 CER과 CER의 표준 편차를 계산합니다.

    Args:
        dataset_id (str): 로드할 데이터셋의 ID.
        split (str): 사용할 데이터셋의 스플릿 (기본값: "train").

    Returns:
        tuple: 평균 CER과 CER의 표준 편차를 담은 튜플.
    """
    dataset = load_dataset(dataset_id, split=split)
    print(dataset)

    typo = dataset["typo_text"]
    original = dataset["original_text"]
    cer_list = dataset["cer"]
    cer_list = [elem for elem in cer_list if 0 <= elem <= 1]

    # 평균 계산
    avg = sum(cer_list) / len(cer_list)

    # 표준 편차 계산
    std = np.std(cer_list)

    num = [(1 if typo[i] == original[i] else 0) for i in range(len(typo))]
    print(f"오타와 원문이 동일한 샘플 수: {sum(num)}")
    print(f"dataset: {dataset_id}")
    print(f"avg CER: {avg}")
    print(f"std CER: {std}")

    return avg, std


if __name__ == "__main__":

    dataset_list = [
        "junyeong-nero/synthetic-ocr-images-korean-dots.ocr",
        "junyeong-nero/synthetic-ocr-images-korean-Nanonets-OCR2-3B",
        "junyeong-nero/synthetic-ocr-images-korean-olmOCR-2-7B-1025",
        "junyeong-nero/synthetic-ocr-images-korean-PaddleOCR-VL",
    ]
    for dataset_id in dataset_list:
        analysis(dataset_id)
