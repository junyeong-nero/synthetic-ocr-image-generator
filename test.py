import numpy as np
from datasets import load_dataset
from src.metrics.edit_distance import cer

dataset = load_dataset(
    "junyeong-nero/synthetic-ocr-images-korean-PaddleOCR-VL", split="train"
)
print(dataset)

typo = dataset["typo_text"]
original = dataset["original_text"]
ocr_result = [text.split("Assistant:")[-1].strip() for text in dataset["ocr_result"]]

cer_list = [cer(y_gt, y_pred) for y_gt, y_pred in zip(typo, ocr_result)]

dataset

dataset = dataset.remove_columns(["cer", "ocr_result"])
dataset = dataset.add_column("cer", cer_list)
dataset = dataset.add_column("ocr_result", ocr_result)

dataset.push_to_hub("junyeong-nero/synthetic-ocr-images-korean-PaddleOCR-VL")

avg = sum(cer_list) / len(cer_list)
std = np.std(cer_list)

num = [(1 if typo[i] == original[i] else 0) for i in range(len(typo))]
