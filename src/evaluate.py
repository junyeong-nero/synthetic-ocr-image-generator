import argparse
from datasets import load_dataset
from tqdm import tqdm  # tqdm 라이브러리 추가
from models.base import Model
from models import DotsOCR, NanonetsOCR, LightOnOCR, OlmOCR
from models.transformers_model.deepseek_ocr import DeepSeekOCR
from models.transformers_model.gemma3_4b_it import Gemma3_4B_IT
from models.transformers_model.got_ocr import GotOCR
from models.transformers_model.paddle_ocr import PaddleOCR
from models.transformers_model.qwen3_vl import Qwen3VL
from models.transformers_model.varco_ocr import VarcoOCR

from metrics.edit_distance import cer

MODELS = {
    "dummy": Model,
    "rednote-hilab/dots.ocr": DotsOCR,
    "nanonets/Nanonets-OCR2-3B": NanonetsOCR,
    "lightonai/LightOnOCR-1B-1025": LightOnOCR,
    "allenai/olmOCR-2-7B-1025": OlmOCR,
    "deepseek-ai/DeepSeek-OCR": DeepSeekOCR,
    "google/gemma-3-4b-it": Gemma3_4B_IT,
    "stepfun-ai/GOT-OCR-2.0-hf": GotOCR,
    "PaddlePaddle/PaddleOCR-VL": PaddleOCR,
    "Qwen/Qwen3-VL-2B-Instruct": Qwen3VL,
    "NCSOFT/VARCO-VISION-2.0-1.7B-OCR": VarcoOCR,
}


def main(
    model_id,
    dataset_id,
    subset,
    split,
    batchsize,
    output_dataset_id,
    image_column="image",
    prompt="OCR this image",
    target_column="response",
):

    print(f"Load Models: {model_id}")
    model = MODELS[model_id]()  # Instantiate the model

    print(f"Load Dataset: {dataset_id}, {subset}, {split}")
    dataset = load_dataset(dataset_id, split=split)
    print(dataset)

    output = []
    cer_list = []

    # tqdm을 사용하여 진행 상황을 표시합니다.
    for i in tqdm(range(0, len(dataset), batchsize), desc="Processing Batches"):
        batch = dataset[i : i + batchsize]

        batch_images = batch[image_column]
        batch_prompts = batch[prompt] if prompt in batch else [prompt] * batchsize
        batch_gt = batch[target_column]
        batch_result = model.run(prompts=batch_prompts, images=batch_images)

        cer_list += [cer(y_gt, y_pred) for y_gt, y_pred in zip(batch_gt, batch_result)]
        output += batch_result

    if output_dataset_id:
        dataset = dataset.add_column("cer", cer_list)
        dataset = dataset.add_column("ocr_result", output)
        print(dataset)
        dataset.push_to_hub(output_dataset_id)

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, default="rednote-hilab/dots.ocr")
    parser.add_argument(
        "dataset_id", type=str, default="junyeong-nero/synthetic-ocr-images-korean"
    )
    parser.add_argument("--subset", type=str, default="default")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument(
        "--output-dataset-id",
        type=str,
        default=None,
    )
    parser.add_argument("--image-column", type=str, default="image")
    parser.add_argument("--target-column", type=str, default="response")
    parser.add_argument("--prompt", type=str, default="OCR this image")

    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)
