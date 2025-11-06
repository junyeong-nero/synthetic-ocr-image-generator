import argparse
from datasets import load_dataset
from tqdm import tqdm
from models.base import Model
from models import (
    DotsOCR,
    NanonetsOCR,
    LightOnOCR,
    OlmOCR,
    DeepSeekOCR,
    Gemma3_4B_IT,
    GotOCR,
    PaddleOCR,
    Qwen3VL,
    VarcoOCR,
)

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
    """
    Evaluates an OCR model on a given dataset.

    Args:
        model_id (str): The ID of the model to use for evaluation.
        dataset_id (str): The ID of the dataset to evaluate on.
        subset (str): The subset of the dataset to use.
        split (str): The split of the dataset to use.
        batchsize (int): The batch size for processing.
        output_dataset_id (str): The ID to use for pushing the results to the Hugging Face Hub.
        image_column (str): The name of the column containing the images.
        prompt (str): The prompt to use for the OCR model.
        target_column (str): The name of the column containing the ground truth text.

    Returns:
        list: A list of the OCR results.
    """

    print(f"Load Models: {model_id}")
    model = MODELS[model_id]()  # Instantiate the model

    print(f"Load Dataset: {dataset_id}, {subset}, {split}")
    dataset = load_dataset(dataset_id, split=split)
    print(dataset)

    output = []
    cer_list = []

    # Use tqdm to display a progress bar.
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
