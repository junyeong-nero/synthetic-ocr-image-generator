import argparse
from datasets import load_dataset
from evaluator.model import DotsOCR

MODELS = {"rednote-hilab/dots.ocr": DotsOCR}


def main(model_id, hf_dataset_id, subset, split, batchsize, output_dataset_id):

    print(f"Load Models: {model_id}")
    model = MODELS[model_id]

    print(f"Load Dataset: {hf_dataset_id}, {subset}, {split}")
    dataset = load_dataset(hf_dataset_id, subset, split=split)

    output = []

    for i in range(0, len(dataset), batchsize):
        batch = dataset[i : i + batchsize]

        batch_images = batch["image"]
        batch_prompts = batch["prompts"]
        batch_responses = batch["response"]

        result = model.run(prompts=batch_prompts, images=batch_images)
        output += result

    dataset.add_column("output", output)
    if output_dataset_id:
        dataset.push_to_hub(output_dataset_id)

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, default="rednote-hilab/dots.ocr")
    parser.add_argument(
        "dataset_id", type=str, default="junyeong-nero/synthetic-ocr-images-korean"
    )
    parser.add_argument("--subset", type=str, default="sentence")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument(
        "--output-dataset-id",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)
