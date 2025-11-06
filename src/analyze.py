import sys
import json
import numpy as np
from datasets import load_dataset

# Add 'src' to the path to import the 'metrics.edit_distance' module.
sys.path.insert(0, "src")
from metrics.edit_distance import cer


def analysis(dataset_id, split="train"):
    """
    Loads a dataset and calculates the average CER and standard deviation of CER.

    Args:
        dataset_id (str): The ID of the dataset to load.
        split (str): The split of the dataset to use (default: "train").

    Returns:
        tuple: A tuple containing the average CER and the standard deviation of CER.
    """
    print(f"\nAnalyzing dataset: {dataset_id}...")
    dataset = load_dataset(dataset_id, split=split)
    # print(dataset) # This is commented out as the dataset info can be very long.

    typo = dataset["typo_text"]
    original = dataset["original_text"]
    ocr_result = dataset["ocr_result"]

    # Recalculate the CER list using the cer function directly.
    cer_list = [cer(y_gt, y_pred) for y_gt, y_pred in zip(typo, ocr_result)]

    # Use all values, including those with CER greater than 1.
    cer_list_filtered = [elem for elem in cer_list]

    # Calculate the average
    avg = sum(cer_list_filtered) / len(cer_list_filtered) if cer_list_filtered else 0

    # Calculate the standard deviation
    std = np.std(cer_list_filtered) if cer_list_filtered else 0

    num = [(1 if typo[i] == original[i] else 0) for i in range(len(typo))]
    print(f"Number of samples where typo and original are identical: {sum(num)}")
    print(f"avg CER: {avg:.4f}")
    print(f"std CER: {std:.4f}")

    return avg, std


def run_all():
    """
    Runs the analysis for a predefined list of models and datasets,
    prints a summary table, and saves the results to a JSON file.
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
        "NCSOFT/VARCO-VISION-2.0-1.7B-OCR",
    ]

    base_url = "junyeong-nero/synthetic-ocr-images-korean-"
    dataset_list = [base_url + name.split("/")[-1] for name in names]
    print("List of datasets to analyze:")
    print(dataset_list)

    results = []
    for name, dataset_id in zip(names, dataset_list):
        avg_cer, std_cer = analysis(dataset_id)
        # Add dataset name, average, and standard deviation to the results list.
        results.append(
            {
                "model": name,
                "dataset": dataset_id,
                "avg_cer": avg_cer,
                "std_cer": std_cer,
            }
        )

    # --- Print Results Table ---
    print("\n\n--- Final Results Summary ---")

    # Print header
    header = f"| {'Model':<65} | {'Avg CER':<10} | {'Std CER':<10} |"
    separator = f"|{'-'*67}|{'-'*12}|{'-'*12}|"
    print(header)
    print(separator)

    # Print results for each dataset
    for result in results:
        row = f"| {result['model']:<65} | {result['avg_cer']:.6f}   | {result['std_cer']:.6f}   |"
        print(row)

    # --- Save Results to JSON File ---
    output_filename = "analysis_results.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\nResults successfully saved to {output_filename}.")
    except Exception as e:
        print(f"\nAn error occurred while saving the JSON file: {e}")


if __name__ == "__main__":

    run_all()
