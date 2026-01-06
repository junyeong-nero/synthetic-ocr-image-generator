import argparse
import json
from typing import Dict, Any, Optional

import numpy as np
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
from metrics.table_document_metrics import evaluate_table, evaluate_document

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

DEFAULT_PROMPTS = {
    "sentence": "Extract all text from the image verbatim, including typos, without translation or character modification.",
    "table": "Extract the table from this image. Return the result as HTML table format.",
    "document": "Extract all text elements from this document image. Return as JSON with 'elements' array containing objects with 'type', 'text', 'bounding_box', and 'reading_order' fields.",
}


def parse_model_output_as_json(output: str) -> Optional[Dict]:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    if "```json" in output:
        start = output.find("```json") + 7
        end = output.find("```", start)
        if end > start:
            try:
                return json.loads(output[start:end].strip())
            except json.JSONDecodeError:
                pass

    if "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        if end > start:
            try:
                return json.loads(output[start:end].strip())
            except json.JSONDecodeError:
                pass

    return None


def extract_html_table(output: str) -> str:
    output = output.strip()

    if "<table" in output.lower():
        start = output.lower().find("<table")
        end = output.lower().find("</table>")
        if end > start:
            return output[start:end + 8]

    if "```html" in output:
        start = output.find("```html") + 7
        end = output.find("```", start)
        if end > start:
            return output[start:end].strip()

    if "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        if end > start:
            content = output[start:end].strip()
            if "<table" in content.lower():
                return content

    return output


def evaluate_sentence(
    model,
    dataset,
    batchsize: int,
    image_column: str,
    target_column: str,
    prompt: str,
) -> Dict[str, Any]:
    output = []
    cer_list = []

    for i in tqdm(range(0, len(dataset), batchsize), desc="Processing Sentences"):
        batch = dataset[i : i + batchsize]
        batch_images = batch[image_column]
        batch_prompts = batch[prompt] if prompt in batch else [prompt] * len(batch_images)
        batch_gt = batch[target_column]
        batch_result = model.run(prompts=batch_prompts, images=batch_images)

        cer_list += [cer(y_gt, y_pred) for y_gt, y_pred in zip(batch_gt, batch_result)]
        output += batch_result

    return {
        "predictions": output,
        "cer_list": cer_list,
        "metrics": {
            "avg_cer": float(np.mean(cer_list)),
            "std_cer": float(np.std(cer_list)),
            "min_cer": float(np.min(cer_list)),
            "max_cer": float(np.max(cer_list)),
        },
    }


def evaluate_table_format(
    model,
    dataset,
    batchsize: int,
    image_column: str,
    prompt: str,
) -> Dict[str, Any]:
    predictions = []
    teds_scores = []
    cell_accuracies = []
    structure_f1_scores = []

    for i in tqdm(range(0, len(dataset), batchsize), desc="Processing Tables"):
        batch = dataset[i : i + batchsize]
        batch_images = batch[image_column]
        batch_prompts = [prompt] * len(batch_images)
        batch_result = model.run(prompts=batch_prompts, images=batch_images)

        for j, result in enumerate(batch_result):
            idx = i + j
            pred_html = extract_html_table(result)
            pred_json = parse_model_output_as_json(result) or {}

            true_html = dataset[idx].get("html", "")
            true_json_str = dataset[idx].get("json", "{}")
            true_json = json.loads(true_json_str) if isinstance(true_json_str, str) else true_json_str

            metrics = evaluate_table(pred_html, pred_json, true_html, true_json)

            predictions.append({
                "pred_html": pred_html,
                "pred_json": pred_json,
                "raw_output": result,
            })

            teds_scores.append(metrics.get("teds", 0.0))
            cell_accuracies.append(metrics.get("cell_accuracy", 0.0))
            structure_f1_scores.append(metrics.get("overall_structure_f1", 0.0))

    return {
        "predictions": predictions,
        "teds_list": teds_scores,
        "cell_accuracy_list": cell_accuracies,
        "structure_f1_list": structure_f1_scores,
        "metrics": {
            "avg_teds": float(np.mean(teds_scores)),
            "std_teds": float(np.std(teds_scores)),
            "avg_cell_accuracy": float(np.mean(cell_accuracies)),
            "std_cell_accuracy": float(np.std(cell_accuracies)),
            "avg_structure_f1": float(np.mean(structure_f1_scores)),
            "std_structure_f1": float(np.std(structure_f1_scores)),
        },
    }


def evaluate_document_format(
    model,
    dataset,
    batchsize: int,
    image_column: str,
    prompt: str,
) -> Dict[str, Any]:
    predictions = []
    layout_f1_scores = []
    reading_order_scores = []
    kv_f1_scores = []
    overall_f1_scores = []

    for i in tqdm(range(0, len(dataset), batchsize), desc="Processing Documents"):
        batch = dataset[i : i + batchsize]
        batch_images = batch[image_column]
        batch_prompts = [prompt] * len(batch_images)
        batch_result = model.run(prompts=batch_prompts, images=batch_images)

        for j, result in enumerate(batch_result):
            idx = i + j
            pred_json = parse_model_output_as_json(result) or {}
            pred_elements = pred_json.get("elements", [])

            true_gt_str = dataset[idx].get("ground_truth", "{}")
            true_gt = json.loads(true_gt_str) if isinstance(true_gt_str, str) else true_gt_str
            true_elements = true_gt.get("elements", [])

            metrics = evaluate_document(pred_elements, true_elements)

            predictions.append({
                "pred_elements": pred_elements,
                "raw_output": result,
            })

            layout_metrics = metrics.get("layout_detection", {})
            reading_metrics = metrics.get("reading_order", {})
            kv_metrics = metrics.get("key_value_extraction", {})

            layout_f1_scores.append(layout_metrics.get("overall_f1", 0.0))
            reading_order_scores.append(reading_metrics.get("order_accuracy", 0.0))
            kv_f1_scores.append(kv_metrics.get("f1", 0.0))
            overall_f1_scores.append(metrics.get("overall_f1", 0.0))

    return {
        "predictions": predictions,
        "layout_f1_list": layout_f1_scores,
        "reading_order_list": reading_order_scores,
        "kv_f1_list": kv_f1_scores,
        "overall_f1_list": overall_f1_scores,
        "metrics": {
            "avg_layout_f1": float(np.mean(layout_f1_scores)),
            "std_layout_f1": float(np.std(layout_f1_scores)),
            "avg_reading_order": float(np.mean(reading_order_scores)),
            "std_reading_order": float(np.std(reading_order_scores)),
            "avg_kv_f1": float(np.mean(kv_f1_scores)),
            "std_kv_f1": float(np.std(kv_f1_scores)),
            "avg_overall_f1": float(np.mean(overall_f1_scores)),
            "std_overall_f1": float(np.std(overall_f1_scores)),
        },
    }


def main(
    model_id: str,
    dataset_id: str,
    subset: str,
    split: str,
    batchsize: int,
    output_dataset_id: Optional[str],
    image_column: str = "image",
    prompt: Optional[str] = None,
    target_column: str = "response",
    format_type: str = "sentence",
):
    print(f"Load Models: {model_id}")
    model = MODELS[model_id]()

    print(f"Load Dataset: {dataset_id}, {subset}, {split}")
    dataset = load_dataset(dataset_id, split=split)
    print(dataset)

    if prompt is None:
        prompt = DEFAULT_PROMPTS.get(format_type, DEFAULT_PROMPTS["sentence"])

    print(f"Format: {format_type}")
    print(f"Prompt: {prompt}")

    if format_type == "sentence":
        result = evaluate_sentence(
            model, dataset, batchsize, image_column, target_column, prompt
        )
        print(f"\n{'='*60}")
        print("Sentence Evaluation Results:")
        print(f"  Average CER: {result['metrics']['avg_cer']:.4f}")
        print(f"  Std CER: {result['metrics']['std_cer']:.4f}")
        print(f"{'='*60}")

        if output_dataset_id:
            dataset = dataset.add_column("cer", result["cer_list"])
            dataset = dataset.add_column("ocr_result", result["predictions"])
            dataset.push_to_hub(output_dataset_id)

    elif format_type == "table":
        result = evaluate_table_format(model, dataset, batchsize, image_column, prompt)
        print(f"\n{'='*60}")
        print("Table Evaluation Results:")
        print(f"  Average TEDS: {result['metrics']['avg_teds']:.4f}")
        print(f"  Std TEDS: {result['metrics']['std_teds']:.4f}")
        print(f"  Average Cell Accuracy: {result['metrics']['avg_cell_accuracy']:.4f}")
        print(f"  Average Structure F1: {result['metrics']['avg_structure_f1']:.4f}")
        print(f"{'='*60}")

        if output_dataset_id:
            dataset = dataset.add_column("teds", result["teds_list"])
            dataset = dataset.add_column("cell_accuracy", result["cell_accuracy_list"])
            dataset = dataset.add_column("structure_f1", result["structure_f1_list"])
            raw_outputs = [p["raw_output"] for p in result["predictions"]]
            dataset = dataset.add_column("ocr_result", raw_outputs)
            dataset.push_to_hub(output_dataset_id)

    elif format_type == "document":
        result = evaluate_document_format(model, dataset, batchsize, image_column, prompt)
        print(f"\n{'='*60}")
        print("Document Evaluation Results:")
        print(f"  Average Layout F1: {result['metrics']['avg_layout_f1']:.4f}")
        print(f"  Average Reading Order Accuracy: {result['metrics']['avg_reading_order']:.4f}")
        print(f"  Average KV Extraction F1: {result['metrics']['avg_kv_f1']:.4f}")
        print(f"  Average Overall F1: {result['metrics']['avg_overall_f1']:.4f}")
        print(f"{'='*60}")

        if output_dataset_id:
            dataset = dataset.add_column("layout_f1", result["layout_f1_list"])
            dataset = dataset.add_column("reading_order", result["reading_order_list"])
            dataset = dataset.add_column("kv_f1", result["kv_f1_list"])
            dataset = dataset.add_column("overall_f1", result["overall_f1_list"])
            raw_outputs = [p["raw_output"] for p in result["predictions"]]
            dataset = dataset.add_column("ocr_result", raw_outputs)
            dataset.push_to_hub(output_dataset_id)

    else:
        raise ValueError(f"Unknown format type: {format_type}. Use 'sentence', 'table', or 'document'.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR models on different formats")
    parser.add_argument("model_id", type=str, default="rednote-hilab/dots.ocr")
    parser.add_argument("dataset_id", type=str, default="junyeong-nero/synthetic-ocr-images-korean")
    parser.add_argument("--subset", type=str, default="default")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--output-dataset-id", type=str, default=None)
    parser.add_argument("--image-column", type=str, default="image")
    parser.add_argument("--target-column", type=str, default="response")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--format",
        type=str,
        default="sentence",
        choices=["sentence", "table", "document"],
        help="Format of the dataset to evaluate (sentence, table, or document)",
    )

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["format_type"] = args_dict.pop("format")

    main(**args_dict)
