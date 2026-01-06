import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union

import numpy as np
from datasets import load_dataset, Dataset
from PIL import Image
from tqdm import tqdm

from metrics.edit_distance import cer
from metrics.table_document_metrics import evaluate_table, evaluate_document


DEFAULT_PROMPTS = {
    "sentence": "Extract all text from the image verbatim, including typos, without translation or character modification.",
    "table": "Extract the table from this image. Return the result as HTML table format.",
    "document": "Extract all text elements from this document image. Return as JSON with 'elements' array containing objects with 'type', 'text', 'bounding_box', and 'reading_order' fields.",
}


def _load_builtin_model(model_id: str):
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

    if model_id not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model_id: {model_id}. Available: {available}")

    return MODELS[model_id]()


def parse_model_output_as_json(output: str) -> Optional[Dict]:
    if not isinstance(output, str):
        return output if isinstance(output, dict) else None

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
    if not isinstance(output, str):
        return str(output)

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


def load_predictions_file(filepath: str) -> List[str]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {filepath}")

    predictions = []

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                predictions = [str(item) if not isinstance(item, str) else item for item in data]
            else:
                raise ValueError("JSON file must contain a list of predictions")

    elif path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict) and "prediction" in item:
                        predictions.append(item["prediction"])
                    elif isinstance(item, str):
                        predictions.append(item)
                    else:
                        predictions.append(json.dumps(item, ensure_ascii=False))

    elif path.suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            predictions = [line.strip() for line in f if line.strip()]

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json, .jsonl, or .txt")

    return predictions


def evaluate_sentence_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, Any]:
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")

    cer_list = [cer(gt, pred) for gt, pred in zip(ground_truths, predictions)]

    return {
        "predictions": predictions,
        "cer_list": cer_list,
        "metrics": {
            "avg_cer": float(np.mean(cer_list)),
            "std_cer": float(np.std(cer_list)),
            "min_cer": float(np.min(cer_list)),
            "max_cer": float(np.max(cer_list)),
        },
    }


def evaluate_table_metrics(
    predictions: List[str],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")

    results = []
    teds_scores = []
    cell_accuracies = []
    structure_f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        pred_html = extract_html_table(pred)
        pred_json = parse_model_output_as_json(pred) or {}

        true_html = gt.get("html", "")
        true_json = gt.get("json", {})
        if isinstance(true_json, str):
            true_json = json.loads(true_json) if true_json else {}

        metrics = evaluate_table(pred_html, pred_json, true_html, true_json)

        results.append({
            "pred_html": pred_html,
            "pred_json": pred_json,
            "raw_output": pred,
        })

        teds_scores.append(metrics.get("teds", 0.0))
        cell_accuracies.append(metrics.get("cell_accuracy", 0.0))
        structure_f1_scores.append(metrics.get("overall_structure_f1", 0.0))

    return {
        "predictions": results,
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


def evaluate_document_metrics(
    predictions: List[str],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")

    results = []
    layout_f1_scores = []
    reading_order_scores = []
    kv_f1_scores = []
    overall_f1_scores = []

    for pred, gt in zip(predictions, ground_truths):
        pred_json = parse_model_output_as_json(pred) or {}
        pred_elements = pred_json.get("elements", [])

        true_gt = gt.get("ground_truth", gt)
        if isinstance(true_gt, str):
            true_gt = json.loads(true_gt) if true_gt else {}
        true_elements = true_gt.get("elements", [])

        metrics = evaluate_document(pred_elements, true_elements)

        results.append({
            "pred_elements": pred_elements,
            "raw_output": pred,
        })

        layout_metrics = metrics.get("layout_detection", {})
        reading_metrics = metrics.get("reading_order", {})
        kv_metrics = metrics.get("key_value_extraction", {})

        layout_f1_scores.append(layout_metrics.get("overall_f1", 0.0))
        reading_order_scores.append(reading_metrics.get("order_accuracy", 0.0))
        kv_f1_scores.append(kv_metrics.get("f1", 0.0))
        overall_f1_scores.append(metrics.get("overall_f1", 0.0))

    return {
        "predictions": results,
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


def evaluate(
    format_type: str,
    dataset: Optional[Union[Dataset, str]] = None,
    predictions: Optional[List[str]] = None,
    ground_truths: Optional[List[Any]] = None,
    inference_fn: Optional[Callable[[List[Image.Image], List[str]], List[str]]] = None,
    model_id: Optional[str] = None,
    image_column: str = "image",
    target_column: str = "typo_text",
    prompt: Optional[str] = None,
    batchsize: int = 1,
    split: str = "train",
) -> Dict[str, Any]:
    """
    Evaluate OCR predictions.

    Three usage modes:
    1. With predictions + ground_truths: Evaluate pre-computed predictions
    2. With inference_fn + dataset: Run inference using custom function
    3. With model_id + dataset: Run inference using built-in model

    Args:
        format_type: "sentence", "table", or "document"
        dataset: HuggingFace Dataset or dataset_id string
        predictions: List of model predictions (strings)
        ground_truths: List of ground truth values
        inference_fn: Custom inference function (images, prompts) -> predictions
        model_id: Built-in model ID (e.g., "allenai/olmOCR-2-7B-1025")
        image_column: Column name for images in dataset
        target_column: Column name for ground truth in dataset (sentence format)
        prompt: Custom prompt (uses format-specific default if None)
        batchsize: Batch size for inference
        split: Dataset split to use

    Returns:
        Evaluation results with metrics
    """
    if prompt is None:
        prompt = DEFAULT_PROMPTS.get(format_type, DEFAULT_PROMPTS["sentence"])

    if predictions is not None and ground_truths is not None:
        pass
    elif dataset is not None:
        if isinstance(dataset, str):
            dataset = load_dataset(dataset, split=split)

        if inference_fn is None and model_id is None:
            raise ValueError("Either inference_fn or model_id must be provided when using dataset")

        if inference_fn is None:
            model = _load_builtin_model(model_id)

            def inference_fn(imgs, prompts):
                return model.run(prompts=prompts, images=imgs)

        predictions = []
        for i in tqdm(range(0, len(dataset), batchsize), desc="Running inference"):
            batch = dataset[i : i + batchsize]
            batch_images = batch[image_column]
            batch_prompts = [prompt] * len(batch_images)
            batch_result = inference_fn(batch_images, batch_prompts)
            predictions.extend(batch_result)

        if format_type == "sentence":
            ground_truths = dataset[target_column]
        elif format_type == "table":
            ground_truths = [
                {"html": dataset[i].get("html", ""), "json": dataset[i].get("json", "{}")}
                for i in range(len(dataset))
            ]
        elif format_type == "document":
            ground_truths = [
                {"ground_truth": dataset[i].get("ground_truth", "{}")}
                for i in range(len(dataset))
            ]
    else:
        raise ValueError("Either (predictions + ground_truths) or dataset must be provided")

    if format_type == "sentence":
        return evaluate_sentence_metrics(predictions, ground_truths)
    elif format_type == "table":
        return evaluate_table_metrics(predictions, ground_truths)
    elif format_type == "document":
        return evaluate_document_metrics(predictions, ground_truths)
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


def print_results(result: Dict[str, Any], format_type: str):
    print(f"\n{'='*60}")

    if format_type == "sentence":
        print("Sentence Evaluation Results:")
        print(f"  Average CER: {result['metrics']['avg_cer']:.4f}")
        print(f"  Std CER: {result['metrics']['std_cer']:.4f}")
        print(f"  Min CER: {result['metrics']['min_cer']:.4f}")
        print(f"  Max CER: {result['metrics']['max_cer']:.4f}")

    elif format_type == "table":
        print("Table Evaluation Results:")
        print(f"  Average TEDS: {result['metrics']['avg_teds']:.4f}")
        print(f"  Std TEDS: {result['metrics']['std_teds']:.4f}")
        print(f"  Average Cell Accuracy: {result['metrics']['avg_cell_accuracy']:.4f}")
        print(f"  Average Structure F1: {result['metrics']['avg_structure_f1']:.4f}")

    elif format_type == "document":
        print("Document Evaluation Results:")
        print(f"  Average Layout F1: {result['metrics']['avg_layout_f1']:.4f}")
        print(f"  Average Reading Order: {result['metrics']['avg_reading_order']:.4f}")
        print(f"  Average KV F1: {result['metrics']['avg_kv_f1']:.4f}")
        print(f"  Average Overall F1: {result['metrics']['avg_overall_f1']:.4f}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR models on different formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using built-in model
  python evaluate.py --model-id "allenai/olmOCR-2-7B-1025" \\
      --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \\
      --format sentence

  # Using pre-computed predictions file
  python evaluate.py --predictions results.jsonl \\
      --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \\
      --format table

  # Predictions file formats:
  - .json: ["pred1", "pred2", ...]
  - .jsonl: {"prediction": "pred1"}\\n{"prediction": "pred2"}\\n...
  - .txt: one prediction per line
        """
    )

    parser.add_argument("--model-id", type=str, default=None,
                        help="Built-in model ID (e.g., allenai/olmOCR-2-7B-1025)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions file (.json, .jsonl, or .txt)")
    parser.add_argument("--dataset-id", type=str, required=True,
                        help="HuggingFace dataset ID")
    parser.add_argument("--subset", type=str, default="default")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--output-dataset-id", type=str, default=None,
                        help="Push results to this HuggingFace dataset")
    parser.add_argument("--image-column", type=str, default="image")
    parser.add_argument("--target-column", type=str, default="typo_text")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--format", type=str, default="sentence",
                        choices=["sentence", "table", "document"],
                        help="Evaluation format")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    if args.model_id is None and args.predictions is None:
        parser.error("Either --model-id or --predictions must be provided")

    print(f"Loading dataset: {args.dataset_id}")
    dataset = load_dataset(args.dataset_id, split=args.split)
    print(f"Dataset loaded: {len(dataset)} samples")

    if args.predictions:
        print(f"Loading predictions from: {args.predictions}")
        predictions = load_predictions_file(args.predictions)
        print(f"Loaded {len(predictions)} predictions")

        if len(predictions) != len(dataset):
            raise ValueError(f"Predictions count ({len(predictions)}) != dataset size ({len(dataset)})")

        if args.format == "sentence":
            ground_truths = dataset[args.target_column]
        elif args.format == "table":
            ground_truths = [
                {"html": dataset[i].get("html", ""), "json": dataset[i].get("json", "{}")}
                for i in range(len(dataset))
            ]
        elif args.format == "document":
            ground_truths = [
                {"ground_truth": dataset[i].get("ground_truth", "{}")}
                for i in range(len(dataset))
            ]

        result = evaluate(
            format_type=args.format,
            predictions=predictions,
            ground_truths=ground_truths,
        )
    else:
        print(f"Using model: {args.model_id}")
        result = evaluate(
            format_type=args.format,
            dataset=dataset,
            model_id=args.model_id,
            image_column=args.image_column,
            target_column=args.target_column,
            prompt=args.prompt,
            batchsize=args.batchsize,
        )

    print_results(result, args.format)

    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result["metrics"], f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output_file}")

    if args.output_dataset_id:
        if args.format == "sentence":
            dataset = dataset.add_column("cer", result["cer_list"])
            dataset = dataset.add_column("ocr_result", result["predictions"])
        elif args.format == "table":
            dataset = dataset.add_column("teds", result["teds_list"])
            dataset = dataset.add_column("cell_accuracy", result["cell_accuracy_list"])
            dataset = dataset.add_column("structure_f1", result["structure_f1_list"])
            raw_outputs = [p["raw_output"] for p in result["predictions"]]
            dataset = dataset.add_column("ocr_result", raw_outputs)
        elif args.format == "document":
            dataset = dataset.add_column("layout_f1", result["layout_f1_list"])
            dataset = dataset.add_column("reading_order", result["reading_order_list"])
            dataset = dataset.add_column("kv_f1", result["kv_f1_list"])
            dataset = dataset.add_column("overall_f1", result["overall_f1_list"])
            raw_outputs = [p["raw_output"] for p in result["predictions"]]
            dataset = dataset.add_column("ocr_result", raw_outputs)

        dataset.push_to_hub(args.output_dataset_id)
        print(f"Results pushed to: {args.output_dataset_id}")

    return result


if __name__ == "__main__":
    main()
