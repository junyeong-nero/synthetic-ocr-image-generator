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
    "markdown": "Extract the markdown content from this image. Return the raw markdown text exactly as shown.",
}


def _load_builtin_model(model_id: str):
    import importlib.util
    import sys
    from pathlib import Path

    def _import_from_file(module_name: str, relative_path: str):
        """Import a module directly from file to avoid circular imports."""
        src_dir = Path(__file__).parent
        file_path = src_dir / relative_path
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    # Map model_id to (module_path, class_name)
    MODEL_REGISTRY = {
        "dummy": ("models/base.py", "Model"),
        "rednote-hilab/dots.ocr": ("models/vllm/dots_ocr.py", "DotsOCR"),
        "nanonets/Nanonets-OCR2-3B": ("models/vllm/nanonets_ocr.py", "NanonetsOCR"),
        "lightonai/LightOnOCR-1B-1025": ("models/vllm/light_on_ocr.py", "LightOnOCR"),
        "lightonai/LightOnOCR-2-1B": ("models/transformers/light_on_ocr2.py", "LightOnOCR2"),
        "allenai/olmOCR-2-7B-1025": ("models/vllm/olm_ocr.py", "OlmOCR"),
        "deepseek-ai/DeepSeek-OCR": ("models/transformers/deepseek_ocr.py", "DeepSeekOCR"),
        "deepseek-ai/DeepSeek-OCR-2": ("models/transformers/deepseek_ocr2.py", "DeepSeekOCR2"),
        "google/gemma-3-4b-it": ("models/transformers/gemma3_4b_it.py", "Gemma3_4B_IT"),
        "stepfun-ai/GOT-OCR-2.0-hf": ("models/transformers/got_ocr.py", "GotOCR"),
        "PaddlePaddle/PaddleOCR-VL": ("models/transformers/paddle_ocr.py", "PaddleOCR"),
        "Qwen/Qwen3-VL-2B-Instruct": ("models/transformers/qwen3_vl.py", "Qwen3VL"),
        "NCSOFT/VARCO-VISION-2.0-1.7B-OCR": ("models/transformers/varco_ocr.py", "VarcoOCR"),
    }

    if model_id not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_id: {model_id}. Available: {available}")

    module_path, class_name = MODEL_REGISTRY[model_id]
    module_name = f"_eval_{class_name}"
    module = _import_from_file(module_name, module_path)
    model_class = getattr(module, class_name)
    return model_class()


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


def get_dataset_subsets(dataset_id: str) -> List[str]:
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(dataset_id)
        return configs if configs else ["default"]
    except Exception:
        return ["default"]


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


def evaluate_markdown_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, Any]:
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")

    cer_list = [cer(gt, pred) for gt, pred in zip(ground_truths, predictions)]

    exact_match_list = [1.0 if pred.strip() == gt.strip() else 0.0 for pred, gt in zip(predictions, ground_truths)]

    def normalize_markdown(text: str) -> str:
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        return '\n'.join(lines)

    normalized_match_list = [
        1.0 if normalize_markdown(pred) == normalize_markdown(gt) else 0.0
        for pred, gt in zip(predictions, ground_truths)
    ]

    def count_markdown_elements(text: str) -> Dict[str, int]:
        counts = {
            "headers": 0,
            "code_blocks": 0,
            "lists": 0,
            "blockquotes": 0,
            "tables": 0,
            "links": 0,
        }
        lines = text.split('\n')
        in_code_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('```'):
                if not in_code_block:
                    counts["code_blocks"] += 1
                in_code_block = not in_code_block
            elif not in_code_block:
                if stripped.startswith('#'):
                    counts["headers"] += 1
                elif stripped.startswith('- ') or stripped.startswith('* ') or (len(stripped) > 2 and stripped[0].isdigit() and '. ' in stripped[:4]):
                    counts["lists"] += 1
                elif stripped.startswith('>'):
                    counts["blockquotes"] += 1
                elif stripped.startswith('|') and '|' in stripped[1:]:
                    counts["tables"] += 1
                if '[' in stripped and '](' in stripped:
                    counts["links"] += stripped.count('](')

        return counts

    element_f1_list = []
    for pred, gt in zip(predictions, ground_truths):
        pred_counts = count_markdown_elements(pred)
        gt_counts = count_markdown_elements(gt)

        total_pred = sum(pred_counts.values())
        total_gt = sum(gt_counts.values())

        if total_gt == 0 and total_pred == 0:
            element_f1_list.append(1.0)
        elif total_gt == 0 or total_pred == 0:
            element_f1_list.append(0.0)
        else:
            matched = sum(min(pred_counts[k], gt_counts[k]) for k in pred_counts)
            precision = matched / total_pred if total_pred > 0 else 0.0
            recall = matched / total_gt if total_gt > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            element_f1_list.append(f1)

    return {
        "predictions": predictions,
        "cer_list": cer_list,
        "exact_match_list": exact_match_list,
        "normalized_match_list": normalized_match_list,
        "element_f1_list": element_f1_list,
        "metrics": {
            "avg_cer": float(np.mean(cer_list)),
            "std_cer": float(np.std(cer_list)),
            "min_cer": float(np.min(cer_list)),
            "max_cer": float(np.max(cer_list)),
            "exact_match_rate": float(np.mean(exact_match_list)),
            "normalized_match_rate": float(np.mean(normalized_match_list)),
            "avg_element_f1": float(np.mean(element_f1_list)),
            "std_element_f1": float(np.std(element_f1_list)),
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
        elif format_type == "markdown":
            ground_truths = [
                {"markdown": dataset[i].get("markdown", "")}
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
    elif format_type == "markdown":
        return evaluate_markdown_metrics(predictions, [gt.get("markdown", "") for gt in ground_truths])
    else:
        raise ValueError(f"Unknown format_type: {format_type}")


def evaluate_subset(
    dataset_id: str,
    subset: str,
    split: str,
    format_type: str,
    model_id: Optional[str] = None,
    predictions: Optional[List[str]] = None,
    inference_fn: Optional[Callable[[List[Image.Image], List[str]], List[str]]] = None,
    image_column: str = "image",
    target_column: str = "typo_text",
    prompt: Optional[str] = None,
    batchsize: int = 1,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Evaluating subset: '{subset}' (format: {format_type})")
    print(f"{'='*60}")

    if subset == "default":
        dataset = load_dataset(dataset_id, split=split)
    else:
        dataset = load_dataset(dataset_id, name=subset, split=split)

    print(f"Loaded {len(dataset)} samples from '{subset}' subset")

    if prompt is None:
        prompt = DEFAULT_PROMPTS.get(format_type, DEFAULT_PROMPTS["sentence"])

    if predictions is not None:
        if len(predictions) != len(dataset):
            raise ValueError(f"Predictions count ({len(predictions)}) != dataset size ({len(dataset)})")

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
        elif format_type == "markdown":
            ground_truths = [
                {"markdown": dataset[i].get("markdown", "")}
                for i in range(len(dataset))
            ]

        if format_type == "sentence":
            result = evaluate_sentence_metrics(predictions, ground_truths)
        elif format_type == "table":
            result = evaluate_table_metrics(predictions, ground_truths)
        elif format_type == "document":
            result = evaluate_document_metrics(predictions, ground_truths)
        elif format_type == "markdown":
            result = evaluate_markdown_metrics(predictions, [gt.get("markdown", "") for gt in ground_truths])
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
    else:
        if model_id is None and inference_fn is None:
            raise ValueError("Either model_id or inference_fn must be provided")

        if inference_fn is None:
            model = _load_builtin_model(model_id)

            def inf_fn(imgs, prompts):
                return model.run(prompts=prompts, images=imgs)
            inference_fn = inf_fn

        result = evaluate(
            format_type=format_type,
            dataset=dataset,
            inference_fn=inference_fn,
            image_column=image_column,
            target_column=target_column,
            prompt=prompt,
            batchsize=batchsize,
        )

    print_results(result, format_type)
    return result


def evaluate_all_subsets(
    dataset_id: str,
    subsets: List[str],
    split: str = "train",
    model_id: Optional[str] = None,
    inference_fn: Optional[Callable[[List[Image.Image], List[str]], List[str]]] = None,
    predictions: Optional[Dict[str, List[str]]] = None,
    image_column: str = "image",
    target_column: str = "typo_text",
    batchsize: int = 1,
) -> Dict[str, Any]:
    all_results = {}

    for subset in subsets:
        lower = subset.lower()
        if "sentence" in lower:
            format_type = "sentence"
        elif "table" in lower:
            format_type = "table"
        elif "document" in lower or "doc" in lower:
            format_type = "document"
        elif "markdown" in lower or "md" in lower:
            format_type = "markdown"
        else:
            format_type = "sentence"

        subset_predictions = predictions.get(subset) if predictions else None

        try:
            result = evaluate_subset(
                dataset_id=dataset_id,
                subset=subset,
                split=split,
                format_type=format_type,
                model_id=model_id,
                predictions=subset_predictions,
                inference_fn=inference_fn,
                image_column=image_column,
                target_column=target_column,
                batchsize=batchsize,
            )
            all_results[subset] = result
        except Exception as e:
            print(f"Error evaluating subset '{subset}': {e}")
            all_results[subset] = {"error": str(e)}

    return all_results


def aggregate_all_results(results: Dict[str, Any]) -> Dict[str, Any]:
    aggregated = {
        "subsets": {},
        "overall_metrics": {},
        "summary": {},
    }

    valid_results = {k: v for k, v in results.items() if "error" not in v and "metrics" in v}

    if not valid_results:
        return {"error": "No valid results to aggregate"}

    all_metrics = {}
    for subset, result in valid_results.items():
        for metric_name, value in result["metrics"].items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)
        aggregated["subsets"][subset] = result["metrics"]

    for metric_name, values in all_metrics.items():
        aggregated["overall_metrics"][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "subsets_evaluated": len(values),
        }

    aggregated["summary"] = {
        "total_subsets": len(results),
        "successful_subsets": len(valid_results),
        "failed_subsets": [k for k, v in results.items() if "error" in v],
    }

    return aggregated


def print_aggregated_results(results: Dict[str, Any]):
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"\n{'='*60}")
    print(" AGGREGATED EVALUATION RESULTS ")
    print(f"{'='*60}")

    print(f"\nTotal subsets: {results['summary']['total_subsets']}")
    print(f"Successful: {results['summary']['successful_subsets']}")

    if results['summary']['failed_subsets']:
        print(f"Failed: {results['summary']['failed_subsets']}")

    print(f"\n--- Overall Metrics ---")
    for metric_name, stats in results['overall_metrics'].items():
        if isinstance(stats, dict) and "mean" in stats:
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print(f"\n--- Per-Subset Summary ---")
    for subset, metrics in results['subsets'].items():
        if "avg_cer" in metrics:
            print(f"  {subset}: CER = {metrics['avg_cer']:.4f}")
        elif "avg_teds" in metrics:
            print(f"  {subset}: TEDS = {metrics['avg_teds']:.4f}")
        elif "avg_overall_f1" in metrics:
            print(f"  {subset}: Overall F1 = {metrics['avg_overall_f1']:.4f}")

    print(f"{'='*60}")


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

    elif format_type == "markdown":
        print("Markdown Evaluation Results:")
        print(f"  Average CER: {result['metrics']['avg_cer']:.4f}")
        print(f"  Std CER: {result['metrics']['std_cer']:.4f}")
        print(f"  Exact Match Rate: {result['metrics']['exact_match_rate']:.4f}")
        print(f"  Normalized Match Rate: {result['metrics']['normalized_match_rate']:.4f}")
        print(f"  Average Element F1: {result['metrics']['avg_element_f1']:.4f}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR models on different formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single subset evaluation
  python evaluate.py --model-id "allenai/olmOCR-2-7B-1025" \\
      --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \\
      --format sentence

  # Evaluate specific subsets
  python evaluate.py --model-id "allenai/olmOCR-2-7B-1025" \\
      --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \\
      --subsets "sentence,table,document"

  # Evaluate all available subsets
  python evaluate.py --model-id "allenai/olmOCR-2-7B-1025" \\
      --dataset-id "junyeong-nero/synthetic-ocr-images-korean" \\
      --all-subsets
        """
    )

    parser.add_argument("--model-id", type=str, default=None,
                        help="Built-in model ID (e.g., allenai/olmOCR-2-7B-1025)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions file (.json, .jsonl, or .txt)")
    parser.add_argument("--dataset-id", type=str, required=True,
                        help="HuggingFace dataset ID")
    parser.add_argument("--subset", type=str, default="default",
                        help="Single subset to evaluate")
    parser.add_argument("--subsets", type=str, default=None,
                        help="Comma-separated list of subsets to evaluate (e.g., 'sentence,table')")
    parser.add_argument("--all-subsets", action="store_true",
                        help="Evaluate all available subsets in the dataset")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--batchsize", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--output-dataset-id", type=str, default=None,
                        help="Push results to this HuggingFace dataset")
    parser.add_argument("--image-column", type=str, default="image",
                        help="Image column name")
    parser.add_argument("--target-column", type=str, default="typo_text",
                        help="Ground truth column name for sentence format")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt")
    parser.add_argument("--format", type=str, default="sentence",
                        choices=["sentence", "table", "document", "markdown"],
                        help="Evaluation format")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    if args.model_id is None and args.predictions is None:
        parser.error("Either --model-id or --predictions must be provided")

    if args.all_subsets:
        print(f"Fetching available subsets for: {args.dataset_id}")
        subsets = get_dataset_subsets(args.dataset_id)
        print(f"Found subsets: {subsets}")
    elif args.subsets:
        subsets = [s.strip() for s in args.subsets.split(",")]
    elif args.subset != "default":
        subsets = [args.subset]
    else:
        subsets = ["default"]

    if len(subsets) > 1 or args.all_subsets:
        results = evaluate_all_subsets(
            dataset_id=args.dataset_id,
            subsets=subsets,
            split=args.split,
            model_id=args.model_id,
            predictions=None,
            inference_fn=None,
            image_column=args.image_column,
            target_column=args.target_column,
            batchsize=args.batchsize,
        )
        aggregated = aggregate_all_results(results)
        print_aggregated_results(aggregated)

        if args.output_file:
            output_path = Path(args.output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(aggregated, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.output_file}")
    else:
        print(f"Loading dataset: {args.dataset_id}")
        if args.subset == "default":
            dataset = load_dataset(args.dataset_id, split=args.split)
        else:
            dataset = load_dataset(args.dataset_id, name=args.subset, split=args.split)
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
            elif args.format == "markdown":
                ground_truths = [
                    {"markdown": dataset[i].get("markdown", "")}
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


if __name__ == "__main__":
    main()
