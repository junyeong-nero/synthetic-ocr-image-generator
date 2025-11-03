"""
Convert document images to markdown using DeepSeek-OCR with vLLM.

This script processes images through the DeepSeek-OCR model to extract
text and structure as markdown, using vLLM for efficient batch processing.

NOTE: Uses vLLM nightly wheels from main (PR #27247 now merged). First run
may take a few minutes to download and install dependencies.

Features:
- Multiple resolution modes (Tiny/Small/Base/Large/Gundam)
- LaTeX equation recognition
- Table extraction and formatting
- Document structure preservation
- Image grounding and descriptions
- Multilingual support
- Batch processing with vLLM for better performance
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Union
from datetime import datetime

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolution mode presets
RESOLUTION_MODES = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
    },  # Dynamic resolution
}

# Prompt mode presets (from DeepSeek-OCR GitHub)
PROMPT_MODES = {
    "document": "<image>\n<|grounding|>Convert the document to markdown.",
    "image": "<image>\n<|grounding|>OCR this image.",
    "free": "<image>\nFree OCR.",
    "figure": "<image>\nParse the figure.",
    "describe": "<image>\nDescribe this image in detail.",
}


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown. ",
) -> List[Dict]:
    """Create chat message for OCR processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to RGB
    pil_img = pil_img.convert("RGB")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Return message in vLLM format
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    resolution_mode: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- deepseek
- deepseek-ocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains markdown-formatted OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using DeepSeek-OCR.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `markdown`
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Resolution Mode**: {resolution_mode}
- **Base Size**: {base_size}
- **Image Size**: {image_size}
- **Crop Mode**: {crop_mode}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

DeepSeek-OCR is a state-of-the-art document OCR model that excels at:
- 📐 **LaTeX equations** - Mathematical formulas preserved in LaTeX format
- 📊 **Tables** - Extracted and formatted as HTML/markdown
- 📝 **Document structure** - Headers, lists, and formatting maintained
- 🖼️ **Image grounding** - Spatial layout and bounding box information
- 🔍 **Complex layouts** - Multi-column and hierarchical structures
- 🌍 **Multilingual** - Supports multiple languages

### Resolution Modes

- **Tiny** (512×512): Fast processing, 64 vision tokens
- **Small** (640×640): Balanced speed/quality, 100 vision tokens
- **Base** (1024×1024): High quality, 256 vision tokens
- **Large** (1280×1280): Maximum quality, 400 vision tokens
- **Gundam** (dynamic): Adaptive multi-tile processing for large documents

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format with preserved structure
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{{{{output_dataset_id}}}}", split="{split}")

# Access the markdown text
for example in dataset:
    print(example["markdown"])
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Column: {{{{info['column_name']}}}} - Model: {{{{info['model_id']}}}}")
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) DeepSeek OCR vLLM script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \\\\
    {source_dataset} \\\\
    <output-dataset> \\\\
    --resolution-mode {resolution_mode} \\\\
    --image-column {image_column}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.1f} images/second
- **Processing Method**: Batch processing with vLLM (2-3x speedup over sequential)

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    prompt_column: str = "prompt",
    batch_size: int = 8,  # Smaller batch size to avoid potential memory issues with DeepSeek-OCR
    model: str = "deepseek-ai/DeepSeek-OCR",
    resolution_mode: str = "gundam",
    base_size: int = None,
    image_size: int = None,
    crop_mode: bool = None,
    max_model_len: int = 8192,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    prompt_mode: str = "document",
    prompt: str = None,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
):
    """Process images from HF dataset through DeepSeek-OCR model with vLLM."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Enable HF_TRANSFER for faster downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Determine resolution settings
    if resolution_mode in RESOLUTION_MODES:
        mode_config = RESOLUTION_MODES[resolution_mode]
        final_base_size = (
            base_size if base_size is not None else mode_config["base_size"]
        )
        final_image_size = (
            image_size if image_size is not None else mode_config["image_size"]
        )
        final_crop_mode = (
            crop_mode if crop_mode is not None else mode_config["crop_mode"]
        )
        logger.info(f"Using resolution mode: {resolution_mode}")
    else:
        # Custom mode - require all parameters
        if base_size is None or image_size is None or crop_mode is None:
            raise ValueError(
                f"Invalid resolution mode '{resolution_mode}'. "
                f"Use one of {list(RESOLUTION_MODES.keys())} or specify "
                f"--base-size, --image-size, and --crop-mode manually."
            )
        final_base_size = base_size
        final_image_size = image_size
        final_crop_mode = crop_mode
        resolution_mode = "custom"

    logger.info(
        f"Resolution: base_size={final_base_size}, "
        f"image_size={final_image_size}, crop_mode={final_crop_mode}"
    )

    # Determine prompt
    if prompt is not None:
        final_prompt = prompt
        logger.info(f"Using custom prompt")
    elif prompt_mode in PROMPT_MODES:
        final_prompt = PROMPT_MODES[prompt_mode]
        logger.info(f"Using prompt mode: {prompt_mode}")
    else:
        raise ValueError(
            f"Invalid prompt mode '{prompt_mode}'. "
            f"Use one of {list(PROMPT_MODES.keys())} or specify --prompt"
        )

    logger.info(f"Prompt: {final_prompt}")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, "sentence_typos", split=split)

    # Validate image column
    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {model}")
    logger.info("This may take a few minutes on first run...")

    # Add specific parameters for DeepSeek-OCR compatibility
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=False,  # Use torch.compile instead of eager execution
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for OCR
        max_tokens=max_tokens,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(
        "Using vLLM for batch processing - should be faster than sequential processing"
    )

    # Process images in batches
    all_markdown = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="DeepSeek-OCR vLLM processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]
        batch_prompts = [dataset[i][prompt_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [
                make_ocr_message(img, prompt)
                for img, prompt in zip(batch_images, batch_prompts)
            ]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_markdown.append(text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_markdown.extend(["[OCR FAILED]"] * len(batch_images))

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add markdown column to dataset
    logger.info("Adding markdown column to dataset")
    dataset = dataset.add_column("markdown", all_markdown)

    # Handle inference_info tracking
    logger.info("Updating inference_info...")

    # Check for existing inference_info
    if "inference_info" in dataset.column_names:
        # Parse existing info from first row (all rows have same info)
        try:
            existing_info = json.loads(dataset[0]["inference_info"])
            if not isinstance(existing_info, list):
                existing_info = [existing_info]  # Convert old format to list
        except (json.JSONDecodeError, TypeError):
            existing_info = []
        # Remove old column to update it
        dataset = dataset.remove_columns(["inference_info"])
    else:
        existing_info = []

    # Add new inference info
    new_info = {
        "column_name": "markdown",
        "model_id": model,
        "processing_date": datetime.now().isoformat(),
        "resolution_mode": resolution_mode,
        "base_size": final_base_size,
        "image_size": final_image_size,
        "crop_mode": final_crop_mode,
        "prompt": final_prompt,
        "prompt_mode": prompt_mode if prompt is None else "custom",
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "script": "deepseek-ocr-vllm.py",
        "script_version": "1.0.0",
        "script_url": "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py",
        "implementation": "vllm (batch processing)",
    }
    existing_info.append(new_info)

    # Add updated inference_info column
    info_json = json.dumps(existing_info, ensure_ascii=False)
    dataset = dataset.add_column("inference_info", [info_json] * len(dataset))

    # Push to hub
    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(output_dataset, private=private, token=HF_TOKEN)

    # Create and push dataset card
    logger.info("Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=model,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        resolution_mode=resolution_mode,
        base_size=final_base_size,
        image_size=final_image_size,
        crop_mode=final_crop_mode,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)
    logger.info("✅ Dataset card created and pushed!")

    logger.info("✅ OCR conversion complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("DeepSeek-OCR to Markdown Converter (vLLM)")
        print("=" * 80)
        print("\nThis script converts document images to markdown using")
        print("DeepSeek-OCR with vLLM for efficient batch processing.")
        print("\nFeatures:")
        print("- Multiple resolution modes (Tiny/Small/Base/Large/Gundam)")
        print("- LaTeX equation recognition")
        print("- Table extraction and formatting")
        print("- Document structure preservation")
        print("- Image grounding and spatial layout")
        print("- Multilingual support")
        print("- ⚡ Fast batch processing with vLLM (2-3x speedup)")
        print("\nExample usage:")
        print("\n1. Basic OCR conversion (Gundam mode - dynamic resolution):")
        print("   uv run deepseek-ocr-vllm.py document-images markdown-docs")
        print("\n2. High quality mode (Large - 1280×1280):")
        print(
            "   uv run deepseek-ocr-vllm.py scanned-pdfs extracted-text --resolution-mode large"
        )
        print("\n3. Fast processing (Tiny - 512×512):")
        print("   uv run deepseek-ocr-vllm.py quick-test output --resolution-mode tiny")
        print("\n4. Parse figures from documents:")
        print(
            "   uv run deepseek-ocr-vllm.py scientific-papers figures --prompt-mode figure"
        )
        print("\n5. Free OCR without layout:")
        print("   uv run deepseek-ocr-vllm.py images text --prompt-mode free")
        print("\n6. Process a subset for testing:")
        print(
            "   uv run deepseek-ocr-vllm.py large-dataset test-output --max-samples 10"
        )
        print("\n7. Custom resolution:")
        print("   uv run deepseek-ocr-vllm.py dataset output \\")
        print("       --base-size 1024 --image-size 640 --crop-mode")
        print("\n8. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print("     -s HF_TOKEN \\")
        print("     -e UV_TORCH_BACKEND=auto \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \\"
        )
        print("       your-document-dataset \\")
        print("       your-markdown-output")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run deepseek-ocr-vllm.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="OCR images to markdown using DeepSeek-OCR (vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Resolution Modes:
  tiny      512×512 pixels, fast processing (64 vision tokens)
  small     640×640 pixels, balanced (100 vision tokens)
  base      1024×1024 pixels, high quality (256 vision tokens)
  large     1280×1280 pixels, maximum quality (400 vision tokens)
  gundam    Dynamic multi-tile processing (adaptive)

Prompt Modes:
  document  Convert document to markdown with grounding (default)
  image     OCR any image with grounding
  free      Free OCR without layout preservation
  figure    Parse figures from documents
  describe  Generate detailed image descriptions

Examples:
  # Basic usage with default Gundam mode
  uv run deepseek-ocr-vllm.py my-images-dataset ocr-results

  # High quality processing
  uv run deepseek-ocr-vllm.py documents extracted-text --resolution-mode large

  # Fast processing for testing
  uv run deepseek-ocr-vllm.py dataset output --resolution-mode tiny --max-samples 100

  # Parse figures from a document dataset
  uv run deepseek-ocr-vllm.py scientific-papers figures --prompt-mode figure

  # Free OCR without layout (fastest)
  uv run deepseek-ocr-vllm.py images text --prompt-mode free

  # Custom prompt for specific task
  uv run deepseek-ocr-vllm.py dataset output --prompt "<image>\nExtract all table data."

  # Custom resolution settings
  uv run deepseek-ocr-vllm.py dataset output --base-size 1024 --image-size 640 --crop-mode

  # With custom batch size for performance tuning
  uv run deepseek-ocr-vllm.py dataset output --batch-size 16 --max-model-len 16384
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID from Hugging Face Hub")
    parser.add_argument("output_dataset", help="Output dataset ID for Hugging Face Hub")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (default: image)",
    )
    parser.add_argument(
        "--prompt-column",
        default="prompt",
        help="Column containing images (default: prompt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8, adjust based on GPU memory)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-OCR",
        help="Model to use (default: deepseek-ai/DeepSeek-OCR)",
    )
    parser.add_argument(
        "--resolution-mode",
        default="gundam",
        choices=list(RESOLUTION_MODES.keys()) + ["custom"],
        help="Resolution mode preset (default: gundam)",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        help="Base resolution size (overrides resolution-mode)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Image tile size (overrides resolution-mode)",
    )
    parser.add_argument(
        "--crop-mode",
        action="store_true",
        help="Enable dynamic multi-tile cropping (overrides resolution-mode)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--prompt-mode",
        default="document",
        choices=list(PROMPT_MODES.keys()),
        help="Prompt mode preset (default: document). Use --prompt for custom prompts.",
    )
    parser.add_argument(
        "--prompt",
        help="Custom OCR prompt (overrides --prompt-mode)",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token")
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before processing (useful for random sampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        prompt_column=args.prompt_column,
        batch_size=args.batch_size,
        model=args.model,
        resolution_mode=args.resolution_mode,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode if args.crop_mode else None,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prompt_mode=args.prompt_mode,
        prompt=args.prompt,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
    )
