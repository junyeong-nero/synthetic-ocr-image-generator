"""HuggingFace Transformers-based VLM model."""

from typing import List

from PIL import Image

from evaluation.config import ModelConfig
from models.base import VLMModel


class TransformersVLM(VLMModel):
    """
    HuggingFace Transformers-based VLM model.

    Supports various vision-language models available on HuggingFace.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        try:
            import torch
            from transformers import AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch packages are required for TransformersVLM. "
                "Install with: pip install transformers torch"
            )

        # Import the appropriate auto model class (name changed in transformers versions)
        try:
            from transformers import AutoModelForImageTextToText as AutoVisionModel
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as AutoVisionModel
            except ImportError:
                from transformers import AutoModelForCausalLM as AutoVisionModel

        self._torch = torch

        # Determine device
        if config.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif config.device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Determine dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            trust_remote_code=True,
        )

        try:
            self.model = AutoVisionModel.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
            )
        except Exception:
            # Fallback to AutoModelForCausalLM for some models
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
            )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        torch = self._torch
        results = []

        for prompt, image in zip(prompts, images):
            # Format message for chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                # Fallback for models without chat template
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                )

            # Move inputs to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=self.config.temperature > 0,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    top_p=self.config.top_p if self.config.temperature > 0 else None,
                )

            # Decode, trimming input tokens
            input_len = inputs.get("input_ids", inputs.get("pixel_values")).shape[-1]
            if hasattr(inputs, "input_ids"):
                input_len = inputs["input_ids"].shape[-1]
                generated_ids_trimmed = outputs[0][input_len:]
            else:
                generated_ids_trimmed = outputs[0]

            decoded = self.processor.decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Clean up common patterns
            if "Assistant:" in decoded:
                decoded = decoded.split("Assistant:")[-1].strip()

            results.append(decoded)

        return results

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kwargs,
    ) -> "TransformersVLM":
        """
        Create a Transformers VLM from model ID.

        Args:
            model_id: HuggingFace model ID.
            device: Device for inference.
            dtype: Data type for model weights.
            **kwargs: Additional config options.

        Returns:
            TransformersVLM instance.
        """
        from evaluation.config import InferenceBackend

        config = ModelConfig(
            model_id=model_id,
            backend=InferenceBackend.TRANSFORMERS,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        return cls(config)
