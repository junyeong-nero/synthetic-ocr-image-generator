"""vLLM-based VLM model."""

from typing import List

from PIL import Image

from evaluation.config import ModelConfig
from models.base import VLMModel


class VLLMModel(VLMModel):
    """
    vLLM-based VLM model for high-performance local inference.

    Supports tensor parallelism and optimized batching.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm package is required for VLLMModel. "
                "Install with: pip install vllm"
            )

        self.llm = LLM(
            model=config.model_id,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            trust_remote_code=True,
            max_model_len=config.max_model_len,
            limit_mm_per_prompt={"image": 1},
        )

        self.sampling_params = SamplingParams(
            temperature=config.temperature if config.temperature > 0 else 0.0,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        inputs = []
        for prompt, image in zip(prompts, images):
            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }
            )

        outputs = self.llm.generate(inputs, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        **kwargs,
    ) -> "VLLMModel":
        """
        Create a vLLM model from model ID.

        Args:
            model_id: HuggingFace model ID.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            dtype: Data type for model weights.
            **kwargs: Additional config options.

        Returns:
            VLLMModel instance.
        """
        from evaluation.config import InferenceBackend

        config = ModelConfig(
            model_id=model_id,
            backend=InferenceBackend.VLLM,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            **kwargs,
        )
        return cls(config)
