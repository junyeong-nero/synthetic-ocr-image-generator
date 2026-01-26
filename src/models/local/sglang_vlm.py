"""SGLang-based VLM model."""

from typing import List, Optional

from PIL import Image

from evaluation.config import ModelConfig
from models.base import VLMModel


class SGLangModel(VLMModel):
    """
    SGLang-based VLM model.

    Supports both local runtime and remote server modes.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._runtime = None

        try:
            import sglang as sgl
        except ImportError:
            raise ImportError(
                "sglang package is required for SGLangModel. "
                "Install with: pip install sglang[all]"
            )

        self._sgl = sgl

        if config.api_base:
            # Use remote server
            from sglang import RuntimeEndpoint

            sgl.set_default_backend(RuntimeEndpoint(config.api_base))
        else:
            # Use local runtime
            self._runtime = sgl.Runtime(
                model_path=config.model_id,
                tp_size=config.tensor_parallel_size,
            )
            sgl.set_default_backend(self._runtime)

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        sgl = self._sgl

        @sgl.function
        def vlm_inference(s, prompt: str, image: Image.Image):
            s += sgl.image(image)
            s += sgl.user(prompt)
            s += sgl.assistant(sgl.gen("response", max_tokens=self.config.max_tokens))

        results = []
        for prompt, image in zip(prompts, images):
            state = vlm_inference.run(prompt=prompt, image=image)
            results.append(state["response"])

        return results

    def run_batch(
        self,
        prompts: List[str],
        images: List[Image.Image],
        batch_size: int = 1,
    ) -> List[str]:
        """
        Run batched inference using SGLang's batch execution.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.
            batch_size: Batch size (SGLang handles batching internally).

        Returns:
            List of model responses.
        """
        sgl = self._sgl

        @sgl.function
        def vlm_inference(s, prompt: str, image: Image.Image):
            s += sgl.image(image)
            s += sgl.user(prompt)
            s += sgl.assistant(sgl.gen("response", max_tokens=self.config.max_tokens))

        # SGLang supports batch execution
        arguments = [
            {"prompt": prompt, "image": image}
            for prompt, image in zip(prompts, images)
        ]

        states = vlm_inference.run_batch(arguments)
        return [state["response"] for state in states]

    def __del__(self):
        """Cleanup runtime on deletion."""
        if self._runtime is not None:
            try:
                self._runtime.shutdown()
            except Exception:
                pass

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        api_base: Optional[str] = None,
        tensor_parallel_size: int = 1,
        **kwargs,
    ) -> "SGLangModel":
        """
        Create an SGLang model from model ID.

        Args:
            model_id: HuggingFace model ID.
            api_base: Optional server URL for remote mode.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            **kwargs: Additional config options.

        Returns:
            SGLangModel instance.
        """
        from evaluation.config import InferenceBackend

        config = ModelConfig(
            model_id=model_id,
            backend=InferenceBackend.SGLANG,
            api_base=api_base,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )
        return cls(config)
