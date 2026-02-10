"""DeepSeek-OCR model wrapper."""

import contextlib
import io
import os
import re
import tempfile
from typing import List

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from models.transformers.base import BaseTransformersOCR


class DeepSeekOCR(BaseTransformersOCR):
    """Wrapper for the DeepSeek-OCR model."""

    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"

    _LOG_PREFIXES = (
        "A new version of the following files was downloaded",
        ". Make sure to double-check",
        "Configuration:",
        "Model:",
        "Backend:",
        "Dataset:",
        "Format:",
        "Batch Size:",
        "Temperature:",
        "Max Tokens:",
        "Config File:",
        "Downloading shards:",
        "Some weights of",
        "You should probably TRAIN",
        "The attention",
        "Setting `pad_token_id`",
        "The `seen_tokens`",
        "`get_max_cache()`",
        "The attention layers",
        "image size:",
        "valid image tokens:",
        "output texts tokens",
        "compression ratio:",
        "BASE:",
        "PATCHES:",
        "NO PATCHES",
    )

    @classmethod
    def _extract_text_from_infer_stdout(cls, stdout_text: str) -> str:
        lines = [line.strip() for line in stdout_text.replace("\r\n", "\n").split("\n")]
        if not lines:
            return ""

        delimiter_indices = [i for i, line in enumerate(lines) if line == "====================="]
        candidate_scope = lines[delimiter_indices[-1] + 1 :] if delimiter_indices else lines

        def _is_candidate(line: str) -> bool:
            if not line:
                return False
            if line.startswith("Evaluating:"):
                return False
            if line.startswith("=================================================="):
                return False
            if line in {"=====================", "NO PATCHES"}:
                return False
            if line.startswith(cls._LOG_PREFIXES):
                return False
            if re.fullmatch(r"[=\-\s]+", line):
                return False
            return bool(re.search(r"[A-Za-z0-9가-힣]", line))

        candidates = [line for line in candidate_scope if _is_candidate(line)]
        if candidates:
            return "\n".join(candidates).strip()

        fallback = [line for line in lines if _is_candidate(line)]
        return "\n".join(fallback).strip()

    @classmethod
    def _normalize_infer_result(cls, result: object, stdout_text: str) -> str:
        if isinstance(result, str):
            normalized = result.strip()
            if normalized:
                return normalized
        elif isinstance(result, (list, tuple)):
            parts = [str(item).strip() for item in result if str(item).strip()]
            if parts:
                return "\n".join(parts)
        elif isinstance(result, dict):
            for key in ("text", "output", "response", "prediction", "result"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return cls._extract_text_from_infer_stdout(stdout_text)

    def _load_model(self, model_id: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
        )
        if self.device == "cuda":
            self.model = self.model.eval().cuda().to(torch.bfloat16)
        else:
            self.model = self.model.eval()

    def run(self, prompts: List[str], images: List[Image.Image]) -> List[str]:
        """
        Run inference on a batch of images.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            List of model responses.
        """
        results = []
        for prompt, image in zip(prompts, images):
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as temp_image_file:
                image.save(temp_image_file, format="PNG")
                temp_image_path = temp_image_file.name

            full_prompt = f"<image>\n{prompt}"

            try:
                with io.StringIO() as infer_stdout_buffer, contextlib.redirect_stdout(
                    infer_stdout_buffer
                ):
                    res = self.model.infer(
                        self.tokenizer,
                        prompt=full_prompt,
                        image_file=temp_image_path,
                        output_path="output/",
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        test_compress=True,
                    )
                    captured_stdout = infer_stdout_buffer.getvalue()
                results.append(self._normalize_infer_result(res, captured_stdout))
            finally:
                os.remove(temp_image_path)
        return results
