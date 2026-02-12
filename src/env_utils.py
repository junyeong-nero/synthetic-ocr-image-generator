import os
import platform
import random
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Optional


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_env_file(env_file: str = ".env", override: bool = False) -> None:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = Path(__file__).resolve().parent.parent / env_file

    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue
        if key in os.environ and not override:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ[key] = value

def _get_package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def get_environment_metadata() -> Dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    env["packages"] = {
        "torch": _get_package_version("torch"),
        "transformers": _get_package_version("transformers"),
        "datasets": _get_package_version("datasets"),
        "numpy": _get_package_version("numpy"),
        "pandas": _get_package_version("pandas"),
        "uv": _get_package_version("uv"),
    }

    try:
        import torch

        env["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None))
            and torch.backends.mps.is_available(),
        }
    except Exception:
        env["torch"] = {
            "cuda_available": False,
            "mps_available": False,
        }

    return env
