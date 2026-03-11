from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def test_model_runtime_deps_are_not_in_base_project_dependencies() -> None:
    pyproject = _load_pyproject()
    base_dependencies = pyproject["project"]["dependencies"]

    assert all(not dep.startswith("datasets") for dep in base_dependencies)
    assert all(not dep.startswith("torch") for dep in base_dependencies)
    assert all(not dep.startswith("torchvision") for dep in base_dependencies)


def test_local_model_groups_explicitly_declare_torch_dependency() -> None:
    pyproject = _load_pyproject()
    groups = pyproject["dependency-groups"]

    expected_torch_groups = {
        "deepseek-ocr",
        "deepseek-ocr2",
        "dots-ocr",
        "gemma3",
        "glm-ocr",
        "got-ocr",
        "hunyuan-ocr",
        "lighton-ocr2",
        "nanonets-ocr2",
        "paddle-ocr-vl",
        "qwen3-vl",
        "typhoon-ocr",
        "varco-ocr",
    }

    for group_name in expected_torch_groups:
        assert any(dep.startswith("torch") for dep in groups[group_name]), group_name


def test_glm_ocr_group_does_not_pin_incompatible_pyarrow() -> None:
    pyproject = _load_pyproject()
    glm_group = pyproject["dependency-groups"]["glm-ocr"]

    assert all(not dep.startswith("pyarrow") for dep in glm_group)
