import yaml

from src.generation.options import GenerationOptions, GenerationTaskContext, PublishOptions
from src.generation.readme_builder import build_dataset_readme


def _front_matter(card: str) -> dict:
    return yaml.safe_load(card.split("---")[1])


def test_card_front_matter_only_uses_values_the_hub_accepts() -> None:
    context = GenerationTaskContext(
        lang="ko",
        size=1000,
        generation=GenerationOptions(seed=1),
        publish=PublishOptions(repo_id="demo/repo", license="cc-by-sa-3.0", text_source="Korean WikiText"),
    )

    card = build_dataset_readme("demo/repo", context, 1000, {"train": 900, "test": 100})
    meta = _front_matter(card)

    # The Hub warns about task ids outside its official list; "optical-character-recognition" is not in it.
    assert "task_ids" not in meta
    assert meta["task_categories"] == ["image-to-text"]
    assert meta["license"] == "cc-by-sa-3.0"
    assert meta["language"] == ["ko"]
    assert "Korean WikiText" in card
