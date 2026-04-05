from __future__ import annotations

import argparse


def run_list_backends(_: argparse.Namespace) -> None:
    from src.models.registry import list_backends

    print("\nAvailable inference backends:")
    print("-" * 40)
    for name, description in list_backends().items():
        print(f"  {name:12} - {description}")
    print()


def run_list_configs(_: argparse.Namespace) -> None:
    from src.evaluation.model_config import ModelConfigLoader

    loader = ModelConfigLoader()
    config_paths = loader.list_public_config_paths()

    print("\nAvailable model configurations:")
    print("-" * 70)
    print("  {:<30} {:<20} {}".format("Config Name", "Dependency Group", "Backend"))
    print("-" * 70)
    if config_paths:
        for config_path in config_paths:
            config_name = config_path.stem
            try:
                config = loader.load_from_path(config_path)
                dep_group = config.dependency_group or "-"
                backend = config.backend
            except Exception:
                dep_group = "?"
                backend = "?"
            print(f"  {config_name:<30} {dep_group:<20} {backend}")
    else:
        print("  No model configs found in configs/models/")
    print("-" * 70)
    print("\nConfig search paths:")
    for path in loader.config_dirs:
        exists = "+" if path.exists() else "-"
        print(f"  {exists} {path}")
    print()
