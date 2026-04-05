from src.generator.base import BaseGenerator as BaseGenerator

__all__ = ["BaseGenerator", "Generator"]


def __getattr__(name: str):
    if name == "Generator":
        from src.generator.generator import Generator

        return Generator
    raise AttributeError(f"module 'generator' has no attribute {name!r}")
