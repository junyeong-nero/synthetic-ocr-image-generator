from typing import Dict, Type
from src.generator.base import BaseGenerator
from src.generator.generator import Generator

class GeneratorRegistry:
    _generators: Dict[str, Type[BaseGenerator]] = {
        "markdown": Generator,
    }

    @classmethod
    def get_generator_class(cls, name: str) -> Type[BaseGenerator]:
        if name not in cls._generators:
            raise ValueError(f"Unknown generator: {name}")
        return cls._generators[name]
