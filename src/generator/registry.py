from typing import Dict, Type
from generator.base import BaseGenerator
from generator.sentence_generator import SentenceGenerator
from generator.table_generator import TableGenerator
from generator.document_generator import DocumentGenerator
from generator.markdown_generator import MarkdownGenerator
from generator.kie_generator import KIEGenerator

class GeneratorRegistry:
    _generators: Dict[str, Type[BaseGenerator]] = {
        "sentence": SentenceGenerator,
        "table": TableGenerator,
        "document": DocumentGenerator,
        "markdown": MarkdownGenerator,
        "kie": KIEGenerator,
    }

    @classmethod
    def get_generator_class(cls, name: str) -> Type[BaseGenerator]:
        if name not in cls._generators:
            raise ValueError(f"Unknown generator: {name}")
        return cls._generators[name]
