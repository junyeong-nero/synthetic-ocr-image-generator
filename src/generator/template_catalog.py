import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE_DIR = Path("configs") / "generator" / "templates"
_DEFAULT_SECTION_TEMPLATE: Dict[str, Any] = {
    "id": "default",
    "family": "sections",
    "complexity": 2,
    "mode": "sections",
    "blueprint": {
        "text": {"section_count": [3, 5], "max_line_chars": 72},
        "table": {"section_count": [1, 2], "rows": [2, 4], "columns": [3, 4]},
        "formula": {"section_count": [1, 2]},
    },
}


def canonicalize_template_ref(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass
class TemplateSpec:
    template_id: str
    family: str = "sections"
    complexity: int = 2
    weight: float = 1.0
    mode: str = "sections"
    blueprint: Optional[Dict[str, Any]] = None
    aliases: Optional[List[str]] = None
    version: str = "2"
    source: str = "builtin"
    tags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.blueprint is None:
            self.blueprint = {}
        if self.aliases is None:
            self.aliases = []
        if self.tags is None:
            self.tags = []

    def refs(self) -> List[str]:
        refs: List[str] = [canonicalize_template_ref(self.template_id)]
        for alias in self.aliases or []:
            normalized = canonicalize_template_ref(alias)
            if normalized and normalized not in refs:
                refs.append(normalized)
        return refs


class TemplateCatalog:
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else _DEFAULT_TEMPLATE_DIR
        self.templates: Dict[str, TemplateSpec] = {}
        self.alias_to_id: Dict[str, str] = {}
        self._loaded = False

    @staticmethod
    def _extract_entries(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            templates = data.get("templates")
            if isinstance(templates, list):
                catalog_version = data.get("version")
                entries: List[Dict[str, Any]] = []
                for item in templates:
                    if not isinstance(item, dict):
                        continue
                    entry = dict(item)
                    if catalog_version is not None and "version" not in entry:
                        entry["version"] = catalog_version
                    entries.append(entry)
                return entries
            if any(key in data for key in ("text", "table", "formula")):
                return [
                    {
                        "id": data.get("id", "default"),
                        "family": data.get("family", "sections"),
                        "complexity": data.get("complexity", 2),
                        "mode": "sections",
                        "version": data.get("version", "2"),
                        "blueprint": {
                            "text": data.get("text", {}),
                            "table": data.get("table", {}),
                            "formula": data.get("formula", {}),
                        },
                    }
                ]
            if "id" in data:
                return [data]
        return []

    @staticmethod
    def _coerce_spec(raw: Dict[str, Any], source: str) -> Optional[TemplateSpec]:
        template_id = canonicalize_template_ref(str(raw.get("id", "")))
        if not template_id:
            return None

        mode = str(raw.get("mode", "sections")).strip().lower()
        if mode != "sections":
            logger.warning(
                "Unsupported template mode '%s' for '%s'. Coercing to sections.",
                mode,
                template_id,
            )
            mode = "sections"

        family = str(raw.get("family") or mode).strip().lower() or mode

        try:
            complexity = int(raw.get("complexity", 2))
        except (TypeError, ValueError):
            complexity = 2
        complexity = max(1, min(5, complexity))

        try:
            weight = float(raw.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        weight = max(0.01, weight)

        aliases_raw = raw.get("aliases", [])
        aliases: List[str] = []
        if isinstance(aliases_raw, list):
            for item in aliases_raw:
                if isinstance(item, str) and item.strip():
                    aliases.append(item.strip())

        tags_raw = raw.get("tags", [])
        tags: List[str] = []
        if isinstance(tags_raw, list):
            for item in tags_raw:
                if isinstance(item, str) and item.strip():
                    tags.append(item.strip())

        blueprint_raw = raw.get("blueprint")
        blueprint: Dict[str, Any] = blueprint_raw if isinstance(blueprint_raw, dict) else {}

        return TemplateSpec(
            template_id=template_id,
            family=family,
            complexity=complexity,
            weight=weight,
            mode=mode,
            blueprint=blueprint,
            aliases=aliases,
            version=str(raw.get("version", "2")),
            source=source,
            tags=tags,
        )

    def load(self) -> None:
        template_by_id: Dict[str, TemplateSpec] = {}

        if self.config_dir.exists():
            import yaml

            for yaml_path in sorted(self.config_dir.glob("*.y*ml")):
                try:
                    with open(yaml_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except Exception as exc:
                    logger.warning("Failed to read template catalog '%s': %s", yaml_path, exc)
                    continue

                for raw in self._extract_entries(data):
                    spec = self._coerce_spec(raw, source=str(yaml_path))
                    if spec is not None:
                        template_by_id[spec.template_id] = spec

        if not template_by_id:
            fallback = self._coerce_spec(dict(_DEFAULT_SECTION_TEMPLATE), source="builtin")
            if fallback is not None:
                template_by_id[fallback.template_id] = fallback

        self.templates = template_by_id
        self.alias_to_id = {}
        for template_id, spec in self.templates.items():
            for ref in spec.refs():
                self.alias_to_id[ref] = template_id

        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def all_specs(self) -> List[TemplateSpec]:
        self._ensure_loaded()
        return [self.templates[key] for key in sorted(self.templates)]

    def get(self, template_ref: str) -> Optional[TemplateSpec]:
        self._ensure_loaded()
        template_id = self.alias_to_id.get(canonicalize_template_ref(template_ref))
        if not template_id:
            return None
        return self.templates.get(template_id)

    def resolve(
        self,
        template: Optional[str],
        template_family: Optional[str],
        min_complexity: Optional[int],
        max_complexity: Optional[int],
    ) -> List[TemplateSpec]:
        self._ensure_loaded()

        if template:
            resolved = self.get(template)
            if resolved is not None:
                return [resolved]
            logger.warning("Unknown template '%s'; applying filters over full catalog.", template)

        candidates = self.all_specs()
        if template_family:
            family = template_family.strip().lower()
            candidates = [spec for spec in candidates if spec.family == family]
        if min_complexity is not None:
            candidates = [spec for spec in candidates if spec.complexity >= min_complexity]
        if max_complexity is not None:
            candidates = [spec for spec in candidates if spec.complexity <= max_complexity]

        if not candidates:
            logger.warning("Template filters returned no candidates; falling back to full catalog.")
            return self.all_specs()

        return candidates


def parse_coverage_targets(raw: Any) -> Dict[str, float]:
    if raw is None:
        return {}

    parsed: Dict[str, float] = {}

    def put(key: str, value: Any) -> None:
        normalized = key.strip().lower()
        if not normalized:
            return
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return
        parsed[normalized] = max(0.0, min(1.0, ratio))

    if isinstance(raw, dict):
        for key, value in raw.items():
            put(str(key), value)
        return parsed

    items: List[str] = []
    if isinstance(raw, str):
        items.extend(token for token in raw.split(",") if token.strip())
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, str):
                items.extend(token for token in item.split(",") if token.strip())

    for item in items:
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            continue
        put(key, value)

    return parsed
