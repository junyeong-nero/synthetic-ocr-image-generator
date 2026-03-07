from typing import Any, Optional


DEFAULT_NOVELTY_WINDOW = 80
DEFAULT_NOVELTY_THRESHOLD = 0.95
DEFAULT_NOVELTY_MAX_ATTEMPTS = 4
A4_MAX_WIDTH_PX = 2480
A4_MAX_HEIGHT_PX = 3508
MAX_RENDER_ASPECT_RATIO = 2.0


def coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_ratio(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, ratio))


def coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def normalize_choice(
    value: Any,
    allowed: set[str],
    fallback: str,
    warning_label: str,
    logger,
) -> str:
    normalized = str(value).strip().lower()
    if normalized in allowed:
        return normalized
    logger.warning(
        "Unknown %s '%s'. Falling back to '%s'.",
        warning_label,
        normalized,
        fallback,
    )
    return fallback


def resolve_effect_settings(
    *,
    enabled_key: str,
    ratio_key: str,
    enabled_default: bool,
    ratio_default: float,
    kwargs: dict[str, Any],
) -> tuple[bool, float]:
    enabled = coerce_bool(kwargs.get(enabled_key), enabled_default)
    ratio = coerce_ratio(kwargs.get(ratio_key), ratio_default)
    if enabled_key in kwargs and kwargs.get(enabled_key) is not None:
        ratio = 1.0 if enabled else 0.0
    return enabled, ratio
