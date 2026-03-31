from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any, Optional

DEFAULT_PROVIDER_PARAMS: dict[str, Any] = {
    "cfg_scale": 7.5,
    "embed_exif_metadata": False,
    "format": "webp",
    "hide_watermark": False,
    "return_binary": False,
    "safe_mode": True,
    "enable_web_search": False,
}

RUNTIME_OVERRIDES_ENV_VAR = "PERCIVAL_IMAGE_MCP_GENERATION_OVERRIDES_JSON"
SUPPORTED_OUTPUT_FORMATS = {"jpeg", "png", "webp"}
_SAFE_PARAM_NAME_RE = re.compile(r"^[A-Za-z0-9_:-]+$")
_ASPECT_RATIO_RE = re.compile(r"^[1-9]\d{0,2}:[1-9]\d{0,2}$")
_RESOLUTION_RE = re.compile(r"^[A-Za-z0-9._-]{1,24}$")
_MAX_STRING_PARAM_CHARS = 4000
_MAX_STYLE_PRESET_CHARS = 120


def parse_parameter_overrides_json(parameter_overrides_json: Optional[str]) -> dict[str, Any]:
    """Parse optional JSON object with runtime overrides for generation parameters."""
    raw = (parameter_overrides_json or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"parameter_overrides_json must be valid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("parameter_overrides_json must decode to a JSON object.")
    return parsed


def _load_runtime_overrides_from_env() -> dict[str, Any]:
    raw = (os.getenv(RUNTIME_OVERRIDES_ENV_VAR) or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{RUNTIME_OVERRIDES_ENV_VAR} must be valid JSON object when set: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{RUNTIME_OVERRIDES_ENV_VAR} must decode to a JSON object.")
    return parsed


def _as_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean.")


def _as_int(key: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    return parsed


def _as_float(key: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    return parsed


def _as_string(key: str, value: Any, *, max_chars: int = _MAX_STRING_PARAM_CHARS) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{key} must be a non-empty string.")
    if len(normalized) > max_chars:
        raise ValueError(f"{key} exceeds max length of {max_chars}.")
    return normalized


def _normalize_provider_param(key: str, value: Any) -> Any:
    if key in {"safe_mode", "hide_watermark", "embed_exif_metadata", "enable_web_search", "return_binary"}:
        return _as_bool(key, value)

    if key in {"steps"}:
        return _as_int(key, value, 1, 150)
    if key in {"seed"}:
        return _as_int(key, value, 0, 2_147_483_647)
    if key in {"variants"}:
        return _as_int(key, value, 1, 8)
    if key in {"lora_strength"}:
        return _as_int(key, value, 0, 100)
    if key in {"width", "height"}:
        return _as_int(key, value, 64, 4096)

    if key in {"cfg_scale"}:
        return _as_float(key, value, 0.0, 50.0)

    if key == "aspect_ratio":
        normalized = _as_string(key, value, max_chars=16)
        if not _ASPECT_RATIO_RE.fullmatch(normalized):
            raise ValueError("aspect_ratio must match '<w>:<h>' (e.g., '16:9').")
        return normalized

    if key == "resolution":
        normalized = _as_string(key, value, max_chars=24)
        if not _RESOLUTION_RE.fullmatch(normalized):
            raise ValueError("resolution contains invalid characters.")
        return normalized

    if key == "format":
        normalized = _as_string(key, value, max_chars=8).lower()
        if normalized not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"format must be one of {sorted(SUPPORTED_OUTPUT_FORMATS)}.")
        return normalized

    if key == "style_preset":
        return _as_string(key, value, max_chars=_MAX_STYLE_PRESET_CHARS)

    if key in {"negative_prompt", "inpaint"}:
        return _as_string(key, value)

    if not _SAFE_PARAM_NAME_RE.fullmatch(key):
        raise ValueError("parameter name contains unsupported characters.")

    # Forward-compatible pass-through for scalar provider params.
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _as_string(key, value)
    raise ValueError("unsupported parameter type; use scalar values only.")


def _merge_param_sources(
    *,
    defaults: dict[str, Any],
    card_recommended_params: dict[str, Any],
    runtime_overrides: dict[str, Any],
    explicit_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    ordered_sources = (
        ("defaults", defaults),
        ("catalog_recommended", card_recommended_params),
        ("runtime_overrides", runtime_overrides),
        ("explicit_input", explicit_params),
    )
    for source_name, source_params in ordered_sources:
        if not source_params:
            continue
        if not isinstance(source_params, dict):
            raise ValueError(f"{source_name} must be a JSON object.")
        for key, value in source_params.items():
            if value is None:
                continue
            merged[key] = value
            sources[key] = source_name
    return merged, sources


def build_venice_generation_request(
    *,
    model: str,
    prompt: str,
    size: str,
    explicit_params: Optional[dict[str, Any]] = None,
    card_recommended_params: Optional[dict[str, Any]] = None,
    runtime_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build validated request payload for Venice image generation.

    Precedence:
    explicit input > runtime overrides > catalog recommendations > server defaults.
    """
    explicit = deepcopy(explicit_params or {})
    recommended = deepcopy(card_recommended_params or {})
    runtime = deepcopy(runtime_overrides or {})
    env_overrides = _load_runtime_overrides_from_env()
    if env_overrides:
        runtime.update(env_overrides)

    merged, sources = _merge_param_sources(
        defaults=deepcopy(DEFAULT_PROVIDER_PARAMS),
        card_recommended_params=recommended,
        runtime_overrides=runtime,
        explicit_params=explicit,
    )

    resolved_provider_params: dict[str, Any] = {}
    resolved_sources: dict[str, str] = {}
    for key, value in merged.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        resolved_provider_params[normalized_key] = _normalize_provider_param(normalized_key, value)
        resolved_sources[normalized_key] = sources.get(key, "unknown")

    openai_request: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
    }
    if resolved_provider_params:
        openai_request["extra_body"] = resolved_provider_params

    return {
        "openai_request": openai_request,
        "resolved_provider_params": resolved_provider_params,
        "param_sources": resolved_sources,
        "runtime_override_keys": sorted(runtime.keys()),
    }
