from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

CATALOG_SCHEMA_VERSION = "3.0"
SUPPORTED_SCHEMA_VERSIONS = {"2.0", "2.1", "3.0"}
SUPPORTED_TASK_TYPES = {
    "text_to_image",
    "image_edit",
    "upscaling",
    "background_removal",
}
QUALITY_TIERS = {"entry", "standard", "pro", "premium"}
SPEED_TIERS = {"fast", "balanced", "slow"}

_FAST_MODEL_IDS = {
    "z-image-turbo",
    "venice-sd35",
    "hidream",
    "seedream-v5-lite",
    "nano-banana-2",
    "qwen-image",
    "chroma",
    "anime-wai",
}

_SLOW_MODEL_IDS = {
    "flux-2-max",
    "nano-banana-pro",
    "recraft-v4-pro",
    "qwen-image-2-pro",
    "qwen-image-2-pro-edit",
    "nano-banana-pro-edit",
    "flux-2-max-edit",
}


class ModelCatalogError(ValueError):
    """Raised when the model catalog is invalid or cannot be loaded."""


def _default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "image_models.json"


def _parse_usd(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    normalized = value.replace("$", "").replace("USD", "").replace(",", "").strip()
    try:
        return float(normalized)
    except ValueError:
        return None


def _infer_task_types(model_id: str) -> list[str]:
    if model_id == "background-remover":
        return ["background_removal"]
    if model_id == "upscaler":
        return ["upscaling"]
    if model_id.endswith("-edit") or model_id.startswith("qwen-edit"):
        return ["image_edit"]
    return ["text_to_image"]


def _infer_capabilities(task_types: list[str]) -> dict[str, bool]:
    return {
        "supports_generation": "text_to_image" in task_types,
        "supports_edit": "image_edit" in task_types,
        "supports_upscaling": "upscaling" in task_types,
        "supports_background_removal": "background_removal" in task_types,
    }


def _infer_quality_tier(price: Optional[float]) -> str:
    if price is None:
        return "standard"
    if price <= 0.01:
        return "entry"
    if price <= 0.05:
        return "standard"
    if price <= 0.10:
        return "pro"
    return "premium"


def _infer_speed_tier(model_id: str) -> str:
    if model_id in _FAST_MODEL_IDS:
        return "fast"
    if model_id in _SLOW_MODEL_IDS:
        return "slow"
    return "balanced"


def _default_use_cases(model_id: str, task_types: list[str]) -> list[str]:
    task = task_types[0]

    if task == "text_to_image":
        cases = ["general_generation", "creative_concepts"]
        if "recraft" in model_id:
            cases.append("graphic_design")
        if "anime" in model_id:
            cases.append("anime_manga")
        if "gpt-image" in model_id:
            cases.append("semantic_accuracy")
        if model_id in {"z-image-turbo", "seedream-v5-lite", "hidream"}:
            cases.append("rapid_prototyping")
        if "lustify" in model_id:
            cases.append("photorealistic_human_figures")
        if model_id == "chroma":
            cases.append("vivid_color_scenes")
        return cases

    if task == "image_edit":
        return ["inpainting", "object_replacement", "targeted_modification"]

    if task == "upscaling":
        return ["resolution_enhancement", "detail_recovery"]

    if task == "background_removal":
        return ["subject_isolation", "transparent_background_assets"]

    return ["general_use"]


def _default_avoid_use_cases(task_types: list[str]) -> list[str]:
    task = task_types[0]

    if task == "text_to_image":
        return ["pixel_perfect_logos_without_post_editing"]
    if task == "image_edit":
        return ["full_scene_generation_from_scratch"]
    if task == "upscaling":
        return ["content_generation"]
    if task == "background_removal":
        return ["complex_manual_masking_with_hair_fine_tuning"]

    return []


def _normalize_task_type(task_type: str) -> str:
    normalized = task_type.strip().lower().replace(" ", "_")
    aliases = {
        "generate": "text_to_image",
        "generation": "text_to_image",
        "text2image": "text_to_image",
        "text_to_image": "text_to_image",
        "edit": "image_edit",
        "image_edit": "image_edit",
        "inpaint": "image_edit",
        "upscale": "upscaling",
        "upscaling": "upscaling",
        "background_removal": "background_removal",
        "remove_background": "background_removal",
        "background_remove": "background_removal",
    }
    return aliases.get(normalized, normalized)


def _ensure_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelCatalogError(f"Invalid '{field}': expected non-empty string.")
    return value.strip()


def _ensure_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ModelCatalogError(f"Invalid '{field}': expected non-empty list of strings.")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ModelCatalogError(f"Invalid '{field}': all entries must be non-empty strings.")
        normalized.append(item.strip())

    return normalized


def _normalize_card_v2(raw_card: dict[str, Any]) -> dict[str, Any]:
    model_id = _ensure_string(raw_card.get("id"), "id")
    name = _ensure_string(raw_card.get("name", model_id), "name")
    description = _ensure_string(raw_card.get("description", "No description provided."), "description")

    task_types_raw = raw_card.get("task_types")
    if not task_types_raw:
        task_types = _infer_task_types(model_id)
    else:
        task_types = [_normalize_task_type(task) for task in _ensure_string_list(task_types_raw, "task_types")]

    unknown = [task for task in task_types if task not in SUPPORTED_TASK_TYPES]
    if unknown:
        raise ModelCatalogError(
            f"Model '{model_id}' has unsupported task_types: {', '.join(sorted(set(unknown)))}"
        )

    # Keep task type order stable while removing duplicates.
    task_types = list(dict.fromkeys(task_types))

    legacy_per_image = _parse_usd(raw_card.get("cost_per_image"))
    legacy_per_edit = _parse_usd(raw_card.get("cost_per_edit"))

    pricing_raw = raw_card.get("pricing", {})
    if pricing_raw is None:
        pricing_raw = {}
    if not isinstance(pricing_raw, dict):
        raise ModelCatalogError(f"Model '{model_id}': 'pricing' must be an object.")

    per_image = _parse_usd(pricing_raw.get("per_image"))
    per_edit = _parse_usd(pricing_raw.get("per_edit"))

    if per_image is None:
        per_image = legacy_per_image
    if per_edit is None:
        per_edit = legacy_per_edit

    pricing = {
        "currency": str(pricing_raw.get("currency") or "USD"),
        "per_image": per_image,
        "per_edit": per_edit,
    }

    status = str(raw_card.get("status") or "active").strip().lower()
    if status not in {"active", "deprecated", "disabled"}:
        raise ModelCatalogError(
            f"Model '{model_id}': status must be one of active/deprecated/disabled, got '{status}'."
        )

    capabilities_raw = raw_card.get("capabilities")
    if capabilities_raw is None:
        capabilities = _infer_capabilities(task_types)
    else:
        if not isinstance(capabilities_raw, dict):
            raise ModelCatalogError(f"Model '{model_id}': 'capabilities' must be an object.")
        capabilities = {
            "supports_generation": bool(capabilities_raw.get("supports_generation")),
            "supports_edit": bool(capabilities_raw.get("supports_edit")),
            "supports_upscaling": bool(capabilities_raw.get("supports_upscaling")),
            "supports_background_removal": bool(capabilities_raw.get("supports_background_removal")),
        }

    reference_price = per_edit if "image_edit" in task_types else per_image

    quality_tier = str(raw_card.get("quality_tier") or _infer_quality_tier(reference_price)).strip().lower()
    if quality_tier not in QUALITY_TIERS:
        raise ModelCatalogError(
            f"Model '{model_id}': quality_tier must be one of {sorted(QUALITY_TIERS)}, got '{quality_tier}'."
        )

    speed_tier = str(raw_card.get("speed_tier") or _infer_speed_tier(model_id)).strip().lower()
    if speed_tier not in SPEED_TIERS:
        raise ModelCatalogError(
            f"Model '{model_id}': speed_tier must be one of {sorted(SPEED_TIERS)}, got '{speed_tier}'."
        )

    recommended = raw_card.get("recommended_use_cases")
    if recommended is None:
        recommended_use_cases = _default_use_cases(model_id, task_types)
    else:
        recommended_use_cases = _ensure_string_list(recommended, "recommended_use_cases")

    avoid = raw_card.get("avoid_use_cases")
    if avoid is None:
        avoid_use_cases = _default_avoid_use_cases(task_types)
    else:
        avoid_use_cases = _ensure_string_list(avoid, "avoid_use_cases")

    aliases_raw = raw_card.get("aliases", [])
    if aliases_raw is None:
        aliases_raw = []
    if not isinstance(aliases_raw, list):
        raise ModelCatalogError(f"Model '{model_id}': aliases must be a list.")

    aliases: list[str] = []
    for alias in aliases_raw:
        if isinstance(alias, str) and alias.strip():
            aliases.append(alias.strip())

    normalized = {
        "id": model_id,
        "name": name,
        "description": description,
        "task_types": task_types,
        "status": status,
        "capabilities": capabilities,
        "pricing": pricing,
        "quality_tier": quality_tier,
        "speed_tier": speed_tier,
        "recommended_use_cases": recommended_use_cases,
        "avoid_use_cases": avoid_use_cases,
        "aliases": aliases,
    }

    # Preserve legacy pricing keys for backward compatibility with older tooling.
    if raw_card.get("cost_per_image") is not None:
        normalized["cost_per_image"] = raw_card.get("cost_per_image")
    if raw_card.get("cost_per_edit") is not None:
        normalized["cost_per_edit"] = raw_card.get("cost_per_edit")

    recommended_api_params = raw_card.get("recommended_api_params")
    if recommended_api_params is not None:
        if not isinstance(recommended_api_params, dict):
            raise ModelCatalogError(
                f"Model '{model_id}': recommended_api_params must be an object when provided."
            )
        normalized["recommended_api_params"] = deepcopy(recommended_api_params)

    # Preserve extra card fields for forward compatibility with newer model-card schemas.
    known_fields = {
        "id",
        "name",
        "description",
        "task_types",
        "status",
        "capabilities",
        "pricing",
        "quality_tier",
        "speed_tier",
        "recommended_use_cases",
        "avoid_use_cases",
        "aliases",
        "cost_per_image",
        "cost_per_edit",
        "recommended_api_params",
    }
    for key, value in raw_card.items():
        if key in known_fields:
            continue
        normalized[key] = deepcopy(value)

    return normalized


def _migrate_catalog(raw_catalog: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_catalog, dict):
        raise ModelCatalogError("Catalog root must be a JSON object.")

    cards_raw = raw_catalog.get("image_models", [])
    if not isinstance(cards_raw, list):
        raise ModelCatalogError("'image_models' must be a list.")

    normalized_cards = [_normalize_card_v2(card if isinstance(card, dict) else {}) for card in cards_raw]

    raw_schema_version = str(raw_catalog.get("schema_version") or "").strip() or "2.0"
    if raw_schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ModelCatalogError(
            f"Unsupported catalog schema_version '{raw_schema_version}'. "
            f"Expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )

    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "provider": str(raw_catalog.get("provider") or "venice.ai"),
        "catalog_version": str(raw_catalog.get("catalog_version") or raw_catalog.get("last_updated") or "unknown"),
        "last_updated": str(raw_catalog.get("last_updated") or "unknown"),
        "default_task_type": _normalize_task_type(
            str(raw_catalog.get("default_task_type") or "text_to_image")
        ),
        "supported_task_types": [
            task for task in raw_catalog.get("supported_task_types", sorted(SUPPORTED_TASK_TYPES))
            if _normalize_task_type(str(task)) in SUPPORTED_TASK_TYPES
        ],
        "image_models": normalized_cards,
    }

    if not catalog["supported_task_types"]:
        catalog["supported_task_types"] = sorted(SUPPORTED_TASK_TYPES)

    if catalog["default_task_type"] not in SUPPORTED_TASK_TYPES:
        catalog["default_task_type"] = "text_to_image"

    # Ensure task list uniqueness while preserving order.
    catalog["supported_task_types"] = list(dict.fromkeys(catalog["supported_task_types"]))

    _validate_catalog(catalog)
    return catalog


def _validate_catalog(catalog: dict[str, Any]) -> None:
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ModelCatalogError(
            f"Unsupported catalog schema_version '{catalog.get('schema_version')}'. "
            f"Expected '{CATALOG_SCHEMA_VERSION}'."
        )

    _ensure_string(catalog.get("provider"), "provider")
    _ensure_string(catalog.get("catalog_version"), "catalog_version")
    _ensure_string(catalog.get("last_updated"), "last_updated")

    default_task_type = _normalize_task_type(_ensure_string(catalog.get("default_task_type"), "default_task_type"))
    if default_task_type not in SUPPORTED_TASK_TYPES:
        raise ModelCatalogError(f"Invalid default_task_type '{default_task_type}'.")

    supported_task_types = _ensure_string_list(catalog.get("supported_task_types"), "supported_task_types")
    normalized_supported = [_normalize_task_type(task) for task in supported_task_types]

    unknown = [task for task in normalized_supported if task not in SUPPORTED_TASK_TYPES]
    if unknown:
        raise ModelCatalogError(f"Unsupported task types in catalog: {', '.join(sorted(set(unknown)))}")

    cards = catalog.get("image_models")
    if not isinstance(cards, list) or not cards:
        raise ModelCatalogError("Catalog must include a non-empty 'image_models' list.")

    seen_ids: set[str] = set()
    for card in cards:
        card_id = _ensure_string(card.get("id"), "id")
        if card_id in seen_ids:
            raise ModelCatalogError(f"Duplicate model id detected: '{card_id}'.")
        seen_ids.add(card_id)


def _resolve_catalog_path(catalog_path: Optional[str | Path]) -> Path:
    if catalog_path is None:
        return _default_catalog_path()
    return Path(catalog_path).expanduser().resolve()


@lru_cache(maxsize=8)
def _load_catalog_cached(path_str: str) -> dict[str, Any]:
    catalog_path = Path(path_str)
    if not catalog_path.exists():
        raise ModelCatalogError(f"Model catalog file not found: {catalog_path}")

    try:
        raw_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelCatalogError(f"Invalid JSON in model catalog '{catalog_path}': {exc}") from exc

    return _migrate_catalog(raw_catalog)


def clear_catalog_cache() -> None:
    """Clear cached catalog reads, forcing the next load to re-read disk."""
    _load_catalog_cached.cache_clear()


def load_catalog(catalog_path: Optional[str | Path] = None, use_cache: bool = True) -> dict[str, Any]:
    """
    Load and validate the model catalog.

    The loader supports both v2 and legacy v1-shaped JSON and always returns a normalized v2 structure.
    """
    resolved_path = _resolve_catalog_path(catalog_path)

    if use_cache:
        return deepcopy(_load_catalog_cached(str(resolved_path)))

    clear_catalog_cache()
    return deepcopy(_load_catalog_cached(str(resolved_path)))


def list_model_cards(
    task_type: Optional[str] = None,
    include_inactive: bool = False,
    catalog: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """List model cards, optionally filtered by task type and status."""
    active_catalog = catalog or load_catalog()
    cards = active_catalog.get("image_models", [])

    normalized_task = _normalize_task_type(task_type) if task_type else None
    if normalized_task and normalized_task not in SUPPORTED_TASK_TYPES:
        raise ModelCatalogError(f"Unsupported task_type '{task_type}'.")

    filtered: list[dict[str, Any]] = []
    for card in cards:
        if not include_inactive and card.get("status") != "active":
            continue

        card_tasks = card.get("task_types", [])
        if normalized_task and normalized_task not in card_tasks:
            continue

        filtered.append(deepcopy(card))

    return filtered


def get_model_card(
    model_id: str,
    catalog: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Get a model card by exact model id or alias."""
    lookup_id = model_id.strip()
    if not lookup_id:
        raise ModelCatalogError("model_id must be a non-empty string.")

    active_catalog = catalog or load_catalog()
    for card in active_catalog.get("image_models", []):
        if card.get("id") == lookup_id:
            return deepcopy(card)
        aliases = card.get("aliases", [])
        if lookup_id in aliases:
            return deepcopy(card)

    return None


def _task_price_reference(card: dict[str, Any], task_type: str) -> float:
    pricing = card.get("pricing", {})
    per_image = pricing.get("per_image")
    per_edit = pricing.get("per_edit")

    if task_type == "image_edit":
        value = per_edit if per_edit is not None else per_image
    elif task_type == "text_to_image":
        value = per_image if per_image is not None else per_edit
    else:
        value = per_image if per_image is not None else per_edit

    return float(value) if value is not None else float("inf")


def _quality_rank(tier: str) -> int:
    return {
        "entry": 1,
        "standard": 2,
        "pro": 3,
        "premium": 4,
    }.get(tier, 0)


def _speed_rank(tier: str) -> int:
    return {
        "fast": 3,
        "balanced": 2,
        "slow": 1,
    }.get(tier, 0)


def find_alternatives(
    model_id: str,
    task_type: Optional[str] = None,
    max_results: int = 3,
    include_inactive: bool = False,
    catalog: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Return similar alternatives in the same task family.

    Ranking preference: higher quality tier, then faster speed tier, then lower price.
    """
    if max_results < 1:
        return []

    active_catalog = catalog or load_catalog()
    current = get_model_card(model_id, catalog=active_catalog)

    normalized_task: Optional[str]
    if task_type:
        normalized_task = _normalize_task_type(task_type)
        if normalized_task not in SUPPORTED_TASK_TYPES:
            raise ModelCatalogError(f"Unsupported task_type '{task_type}'.")
    elif current and current.get("task_types"):
        normalized_task = current["task_types"][0]
    else:
        normalized_task = active_catalog.get("default_task_type", "text_to_image")

    candidates = list_model_cards(
        task_type=normalized_task,
        include_inactive=include_inactive,
        catalog=active_catalog,
    )

    ranked = []
    for card in candidates:
        if card.get("id") == model_id:
            continue

        ranked.append(
            (
                -_quality_rank(str(card.get("quality_tier", ""))),
                -_speed_rank(str(card.get("speed_tier", ""))),
                _task_price_reference(card, normalized_task),
                str(card.get("id", "")),
                card,
            )
        )

    ranked.sort(key=lambda item: item[:4])
    return [deepcopy(item[4]) for item in ranked[:max_results]]


def get_catalog_metadata(catalog: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return top-level catalog metadata without all cards."""
    active_catalog = catalog or load_catalog()
    return {
        "schema_version": active_catalog.get("schema_version"),
        "provider": active_catalog.get("provider"),
        "catalog_version": active_catalog.get("catalog_version"),
        "last_updated": active_catalog.get("last_updated"),
        "default_task_type": active_catalog.get("default_task_type"),
        "supported_task_types": list(active_catalog.get("supported_task_types", [])),
        "model_count": len(active_catalog.get("image_models", [])),
    }
