import json
import inspect
import time
import re
import difflib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from server import mcp
from utils.config import get_env_bool, get_env_int, get_env_str
from utils.client import (
    jarvina_client as client,
    generate_images_with_transport,
    get_jarvina_base_url,
    get_image_info,
    list_provider_image_styles,
    save_base64_image,
    download_image_from_url,
    ImageStyle,
    ImagePayload,
    GenerationResponse,
)
from utils.model_catalog import (
    find_alternatives as catalog_find_alternatives,
    get_model_card as catalog_get_model_card,
    list_model_cards as catalog_list_model_cards,
)
from utils.nanobot_profile import (
    CONTRACT_VERSION,
    SERVER_NAME,
)
from utils.path_utils import (
    validate_image_path, 
    validate_working_directory,
    sanitize_input_text,
    enforce_path_within_working_dir,
    is_relative_to
)
from utils.security_utils import (
    redact_sensitive_structure,
    redact_sensitive_text,
    record_security_event,
)
from utils.venice_image_payload import (
    build_venice_generation_request,
    parse_parameter_overrides_json,
)

logger = logging.getLogger(__name__)

MODEL_LIST_CACHE_TTL_SECONDS = 300
STYLE_LIST_CACHE_TTL_SECONDS = 300
_provider_models_cache: dict[str, Any] = {
    "model_ids": [],
    "fetched_at": None,
    "expires_at": 0.0,
}
_provider_styles_cache: dict[str, Any] = {
    "styles": [],
    "fetched_at": None,
    "expires_at": 0.0,
}

DEFAULT_CARD_FIELDS = (
    "id",
    "name",
    "task_types",
    "quality_tier",
    "speed_tier",
    "pricing",
    "recommended_api_params",
    "status",
)
_ALLOWED_CARD_FIELDS = {
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

MAX_EDIT_IMAGES_PER_REQUEST = 10
DEFAULT_MAX_PROMPT_CHARS = 4000
DEFAULT_MAX_NEGATIVE_PROMPT_CHARS = 2000
DEFAULT_MAX_FILENAME_PREFIX_CHARS = 80
DEFAULT_MAX_MODEL_ID_CHARS = 128
DEFAULT_MAX_LIST_FILES = 200
DEFAULT_OUTPUT_DIRECTORY = "~/Pictures"
_SAFE_FILENAME_PREFIX_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _new_request_id() -> str:
    return f"img-{int(time.time() * 1000)}-{uuid4().hex[:12]}"


def _get_default_output_directory() -> Path:
    configured = get_env_str("PERCIVAL_IMAGE_MCP_DEFAULT_OUTPUT_DIR", DEFAULT_OUTPUT_DIRECTORY)
    return Path(configured).expanduser().resolve(strict=False)


def _resolve_output_path(
    path_value: str,
    working_path: Path,
    label: str,
) -> tuple[Optional[Path], Optional[str]]:
    raw_candidate = Path(path_value.strip()).expanduser()
    is_absolute_input = raw_candidate.is_absolute()
    candidate = raw_candidate if is_absolute_input else (working_path / raw_candidate)
    resolved_candidate = candidate.resolve(strict=False)
    default_output_root = _get_default_output_directory()

    if is_relative_to(resolved_candidate, working_path):
        return resolved_candidate, None

    # Allow default output root only when caller provided an absolute path.
    if is_absolute_input and is_relative_to(resolved_candidate, default_output_root):
        return resolved_candidate, None

    record_security_event(
        "path_escape_blocked",
        {
            "label": label,
            "provided_path": path_value,
            "resolved_path": str(resolved_candidate),
            "working_dir": str(working_path),
            "default_output_root": str(default_output_root),
        },
    )
    return (
        None,
        (
            f"Error: {label} must resolve inside working_dir or default output directory.\n"
            f"• Provided: '{path_value}'\n"
            f"• Resolved: '{resolved_candidate}'\n"
            f"• working_dir: '{working_path}'\n"
            f"• default_output_dir: '{default_output_root}'"
        ),
    )


async def _get_provider_model_ids(force_refresh: bool = False) -> tuple[list[str], str, bool]:
    """
    Fetch model ids from provider with short-lived cache.
    """
    now_ts = time.time()
    if (
        not force_refresh
        and _provider_models_cache["model_ids"]
        and now_ts < float(_provider_models_cache["expires_at"])
    ):
        return (
            list(_provider_models_cache["model_ids"]),
            str(_provider_models_cache["fetched_at"]),
            True,
        )

    models = await client.models.list()
    model_ids = sorted(
        {
            model.id
            for model in models.data
            if hasattr(model, "id") and isinstance(model.id, str) and model.id.strip()
        }
    )
    fetched_at = _utc_now_iso()
    _provider_models_cache["model_ids"] = model_ids
    _provider_models_cache["fetched_at"] = fetched_at
    _provider_models_cache["expires_at"] = now_ts + MODEL_LIST_CACHE_TTL_SECONDS
    return model_ids, fetched_at, False


async def _get_provider_image_styles(force_refresh: bool = False) -> tuple[list[ImageStyle], str, bool]:
    """
    Fetch image style presets from provider with short-lived cache.
    """
    now_ts = time.time()
    if (
        not force_refresh
        and _provider_styles_cache["styles"]
        and now_ts < float(_provider_styles_cache["expires_at"])
    ):
        return (
            list(_provider_styles_cache["styles"]),
            str(_provider_styles_cache["fetched_at"]),
            True,
        )

    styles = await list_provider_image_styles()
    fetched_at = _utc_now_iso()
    _provider_styles_cache["styles"] = styles
    _provider_styles_cache["fetched_at"] = fetched_at
    _provider_styles_cache["expires_at"] = now_ts + STYLE_LIST_CACHE_TTL_SECONDS
    return styles, fetched_at, False


async def _build_style_validation_payload(
    style_preset: str,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    checked_at = _utc_now_iso()
    selected = style_preset.strip()
    normalized_selected = (selected or "").strip().lower()
    try:
        styles, fetched_at, used_cache = await _get_provider_image_styles(force_refresh=force_refresh)
    except Exception as exc:
        return {
            "ok": False,
            "style_preset": selected,
            "error": f"Falha ao consultar estilos no provedor: {exc}",
            "checked_at": checked_at,
        }

    normalized_map: dict[str, str] = {}
    all_names: list[str] = []
    for entry in styles:
        style_id = str(entry.id or "").strip()
        style_name = str(entry.name or "").strip()
        if style_id:
            normalized_map[style_id.lower()] = style_id
            all_names.append(style_id)
        if style_name:
            normalized_map[style_name.lower()] = style_name
            all_names.append(style_name)

    available = normalized_selected in normalized_map
    suggestions = difflib.get_close_matches(selected, all_names, n=5, cutoff=0.45) if all_names else []

    payload = {
        "ok": True,
        "style_preset": selected,
        "available": available,
        "availability_state": "available" if available else "unavailable",
        "matched_style": normalized_map.get(normalized_selected),
        "suggestions": suggestions,
        "provider_check": {
            "provider_base_url": get_jarvina_base_url(),
            "fetched_at": fetched_at,
            "used_cache": used_cache,
            "provider_style_count": len(styles),
            "ttl_seconds": STYLE_LIST_CACHE_TTL_SECONDS,
        },
        "checked_at": checked_at,
    }
    if available:
        payload["recommendation"] = f"Style preset '{selected}' is available."
    else:
        payload["recommendation"] = "Escolha um style_preset válido da lista de estilos do provedor."
    return payload


def _json_response(payload: dict[str, Any]) -> str:
    # Compact JSON helps avoid nanobot tool-result truncation.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _success_response(
    data: dict[str, Any],
    legacy_text: Optional[str] = None,
    request_id: Optional[str] = None,
    tool_name: Optional[str] = None
) -> str:
    effective_request_id = request_id or _new_request_id()
    frame = inspect.currentframe()
    caller_name = frame.f_back.f_code.co_name if frame and frame.f_back else "unknown_tool"
    effective_tool_name = tool_name or caller_name
    payload: dict[str, Any] = {"ok": True, "data": data}
    payload["request_id"] = effective_request_id
    payload["meta"] = {
        "server": SERVER_NAME,
        "contract_version": CONTRACT_VERSION,
        "request_id": effective_request_id,
        "timestamp": _utc_now_iso(),
        "tool": effective_tool_name,
    }
    if legacy_text:
        payload["legacy_text"] = legacy_text
    return _json_response(payload)


def _error_response(
    error: str,
    code: str = "tool_error",
    details: Optional[dict[str, Any]] = None,
    legacy_text: Optional[str] = None,
    request_id: Optional[str] = None,
    tool_name: Optional[str] = None
) -> str:
    effective_request_id = request_id or _new_request_id()
    frame = inspect.currentframe()
    caller_name = frame.f_back.f_code.co_name if frame and frame.f_back else "unknown_tool"
    effective_tool_name = tool_name or caller_name
    safe_error = redact_sensitive_text(error)
    safe_legacy = redact_sensitive_text(legacy_text) if legacy_text else None
    safe_details = redact_sensitive_structure(details) if details is not None else None
    payload: dict[str, Any] = {"ok": False, "error": error, "code": code}
    payload["error"] = safe_error
    payload["request_id"] = effective_request_id
    payload["meta"] = {
        "server": SERVER_NAME,
        "contract_version": CONTRACT_VERSION,
        "request_id": effective_request_id,
        "timestamp": _utc_now_iso(),
        "tool": effective_tool_name,
    }
    if safe_details is not None:
        payload["details"] = safe_details
    if safe_legacy:
        payload["legacy_text"] = safe_legacy
    return _json_response(payload)


def _normalize_task_type(value: str) -> str:
    normalized = (value or "").strip().lower().replace(" ", "_")
    aliases = {
        "generate": "text_to_image",
        "generation": "text_to_image",
        "text2image": "text_to_image",
        "edit": "image_edit",
        "inpaint": "image_edit",
        "upscale": "upscaling",
        "remove_background": "background_removal",
        "background_remove": "background_removal",
    }
    return aliases.get(normalized, normalized)


def _normalize_model_identifier(value: str) -> str:
    return (value or "").strip().lower().replace("_", "-")


def _catalog_provider_overlap_count(provider_ids: list[str]) -> int:
    """
    Count how many active catalog model IDs (or aliases) are visible in provider /models.

    Some providers expose a generic model list that does not include image models.
    When overlap is zero, strict availability checks based only on /models become unreliable.
    """
    try:
        catalog_cards = catalog_list_model_cards(task_type=None, include_inactive=False)
    except Exception:
        return 0

    provider_exact = set(provider_ids)
    provider_norm = {_normalize_model_identifier(model_id) for model_id in provider_ids}

    overlap = 0
    for card in catalog_cards:
        candidates: list[str] = []
        card_id = card.get("id")
        if isinstance(card_id, str) and card_id.strip():
            candidates.append(card_id.strip())

        aliases = card.get("aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    candidates.append(alias.strip())

        if any(
            candidate in provider_exact
            or _normalize_model_identifier(candidate) in provider_norm
            for candidate in candidates
        ):
            overlap += 1

    return overlap


def _normalize_limit_offset(limit: int, offset: int, max_limit: int = 100) -> tuple[int, int]:
    safe_limit = max(1, min(int(limit), max_limit))
    safe_offset = max(0, int(offset))
    return safe_limit, safe_offset


def _parse_card_fields(fields: Optional[str]) -> tuple[list[str], Optional[str]]:
    if not fields or not fields.strip():
        return list(DEFAULT_CARD_FIELDS), None

    requested = [item.strip() for item in fields.split(",") if item.strip()]
    if not requested:
        return list(DEFAULT_CARD_FIELDS), None

    unknown = [field for field in requested if field not in _ALLOWED_CARD_FIELDS]
    if unknown:
        return [], f"Unknown fields requested: {', '.join(sorted(set(unknown)))}"

    # Deduplicate while keeping order.
    unique_fields = list(dict.fromkeys(requested))
    return unique_fields, None


def _project_card_fields(card: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: card.get(field) for field in fields}


def _validate_working_dir(working_dir: str) -> tuple[Optional[Path], Optional[str]]:
    return validate_working_directory(working_dir)


async def _save_images_from_response(response: GenerationResponse, output_path: Path, filename_prefix: str) -> tuple[list[str], Optional[str]]:
    data_items = response.data
    if not data_items:
        return [], "Error: Provider response does not contain image data."

    saved_files: list[str] = []
    timestamp = int(time.time())

    for i, image_data in enumerate(data_items):
        filename = f"{filename_prefix}_{timestamp}_{i+1}.png"
        file_path = output_path / filename

        if image_data.b64_json:
            if await save_base64_image(image_data.b64_json, file_path):
                saved_files.append(str(file_path))
            else:
                return [], f"Error: Failed to save image {i+1}"
        elif image_data.url:
            if await download_image_from_url(image_data.url, file_path):
                saved_files.append(str(file_path))
            else:
                return [], f"Error: Failed to download image {i+1} from URL"
        else:
            return [], f"Error: No image payload found for image {i+1}"

    return saved_files, None


async def _build_model_availability_payload(
    model_id: str,
    task_type: str = "text_to_image",
    force_refresh: bool = False,
    include_alternatives: bool = True
) -> dict[str, Any]:
    checked_at = _utc_now_iso()
    selected_model_id = model_id.strip()
    normalized_task_type = _normalize_task_type(task_type)

    try:
        provider_ids, fetched_at, used_cache = await _get_provider_model_ids(force_refresh=force_refresh)
    except Exception as exc:
        return {
            "ok": False,
            "model_id": selected_model_id,
            "task_type": normalized_task_type,
            "error": f"Falha ao consultar modelos no provedor: {exc}",
            "checked_at": checked_at,
        }

    provider_id_set = set(provider_ids)
    provider_norm_set = {_normalize_model_identifier(model_id) for model_id in provider_ids}
    card = None
    catalog_error = None
    task_type_matches_card = None
    try:
        card = catalog_get_model_card(selected_model_id)
        if card:
            task_type_matches_card = normalized_task_type in card.get("task_types", [])
    except Exception as exc:
        catalog_error = str(exc)

    lookup_candidates = [selected_model_id]
    if card:
        canonical_id = str(card.get("id") or "").strip()
        if canonical_id:
            lookup_candidates.append(canonical_id)

        aliases = card.get("aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    lookup_candidates.append(alias.strip())

    # Deduplicate while preserving order.
    lookup_candidates = list(dict.fromkeys(lookup_candidates))

    available = any(
        candidate in provider_id_set
        or _normalize_model_identifier(candidate) in provider_norm_set
        for candidate in lookup_candidates
    )

    catalog_overlap = _catalog_provider_overlap_count(provider_ids)
    provider_catalog_visible = catalog_overlap > 0
    provider_scope_unknown = bool(provider_ids) and not provider_catalog_visible

    availability_state = "available" if available else "unavailable"
    if provider_scope_unknown:
        # Provider /models endpoint is likely not image-aware for this account/plan.
        # Avoid false negatives that would block valid image requests.
        availability_state = "unknown"
        if card is not None:
            available = True

    alternatives: list[dict[str, Any]] = []
    if include_alternatives and availability_state == "unavailable":
        try:
            alternatives = catalog_find_alternatives(
                model_id=selected_model_id,
                task_type=normalized_task_type,
                max_results=3
            )
            alternatives = [alt for alt in alternatives if alt.get("id") in provider_id_set]
        except Exception:
            alternatives = []

    payload: dict[str, Any] = {
        "ok": True,
        "model_id": selected_model_id,
        "task_type": normalized_task_type,
        "available": available,
        "availability_state": availability_state,
        "provider_check": {
            "provider_base_url": get_jarvina_base_url(),
            "fetched_at": fetched_at,
            "used_cache": used_cache,
            "provider_model_count": len(provider_ids),
            "catalog_overlap_count": catalog_overlap,
            "catalog_visibility": "visible" if provider_catalog_visible else "not_visible",
        },
        "catalog_check": {
            "found_in_catalog": card is not None,
            "task_type_matches": task_type_matches_card,
        },
        "checked_at": checked_at,
    }

    if card:
        payload["catalog_model"] = {
            "id": card.get("id"),
            "name": card.get("name"),
            "task_types": card.get("task_types", []),
            "quality_tier": card.get("quality_tier"),
            "speed_tier": card.get("speed_tier"),
        }

    if catalog_error:
        payload["catalog_error"] = catalog_error

    if availability_state == "unknown":
        payload["recommendation"] = (
            "O endpoint /models do provedor não expôs modelos do catálogo de imagem; "
            "tratando disponibilidade como incerta para evitar falso negativo."
        )
    elif availability_state == "unavailable":
        payload["recommendation"] = "Escolha um modelo alternativo ativo e tente novamente."
        payload["alternatives"] = alternatives
    else:
        payload["recommendation"] = f"Modelo '{selected_model_id}' está ativo no provedor."

    return payload


async def _enforce_model_precheck(
    model_id: str,
    task_type: str,
    strict_model_check: bool,
    force_model_refresh: bool
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if not strict_model_check:
        return None, None

    check_payload = await _build_model_availability_payload(
        model_id=model_id,
        task_type=task_type,
        force_refresh=force_model_refresh,
        include_alternatives=True,
    )
    if not check_payload.get("ok"):
        return {
            "error": "Could not verify model status before execution.",
            "code": "model_precheck_failed",
            "details": check_payload,
        }, None

    availability_state = str(check_payload.get("availability_state") or "").strip().lower()
    if not availability_state:
        availability_state = "available" if check_payload.get("available") else "unavailable"

    if availability_state == "unavailable":
        return {
            "error": f"Model '{model_id}' is not currently available on provider.",
            "code": "model_not_available",
            "details": check_payload,
        }, None

    catalog_check = check_payload.get("catalog_check", {})
    if not catalog_check.get("found_in_catalog"):
        return {
            "error": (
                f"Model '{model_id}' is active in provider but missing in local model cards. "
                "Use list_model_cards/get_model_card to pick a cataloged model."
            ),
            "code": "model_missing_in_catalog",
            "details": check_payload,
        }, None

    if catalog_check.get("task_type_matches") is False:
        return {
            "error": (
                f"Model '{model_id}' is not classified for task "
                f"'{_normalize_task_type(task_type)}' in model cards."
            ),
            "code": "model_task_mismatch",
            "details": check_payload,
        }, None

    return None, check_payload


_QUALITY_RANK = {"entry": 1, "standard": 2, "pro": 3, "premium": 4}
_SPEED_RANK = {"slow": 1, "balanced": 2, "fast": 3}


def _extract_task_price(card: dict[str, Any], task_type: str) -> Optional[float]:
    pricing = card.get("pricing", {})
    per_image = pricing.get("per_image")
    per_edit = pricing.get("per_edit")
    value: Any
    if task_type == "image_edit":
        value = per_edit if per_edit is not None else per_image
    else:
        value = per_image if per_image is not None else per_edit
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _infer_intent_use_case_hints(intent: str) -> set[str]:
    normalized = (intent or "").strip().lower()
    if not normalized:
        return set()

    mapping = {
        "anime_manga": {"anime", "manga", "otaku"},
        "graphic_design": {"logo", "branding", "brand", "vector", "poster"},
        "semantic_accuracy": {"accurate", "accuracy", "fiel", "preciso", "instruction"},
        "rapid_prototyping": {"fast", "quick", "rápido", "rapido", "prototype", "iteração"},
        "photorealistic_human_figures": {"portrait", "photo", "photoreal", "human", "person"},
        "vivid_color_scenes": {"vibrant", "vivid", "colorful", "cores"},
        "creative_concepts": {"concept", "concept art", "creative", "criativo"},
        "general_generation": {"image", "scene", "render"},
    }
    hints: set[str] = set()
    for use_case, keywords in mapping.items():
        if any(keyword in normalized for keyword in keywords):
            hints.add(use_case)
    return hints


def _infer_preferred_quality_tier(intent: str) -> Optional[str]:
    normalized = (intent or "").strip().lower()
    if not normalized:
        return None
    if any(token in normalized for token in {"premium", "ultra", "best quality", "maximum detail"}):
        return "premium"
    if any(token in normalized for token in {"high quality", "high-detail", "detailed", "pro"}):
        return "pro"
    return None


def _infer_preferred_speed_tier(intent: str) -> Optional[str]:
    normalized = (intent or "").strip().lower()
    if not normalized:
        return None
    if any(token in normalized for token in {"fast", "quick", "rápido", "rapido", "urgent"}):
        return "fast"
    if any(token in normalized for token in {"balanced", "equilibrado"}):
        return "balanced"
    return None


def _normalize_quality_tier(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _QUALITY_RANK else None


def _normalize_speed_tier(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _SPEED_RANK else None


def _compute_tier_alignment_score(
    card_tier: Optional[str],
    preferred_tier: Optional[str],
    rank_map: dict[str, int],
    *,
    exact_bonus: float,
    near_bonus: float,
) -> float:
    if not preferred_tier or not card_tier:
        return 0.0
    if card_tier not in rank_map or preferred_tier not in rank_map:
        return 0.0
    diff = abs(rank_map[card_tier] - rank_map[preferred_tier])
    if diff == 0:
        return exact_bonus
    if diff == 1:
        return near_bonus
@mcp.tool()
async def recommend_model_for_intent(
    task_type: str = "text_to_image",
    intent: str = "",
    max_results: int = 5,
    budget_per_image: Optional[float] = None,
    preferred_quality_tier: Optional[str] = None,
    preferred_speed_tier: Optional[str] = None,
    prioritize_cost: bool = False,
    verify_online: bool = True,
    force_model_refresh: bool = False,
    include_unavailable: bool = False,
    fields: Optional[str] = None,
) -> str:
    """
    Recommend best-fit models for a human intent using model-card metadata.
    """
    request_id = _new_request_id()
    try:
        normalized_task_type = _normalize_task_type(task_type)
        selected_fields, fields_error = _parse_card_fields(fields)
        if fields_error:
            return _error_response(
                error=fields_error, code="invalid_fields", details={"fields": fields}, request_id=request_id
            )

        if max_results < 1:
            return _error_response(
                error="max_results must be >= 1.", code="invalid_max_results", details={"max_results": max_results}, request_id=request_id
            )

        effective_quality_pref = _normalize_quality_tier(preferred_quality_tier) or _infer_preferred_quality_tier(intent)
        effective_speed_pref = _normalize_speed_tier(preferred_speed_tier) or _infer_preferred_speed_tier(intent)

        cards = catalog_list_model_cards(task_type=normalized_task_type, include_inactive=False)
        intent_hints = _infer_intent_use_case_hints(intent)

        provider_ids: list[str] = []
        provider_id_set: set[str] = set()
        provider_norm_set: set[str] = set()
        provider_visibility = "not_checked"
        provider_fetched_at: Optional[str] = None
        provider_used_cache = False

        if verify_online:
            try:
                provider_ids, provider_fetched_at, provider_used_cache = await _get_provider_model_ids(
                    force_refresh=force_model_refresh
                )
                provider_id_set = set(provider_ids)
                provider_norm_set = {_normalize_model_identifier(model_id) for model_id in provider_ids}
                overlap = _catalog_provider_overlap_count(provider_ids)
                provider_visibility = "visible" if overlap > 0 else "not_visible"
            except Exception:
                provider_visibility = "query_failed"

        ranked: list[dict[str, Any]] = []
        for card in cards:
            model_id = str(card.get("id") or "").strip()
            if not model_id: continue

            price = _extract_task_price(card, normalized_task_type)
            if budget_per_image is not None and price is not None and price > budget_per_image:
                continue

            availability_state = "not_checked"
            is_available = None
            if verify_online and provider_visibility in {"visible", "not_visible"}:
                aliases = card.get("aliases", [])
                candidates = [model_id] + [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
                candidates = list(dict.fromkeys(candidates))
                available_by_models = any(c in provider_id_set or _normalize_model_identifier(c) in provider_norm_set for c in candidates)
                if provider_visibility == "not_visible":
                    availability_state = "unknown"
                    is_available = True
                else:
                    availability_state = "available" if available_by_models else "unavailable"
                    is_available = available_by_models

            if availability_state == "unavailable" and not include_unavailable:
                continue

            score = 0.0
            reasons: list[str] = []
            recommended_use_cases = {str(item).strip().lower() for item in card.get("recommended_use_cases", []) if isinstance(item, str) and item.strip()}
            avoid_use_cases = {str(item).strip().lower() for item in card.get("avoid_use_cases", []) if isinstance(item, str) and item.strip()}
            matched_use_cases = sorted(intent_hints.intersection(recommended_use_cases))
            avoided_hits = sorted(intent_hints.intersection(avoid_use_cases))

            if matched_use_cases:
                score += min(42.0, 14.0 * len(matched_use_cases))
                reasons.append(f"use_case_match={matched_use_cases}")
            elif intent_hints: score += 2.0
            else:
                score += 8.0
                reasons.append("generic_intent")

            if avoided_hits:
                score -= 18.0 * len(avoided_hits)
                reasons.append(f"avoid_use_case_penalty={avoided_hits}")

            quality_tier = str(card.get("quality_tier") or "").strip().lower()
            quality_score = _compute_tier_alignment_score(quality_tier, effective_quality_pref, _QUALITY_RANK, exact_bonus=12.0, near_bonus=6.0)
            if quality_score:
                score += quality_score
                reasons.append(f"quality_alignment={quality_tier}")

            speed_tier = str(card.get("speed_tier") or "").strip().lower()
            speed_score = _compute_tier_alignment_score(speed_tier, effective_speed_pref, _SPEED_RANK, exact_bonus=10.0, near_bonus=5.0)
            if speed_score:
                score += speed_score
                reasons.append(f"speed_alignment={speed_tier}")

            if prioritize_cost and price is not None:
                score += max(0.0, 16.0 - (price * 320.0))
                reasons.append("cost_priority")

            if availability_state == "available":
                score += 10.0
                reasons.append("provider_available")
            elif availability_state == "unknown":
                score += 3.0
                reasons.append("provider_availability_unknown")
            elif availability_state == "unavailable":
                score -= 20.0
                reasons.append("provider_unavailable")

            ranked.append({
                "model_id": model_id,
                "score": round(score, 4),
                "price": price,
                "availability_state": availability_state,
                "available": is_available,
                "matched_use_cases": matched_use_cases,
                "reasons": reasons,
                "card": card,
            })

        ranked.sort(key=lambda item: (-item["score"], item["price"] if item["price"] is not None else float("inf"), item["model_id"]))
        limited = ranked[:max_results]
        candidates = []
        for item in limited:
            projected_card = _project_card_fields(item["card"], selected_fields)
            candidates.append({
                "model_id": item["model_id"],
                "score": item["score"],
                "price": item["price"],
                "availability_state": item["availability_state"],
                "available": item["available"],
                "matched_use_cases": item["matched_use_cases"],
                "reasons": item["reasons"],
                "model": projected_card,
            })

        return _success_response(
            data={
                "task_type": normalized_task_type,
                "intent": intent,
                "intent_hints": sorted(intent_hints),
                "preferences": {
                    "quality_tier": effective_quality_pref,
                    "speed_tier": effective_speed_pref,
                    "budget_per_image": budget_per_image,
                    "prioritize_cost": prioritize_cost,
                },
                "online_check": {
                    "enabled": verify_online,
                    "provider_model_count": len(provider_ids),
                    "provider_catalog_visibility": provider_visibility,
                    "fetched_at": provider_fetched_at,
                    "used_cache": provider_used_cache,
                },
                "fields": selected_fields,
                "count": len(candidates),
                "candidates": candidates,
            },
            request_id=request_id,
        )
    except Exception as exc:
        return _error_response(f"Erro ao recomendar modelos: {exc}", request_id=request_id)


@mcp.tool()
async def list_model_cards(
    task_type: str = "text_to_image",
    include_inactive: bool = False,
    limit: int = 12,
    offset: int = 0,
    fields: Optional[str] = None
) -> str:
    """List local model cards without calling the provider."""
    request_id = _new_request_id()
    try:
        normalized_task_type = _normalize_task_type(task_type)
        selected_fields, fields_error = _parse_card_fields(fields)
        if fields_error:
            return _error_response(error=fields_error, code="invalid_fields", details={"fields": fields}, request_id=request_id)

        safe_limit, safe_offset = _normalize_limit_offset(limit, offset)
        cards = catalog_list_model_cards(task_type=normalized_task_type, include_inactive=include_inactive)
        total_count = len(cards)
        page = cards[safe_offset:safe_offset + safe_limit]
        projected_models = [_project_card_fields(card, selected_fields) for card in page]

        return _success_response(
            data={
                "task_type": normalized_task_type,
                "total_count": total_count,
                "count": len(projected_models),
                "limit": safe_limit,
                "offset": safe_offset,
                "models": projected_models,
            },
            request_id=request_id,
        )
    except Exception as exc:
        return _error_response(f"Erro ao listar model cards: {exc}", request_id=request_id)


@mcp.tool()
async def get_model_card(model_id: str, fields: Optional[str] = None) -> str:
    """Return a single model card from the local catalog."""
    request_id = _new_request_id()
    try:
        card = catalog_get_model_card(model_id)
        if not card:
            return _error_response(f"Model '{model_id}' não encontrado no catálogo.", code="model_not_found", request_id=request_id)

        if fields is None:
            projected_card = card
        else:
            selected_fields, fields_error = _parse_card_fields(fields)
            if fields_error:
                return _error_response(error=fields_error, code="invalid_fields", request_id=request_id)
            projected_card = _project_card_fields(card, selected_fields)

        return _success_response(data={"model": projected_card}, request_id=request_id)
    except Exception as exc:
        return _error_response(f"Erro ao buscar model card: {exc}", request_id=request_id)
    except ModelCatalogError as exc:
        return _error_response(
            error=str(exc),
            code="catalog_error",
            details={"model_id": model_id},
            legacy_text=f"Error: {str(exc)}",
            request_id=request_id,
        )
    except Exception as exc:
        return _error_response(
            error=f"Erro inesperado ao buscar model card: {exc}",
            code="unexpected_error",
            details={"model_id": model_id},
            legacy_text=f"Error: {str(exc)}",
            request_id=request_id,
        )


@mcp.tool()
async def list_image_styles(
    force_refresh: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """List image style presets available in provider."""
    request_id = _new_request_id()
    try:
        safe_limit, safe_offset = _normalize_limit_offset(limit, offset, max_limit=200)
        styles, fetched_at, used_cache = await _get_provider_image_styles(force_refresh=force_refresh)
        total_count = len(styles)
        page = styles[safe_offset:safe_offset + safe_limit]
        return _success_response(
            data={
                "styles": [s.model_dump() for s in page],
                "total_count": total_count,
                "metadata": {
                    "fetched_at": fetched_at,
                    "used_cache": used_cache,
                },
            },
            request_id=request_id,
        )
    except Exception as exc:
        return _error_response(f"Erro ao buscar estilos: {exc}", request_id=request_id)


@mcp.tool()
async def list_available_models(force_refresh: bool = False) -> str:
    """List currently available model IDs from the configured provider."""
    request_id = _new_request_id()
    try:
        model_ids, fetched_at, used_cache = await _get_provider_model_ids(force_refresh=force_refresh)
        return _success_response(
            data={
                "models": model_ids,
                "metadata": {
                    "fetched_at": fetched_at,
                    "used_cache": used_cache,
                },
            },
            request_id=request_id,
        )
    except Exception as e:
        return _error_response(f"Erro ao buscar modelos: {e}", request_id=request_id)


@mcp.tool()
async def verify_model_availability(
    model_id: str,
    task_type: str = "text_to_image",
    force_refresh: bool = False,
    include_alternatives: bool = True
) -> str:
    """Verify model viability before execution."""
    request_id = _new_request_id()
    payload = await _build_model_availability_payload(
        model_id=model_id,
        task_type=task_type,
        force_refresh=force_refresh,
        include_alternatives=include_alternatives,
    )
    if not payload.get("ok"):
        return _error_response(payload.get("error", "Falha ao verificar modelo."), request_id=request_id)
    return _success_response(data=payload, request_id=request_id)


@mcp.tool()
def get_nanobot_profile() -> str:
    """
    Return machine-readable server profile for nanobot orchestration.

    Includes:
    - contract version;
    - recommended workflows;
    - tool enablement guidance.

    Use this as the canonical integration contract when bootstrapping an agent.
    """
    request_id = _new_request_id()
    profile = build_nanobot_profile()
    return _success_response(
        data=profile,
        legacy_text=(
            "Nanobot profile loaded. Recommended flow: "
            "list_model_cards -> verify_model_availability -> generate_image/edit_image."
        ),
        request_id=request_id,
        tool_name="get_nanobot_profile",
    )


@mcp.tool()
def get_security_metrics() -> str:
    """
    Return in-memory security counters/events for audit and incident triage.

    Typical use:
    - check blocked-path, prompt-injection, auth, egress and cache events;
    - correlate recent failures with security controls.

    Notes:
    - data is in-memory and process-local;
    - values reset on server restart or `clear_security_metrics`.
    """
    request_id = _new_request_id()
    return _success_response(
        data={
            "operation": "get_security_metrics",
            "security_metrics": get_security_metrics_snapshot(),
        },
        legacy_text="Security metrics snapshot generated.",
        request_id=request_id,
        tool_name="get_security_metrics",
    )


@mcp.tool()
def clear_security_metrics() -> str:
    """
    Clear in-memory security counters/events and return reset summary.

    Use when starting a new diagnostic window or incident timeline.
    """
    request_id = _new_request_id()
    cleared = clear_security_metrics_snapshot()
    return _success_response(
        data={
            "operation": "clear_security_metrics",
            **cleared,
        },
        legacy_text=(
            "Security metrics cleared. "
            f"events={cleared['cleared_recent_events_total']} counters={cleared['cleared_counters_total']}"
        ),
        request_id=request_id,
        tool_name="clear_security_metrics",
    )


@mcp.tool()
def get_security_posture() -> str:
    """
    Return effective runtime security posture for auditability.

    Includes:
    - working-dir sandbox policy;
    - HTTP/auth policy;
    - egress/download policy;
    - input limit policy;
    - warning list for permissive configurations.
    """
    request_id = _new_request_id()

    allowed_roots = [str(root) for root in get_allowed_working_roots()]
    root_sandbox_disabled = _env_bool("PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX", False)
    allow_remote_http = _env_bool("PERCIVAL_IMAGE_MCP_ALLOW_REMOTE_HTTP", False)
    allow_private_provider = _env_bool("PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL", False)
    allow_insecure_provider = _env_bool("PERCIVAL_IMAGE_MCP_ALLOW_INSECURE_PROVIDER_URL", False)
    allow_private_downloads = _env_bool("PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_DOWNLOADS", False)
    allow_http_downloads = _env_bool("PERCIVAL_IMAGE_MCP_ALLOW_HTTP_DOWNLOADS", False)
    default_output_dir = _get_default_output_directory()

    warnings: list[str] = []
    if root_sandbox_disabled:
        warnings.append("Root sandbox disabled: working_dir containment is bypassed.")
    if allow_remote_http:
        warnings.append("Remote HTTP bind allowed: ensure token + trusted network boundary.")
    if allow_private_provider:
        warnings.append("Private provider URLs allowed.")
    if allow_insecure_provider:
        warnings.append("Insecure provider URLs (http) allowed.")
    if allow_private_downloads:
        warnings.append("Private download URLs allowed.")
    if allow_http_downloads:
        warnings.append("HTTP download URLs allowed.")
    if not default_output_dir.exists():
        warnings.append(f"Default output directory does not exist: {default_output_dir}")

    posture = {
        "server": SERVER_NAME,
        "contract_version": CONTRACT_VERSION,
        "working_dir_policy": {
            "root_sandbox_disabled": root_sandbox_disabled,
            "allowed_roots": allowed_roots,
            "configured_allowed_roots_env": os.getenv("PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS", ""),
            "default_output_dir": str(default_output_dir),
        },
        "http_policy": {
            "allow_remote_http": allow_remote_http,
            "auth_token_env": os.getenv("PERCIVAL_IMAGE_MCP_AUTH_TOKEN_ENV", "PERCIVAL_IMAGE_MCP_AUTH_TOKEN"),
        },
        "egress_policy": {
            "provider_base_url": get_jarvina_base_url(),
            "allow_private_provider_url": allow_private_provider,
            "allow_insecure_provider_url": allow_insecure_provider,
            "allowed_provider_hosts": os.getenv("PERCIVAL_IMAGE_MCP_ALLOWED_PROVIDER_HOSTS", ""),
            "allow_private_downloads": allow_private_downloads,
            "allow_http_downloads": allow_http_downloads,
            "allowed_download_hosts": os.getenv("PERCIVAL_IMAGE_MCP_ALLOWED_DOWNLOAD_HOSTS", ""),
            "download_max_bytes": _env_int("PERCIVAL_IMAGE_MCP_DOWNLOAD_MAX_BYTES", 25 * 1024 * 1024),
        },
        "input_limits": {
            "max_prompt_chars": _env_int("PERCIVAL_IMAGE_MCP_MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS),
            "max_negative_prompt_chars": _env_int(
                "PERCIVAL_IMAGE_MCP_MAX_NEGATIVE_PROMPT_CHARS",
                DEFAULT_MAX_NEGATIVE_PROMPT_CHARS,
            ),
            "max_filename_prefix_chars": _env_int(
                "PERCIVAL_IMAGE_MCP_MAX_FILENAME_PREFIX_CHARS",
                DEFAULT_MAX_FILENAME_PREFIX_CHARS,
            ),
            "max_model_id_chars": _env_int("PERCIVAL_IMAGE_MCP_MAX_MODEL_ID_CHARS", DEFAULT_MAX_MODEL_ID_CHARS),
            "max_list_files": _env_int("PERCIVAL_IMAGE_MCP_MAX_LIST_FILES", DEFAULT_MAX_LIST_FILES),
            "max_analysis_prompt_chars": _env_int(
                "PERCIVAL_IMAGE_MCP_MAX_ANALYSIS_PROMPT_CHARS",
                4000,
            ),
            "max_comparison_focus_chars": _env_int(
                "PERCIVAL_IMAGE_MCP_MAX_COMPARISON_FOCUS_CHARS",
                200,
            ),
        },
        "warnings": warnings,
    }
    record_security_event(
        "security_posture_checked",
        {"warning_count": len(warnings)},
    )
    return _success_response(
        data={"operation": "get_security_posture", "posture": posture},
        legacy_text=f"Security posture generated with {len(warnings)} warning(s).",
        request_id=request_id,
        tool_name="get_security_posture",
    )


@mcp.tool()
async def generate_image(
    working_dir: str,
    prompt: str,
    model: str = "venice-sd35",
    size: str = "1024x1024",
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    steps: Optional[int] = None,
    style_preset: Optional[str] = None,
    safe_mode: Optional[bool] = None,
    variants: Optional[int] = None,
    seed: Optional[int] = None,
    format: Optional[str] = None,
    lora_strength: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    hide_watermark: Optional[bool] = None,
    embed_exif_metadata: Optional[bool] = None,
    enable_web_search: Optional[bool] = None,
    parameter_overrides_json: Optional[str] = None,
    output_dir: Optional[str] = None,
    filename_prefix: str = "jarvina_gen",
    strict_model_check: bool = True,
    force_model_refresh: bool = False,
    strict_style_check: bool = True,
    force_style_refresh: bool = False,
) -> str:
    """
    Generate image(s) from a text prompt with catalog-aware parameter planning.
    """
    request_id = _new_request_id()
    try:
        working_path, working_error = validate_working_directory(working_dir)
        if working_error:
            return _error_response(error=working_error, code="invalid_working_dir", request_id=request_id)

        prompt_max_chars = get_env_int("PERCIVAL_IMAGE_MCP_MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS)
        sanitized_prompt, prompt_error = sanitize_input_text(prompt, field_name="prompt", max_chars=prompt_max_chars)
        if prompt_error:
            return _error_response(error=prompt_error, code="invalid_prompt", request_id=request_id)

        if negative_prompt is not None:
            negative_max_chars = get_env_int("PERCIVAL_IMAGE_MCP_MAX_NEGATIVE_PROMPT_CHARS", DEFAULT_MAX_NEGATIVE_PROMPT_CHARS)
            sanitized_negative_prompt, negative_error = sanitize_input_text(negative_prompt, field_name="negative_prompt", max_chars=negative_max_chars, allow_empty=True)
            if negative_error:
                return _error_response(error=negative_error, code="invalid_negative_prompt", request_id=request_id)
            negative_prompt = sanitized_negative_prompt

        effective_output_dir = output_dir or str(_get_default_output_directory())
        resolved_output_path, output_error = _resolve_output_path(path_value=effective_output_dir, working_path=working_path, label="output_dir")
        if output_error:
            return _error_response(error=output_error, code="invalid_output_dir", request_id=request_id)

        precheck_error, precheck_payload = await _enforce_model_precheck(
            model_id=model,
            task_type="text_to_image",
            strict_model_check=strict_model_check,
            force_model_refresh=force_model_refresh,
        )
        if precheck_error:
            return _error_response(error=precheck_error["error"], code=precheck_error["code"], details=precheck_error.get("details"), request_id=request_id)

        output_path = resolved_output_path
        output_path.mkdir(parents=True, exist_ok=True)

        model_card = catalog_get_model_card(model)
        recommended_params = dict(model_card.get("recommended_api_params", {})) if model_card else {}

        try:
            runtime_overrides = parse_parameter_overrides_json(parameter_overrides_json)
        except ValueError as exc:
            return _error_response(error=str(exc), code="invalid_parameter_overrides", request_id=request_id)

        explicit_params = {
            "aspect_ratio": aspect_ratio, "resolution": resolution, "cfg_scale": cfg_scale,
            "negative_prompt": negative_prompt, "steps": steps, "style_preset": style_preset,
            "safe_mode": safe_mode, "variants": variants, "seed": seed, "format": format,
            "lora_strength": lora_strength, "width": width, "height": height,
            "hide_watermark": hide_watermark, "embed_exif_metadata": embed_exif_metadata,
            "enable_web_search": enable_web_search,
        }

        payload_plan = build_venice_generation_request(
            model=model, prompt=sanitized_prompt, size=size,
            explicit_params=explicit_params, card_recommended_params=recommended_params,
            runtime_overrides=runtime_overrides,
        )

        style_check_payload = None
        resolved_style_preset = payload_plan.get("resolved_provider_params", {}).get("style_preset")
        if isinstance(resolved_style_preset, str) and resolved_style_preset.strip():
            style_check_payload = await _build_style_validation_payload(resolved_style_preset, force_refresh=force_style_refresh)
            if not style_check_payload.get("ok") and strict_style_check:
                return _error_response(error=style_check_payload.get("error", "style precheck failed"), code="style_precheck_failed", details=style_check_payload, request_id=request_id)
            if style_check_payload.get("availability_state") == "unavailable" and strict_style_check:
                return _error_response(error=f"style_preset '{resolved_style_preset}' is not available.", code="invalid_style_preset", details=style_check_payload, request_id=request_id)

        response, transport_meta = await generate_images_with_transport(payload_plan["openai_request"])

        saved_files, save_error = await _save_images_from_response(response=response, output_path=output_path, filename_prefix=filename_prefix)
        if save_error:
            return _error_response(error=save_error, code="image_save_failed", details={"model": model}, request_id=request_id)

        return _success_response(
            data={
                "operation": "generate_image",
                "model": model,
                "prompt": sanitized_prompt,
                "params": payload_plan.get("resolved_provider_params", {}),
                "transport": transport_meta,
                "files": saved_files,
            },
            request_id=request_id,
        )
    except Exception as e:
        return _error_response(f"Erro ao gerar imagem: {e}", request_id=request_id)

@mcp.tool()
async def edit_image(
    working_dir: str,
    image_path: str,
    prompt: str,
    mask_path: Optional[str] = None,
    model: str = "qwen-image-2-edit",
    size: Optional[str] = None,
    quality: Optional[str] = None,
    n: int = 1,
    output_dir: Optional[str] = None,
    filename_prefix: str = "edited",
    strict_model_check: bool = True,
    force_model_refresh: bool = False
) -> str:
    """Edit an existing image with model-card and safety validation."""
    request_id = _new_request_id()
    try:
        working_path, working_error = validate_working_directory(working_dir)
        if working_error:
            return _error_response(error=working_error, code="invalid_working_dir", request_id=request_id)

        prompt_max_chars = get_env_int("PERCIVAL_IMAGE_MCP_MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS)
        sanitized_prompt, prompt_error = sanitize_input_text(prompt, field_name="prompt", max_chars=prompt_max_chars)
        if prompt_error:
            return _error_response(error=prompt_error, code="invalid_prompt", request_id=request_id)

        precheck_error, precheck_payload = await _enforce_model_precheck(
            model_id=model,
            task_type="image_edit",
            strict_model_check=strict_model_check,
            force_model_refresh=force_model_refresh,
        )
        if precheck_error:
            return _error_response(error=precheck_error["error"], code=precheck_error["code"], details=precheck_error.get("details"), request_id=request_id)

        is_valid, error_message, resolved_image_path = validate_image_path(image_path, "read", working_dir)
        if not is_valid:
            return _error_response(error=error_message or "Invalid image_path.", code="invalid_image_path", request_id=request_id)

        # Basic confinement check (utils.path_utils handles most of this)
        if not is_relative_to(resolved_image_path, working_path):
             return _error_response(error="Path escape blocked", code="invalid_image_path_scope", request_id=request_id)

        resolved_mask_path: Optional[Path] = None
        if mask_path:
            mask_ok, mask_error, resolved_mask_path = validate_image_path(mask_path, "read", working_dir)
            if not mask_ok:
                return _error_response(error=mask_error or "Invalid mask_path.", code="invalid_mask_path", request_id=request_id)
            if not is_relative_to(resolved_mask_path, working_path):
                 return _error_response(error="Path escape blocked", code="invalid_mask_path_scope", request_id=request_id)

        effective_output_dir = output_dir or str(_get_default_output_directory())
        resolved_output_path, output_error = _resolve_output_path(path_value=effective_output_dir, working_path=working_path, label="output_dir")
        if output_error:
            return _error_response(error=output_error, code="invalid_output_dir", request_id=request_id)

        output_path = resolved_output_path
        output_path.mkdir(parents=True, exist_ok=True)

        edit_kwargs: dict[str, Any] = {
            "model": model,
            "prompt": sanitized_prompt,
            "n": n,
            "response_format": "b64_json",
        }
        if size: edit_kwargs["size"] = size
        if quality: edit_kwargs["quality"] = quality

        # Using bytes content for AsyncOpenAI
        with open(resolved_image_path, "rb") as image_file:
            edit_kwargs["image"] = image_file.read()
        
        if resolved_mask_path:
            with open(resolved_mask_path, "rb") as mask_file:
                edit_kwargs["mask"] = mask_file.read()

        response = await client.images.edit(**edit_kwargs)
        saved_files, save_error = await _save_images_from_response(response=response, output_path=output_path, filename_prefix=filename_prefix)
        if save_error:
            return _error_response(error=save_error, code="image_save_failed", details={"model": model}, request_id=request_id)

        return _success_response(
            data={
                "operation": "edit_image",
                "model": model,
                "prompt": sanitized_prompt,
                "files": saved_files,
            },
            request_id=request_id,
        )
    except Exception as e:
        logger.error(f"Error in edit_image: {e}")
        return _error_response(f"Erro ao editar imagem: {e}", request_id=request_id)
    except Exception as exc:
        return _error_response(
            error=f"Erro ao editar imagem com o provedor: {str(exc)}",
            code="edit_failed",
            details={"model": model},
            legacy_text=f"Erro ao editar imagem com o provedor: {str(exc)}",
            request_id=request_id,
        )

def create_image_variations(
    working_dir: str,
    image_path: str,
    n: int = 2,
    size: Optional[str] = "1024x1024",
    output_dir: str = "./image_variations",
    filename_prefix: str = "variation"
) -> str:
    """
    [AVISO: Ferramenta temporariamente desativada]
    Create variations of an existing image using DALL-E 2.
    """
    # Intencionalmente não registrado como tool MCP para evitar uso pelo agente.
    return "Aviso: A criação de variações de imagem permanece desativada neste servidor."


@mcp.tool()

async def list_generated_images(working_dir: str, directory: Optional[str] = None) -> str:
    """List images in a specific directory (within working_dir scope)."""
    request_id = _new_request_id()
    try:
        working_path, working_error = validate_working_directory(working_dir)
        if working_error:
            return _error_response(error=working_error, code="invalid_working_dir", request_id=request_id)

        effective_directory = directory or str(_get_default_output_directory())
        dir_path, dir_error = _resolve_output_path(path_value=effective_directory, working_path=working_path, label="directory")
        if dir_error:
            return _error_response(error=dir_error, code="invalid_directory_scope", request_id=request_id)

        if not dir_path.exists() or not dir_path.is_dir():
             return _error_response(error="Directory not found", code="directory_not_found", request_id=request_id)

        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        image_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        max_list_items = get_env_int("PERCIVAL_IMAGE_MCP_MAX_LIST_FILES", DEFAULT_MAX_LIST_FILES)
        if len(image_files) > max_list_items:
            image_files = image_files[:max_list_items]

        images_data = []
        for i, file_path in enumerate(image_files, 1):
            file_stats = file_path.stat()
            image_info = get_image_info(file_path)
            images_data.append({
                "index": i,
                "name": file_path.name,
                "path": str(file_path),
                "size_bytes": file_stats.st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_stats.st_mtime)),
                "dimensions": f"{image_info['size'][0]}x{image_info['size'][1]}" if "error" not in image_info else "unknown",
            })

        return _success_response(data={"images": images_data}, request_id=request_id)
    except Exception as e:
        logger.error(f"Error in list_generated_images: {e}")
        return _error_response(f"Error listing images: {e}", request_id=request_id)


@mcp.tool()
async def get_image_metadata(image_path: str, working_dir: str) -> str:
    """Return dimensions and metadata for a specific image."""
    request_id = _new_request_id()
    try:
        is_valid, error_message, resolved_path = validate_image_path(image_path, "read", working_dir)
        if not is_valid:
            return _error_response(error=error_message or "Invalid image_path.", code="invalid_image_path", request_id=request_id)

        image_info = get_image_info(resolved_path)
        if "error" in image_info:
             return _error_response(error=image_info["error"], code="metadata_extraction_failed", request_id=request_id)

        return _success_response(data={"metadata": image_info}, request_id=request_id)
    except Exception as e:
        logger.error(f"Error in get_image_metadata: {e}")
        return _error_response(f"Error getting metadata: {e}", request_id=request_id)


@mcp.tool()
async def get_security_metrics() -> str:
    """Return in-memory security counters for audit."""
    from utils.security_utils import get_security_metrics_snapshot
    request_id = _new_request_id()
    return _success_response(data={"metrics": get_security_metrics_snapshot()}, request_id=request_id)


@mcp.tool()
async def clear_security_metrics() -> str:
    """Clear in-memory security counters."""
    from utils.security_utils import clear_security_metrics as clear_metrics
    request_id = _new_request_id()
    return _success_response(data=clear_metrics(), request_id=request_id)


@mcp.tool()
async def get_security_posture() -> str:
    """Return effective runtime security configuration."""
    from utils.security_utils import get_security_posture as get_posture
    request_id = _new_request_id()
    return _success_response(data={"posture": get_posture()}, request_id=request_id)


@mcp.tool()
async def get_nanobot_profile() -> str:
    """Return machine-readable server profile for nanobot orchestration."""
    request_id = _new_request_id()
    return _success_response(
        data={
            "contract_version": CONTRACT_VERSION,
            "server_name": SERVER_NAME,
            "workflows": ["recommendation -> verification -> execution"],
        },
        request_id=request_id
    )
