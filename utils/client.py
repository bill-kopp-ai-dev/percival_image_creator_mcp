import base64
import io
import logging
import os
import re
import socket
from ipaddress import ip_address
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import Optional
from urllib.parse import urlparse

import requests
from openai import OpenAI
from PIL import Image

from utils.security_utils import record_security_event

logger = logging.getLogger(__name__)

DEFAULT_JARVINA_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JARVINA_VISION_MODEL = "qwen-2.5-vl"
DEFAULT_DOWNLOAD_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 90
SUPPORTED_IMAGE_TRANSPORTS = {"auto", "openai_compat", "venice_native"}


def _env_bool(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(var_name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except Exception:
        return default
    return max(minimum, value)


def _parse_size_to_dimensions(size: str) -> tuple[Optional[int], Optional[int]]:
    normalized = (size or "").strip().lower()
    if "x" not in normalized:
        return None, None
    width_text, height_text = normalized.split("x", 1)
    try:
        width = int(width_text.strip())
        height = int(height_text.strip())
    except Exception:
        return None, None
    if width <= 0 or height <= 0:
        return None, None
    return width, height


def _parse_host_allowlist(var_name: str) -> list[str]:
    raw = os.getenv(var_name, "").strip()
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _host_matches_allowlist(host: str, allowed_hosts: list[str]) -> bool:
    normalized = host.strip().lower()
    for allowed in allowed_hosts:
        if normalized == allowed or normalized.endswith(f".{allowed}"):
            return True
    return False


def _is_non_public_ip(value: str) -> bool:
    try:
        ip_obj = ip_address(value)
    except ValueError:
        return False
    return bool(
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def _resolve_host_ips(hostname: str) -> set[str]:
    ips: set[str] = set()
    infos = socket.getaddrinfo(hostname, None)
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_value = sockaddr[0]
        if isinstance(ip_value, str):
            ips.add(ip_value)
    return ips


def validate_outbound_url(url: str, *, purpose: str) -> str:
    """
    Validate outbound HTTP destination to reduce SSRF risk.

    purpose:
      - "provider": validates JARVINA_BASE_URL
      - "download": validates provider-returned image URLs
    """
    normalized_url = (url or "").strip()
    parsed = urlparse(normalized_url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").strip().lower()

    if not scheme or not host:
        record_security_event("upstream_url_blocked", {"purpose": purpose, "reason": "invalid_url"})
        raise ValueError("Invalid URL: missing scheme or host.")

    if purpose == "provider":
        allow_http = _env_bool("PERCIVAL_IMAGE_MCP_ALLOW_INSECURE_PROVIDER_URL", False)
        allow_private = _env_bool("PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL", False)
        allowlist = _parse_host_allowlist("PERCIVAL_IMAGE_MCP_ALLOWED_PROVIDER_HOSTS")
    elif purpose == "download":
        allow_http = _env_bool("PERCIVAL_IMAGE_MCP_ALLOW_HTTP_DOWNLOADS", False)
        allow_private = _env_bool("PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_DOWNLOADS", False)
        allowlist = _parse_host_allowlist("PERCIVAL_IMAGE_MCP_ALLOWED_DOWNLOAD_HOSTS")
    else:
        raise ValueError(f"Unknown outbound URL purpose: {purpose}")

    allowed_schemes = {"https"}
    if allow_http:
        allowed_schemes.add("http")

    if scheme not in allowed_schemes:
        record_security_event(
            "upstream_url_blocked",
            {"purpose": purpose, "reason": "scheme_not_allowed", "scheme": scheme, "host": host},
        )
        raise ValueError(f"Blocked URL scheme '{scheme}' for {purpose}.")

    if allowlist and not _host_matches_allowlist(host, allowlist):
        record_security_event(
            "upstream_url_blocked",
            {"purpose": purpose, "reason": "host_not_allowlisted", "host": host},
        )
        raise ValueError(f"Blocked host '{host}' not present in allowlist for {purpose}.")

    if not allow_private:
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            record_security_event(
                "upstream_url_blocked",
                {"purpose": purpose, "reason": "local_hostname", "host": host},
            )
            raise ValueError(f"Blocked local hostname for {purpose}: {host}")

        if _is_non_public_ip(host):
            record_security_event(
                "upstream_url_blocked",
                {"purpose": purpose, "reason": "private_ip_literal", "host": host},
            )
            raise ValueError(f"Blocked non-public IP host for {purpose}: {host}")

        try:
            resolved_ips = _resolve_host_ips(host)
        except Exception as exc:
            record_security_event(
                "upstream_url_blocked",
                {"purpose": purpose, "reason": "dns_resolution_failed", "host": host, "error": str(exc)},
            )
            raise ValueError(f"Failed to resolve host '{host}' for {purpose}: {exc}")

        blocked_ips = sorted(ip for ip in resolved_ips if _is_non_public_ip(ip))
        if blocked_ips:
            record_security_event(
                "upstream_url_blocked",
                {
                    "purpose": purpose,
                    "reason": "resolved_private_ip",
                    "host": host,
                    "blocked_ip_count": len(blocked_ips),
                },
            )
            raise ValueError(
                f"Blocked host '{host}' for {purpose}: resolves to non-public IP addresses."
            )

    record_security_event("upstream_url_allowed", {"purpose": purpose, "scheme": scheme, "host": host})
    return normalized_url


def get_jarvina_base_url() -> str:
    return os.getenv("JARVINA_BASE_URL", DEFAULT_JARVINA_BASE_URL)


def get_jarvina_api_key() -> Optional[str]:
    return (
        os.getenv("JARVINA_API_KEY")
        or os.getenv("VENICE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def get_jarvina_vision_model() -> str:
    return os.getenv("JARVINA_VISION_MODEL", DEFAULT_JARVINA_VISION_MODEL)


def get_image_transport_mode() -> str:
    raw_mode = (os.getenv("PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT") or "auto").strip().lower()
    if raw_mode not in SUPPORTED_IMAGE_TRANSPORTS:
        return "auto"
    return raw_mode


def _is_venice_base_url(base_url: str) -> bool:
    try:
        host = (urlparse(base_url).hostname or "").strip().lower()
    except Exception:
        return False
    return host == "venice.ai" or host.endswith(".venice.ai")


def _sanitize_for_openai_compat(
    openai_request: dict[str, Any],
    *,
    drop_extra_body: bool,
) -> tuple[dict[str, Any], list[str]]:
    sanitized = dict(openai_request)
    dropped_keys: list[str] = []
    if drop_extra_body and isinstance(sanitized.get("extra_body"), dict):
        dropped_keys = sorted(str(key) for key in sanitized["extra_body"].keys())
        sanitized.pop("extra_body", None)
    return sanitized, dropped_keys


# Compatibility exports for modules that import constants directly.
JARVINA_BASE_URL = get_jarvina_base_url()
JARVINA_API_KEY = get_jarvina_api_key()
JARVINA_VISION_MODEL = get_jarvina_vision_model()


_client_instance: Optional[OpenAI] = None


def get_jarvina_client() -> OpenAI:
    """
    Lazily initialize the provider client.

    This avoids import-time crashes when environment variables are not set yet.
    """
    global _client_instance

    if _client_instance is not None:
        return _client_instance

    api_key = get_jarvina_api_key()
    if not api_key:
        msg = "ERRO: Nenhuma chave de API configurada. Defina JARVINA_API_KEY (ou VENICE_API_KEY/OPENAI_API_KEY)."
        raise RuntimeError(msg)

    base_url = validate_outbound_url(get_jarvina_base_url(), purpose="provider")
    _client_instance = OpenAI(api_key=api_key, base_url=base_url)
    logger.info("Jarvina Client inicializado apontando para: %s", base_url)
    return _client_instance


class _LazyJarvinaClient:
    """Proxy that defers OpenAI client creation until first attribute access."""

    def __getattr__(self, name: str):
        return getattr(get_jarvina_client(), name)


# Backward-compatible name used by existing tools.
jarvina_client = _LazyJarvinaClient()


def _build_provider_endpoint(base_url: str, endpoint_path: str) -> str:
    base = base_url.rstrip("/")
    path = endpoint_path.lstrip("/")
    return f"{base}/{path}"


def _normalize_style_records(payload: Any) -> list[dict[str, Any]]:
    raw_items: list[Any] = []
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        for key in ("data", "styles", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_items = value
                break

    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            name = item.strip()
            if not name:
                continue
            normalized.append({"id": name, "name": name, "description": None})
            continue

        if not isinstance(item, dict):
            continue

        style_id_raw = (
            item.get("id")
            or item.get("slug")
            or item.get("style")
            or item.get("preset")
            or item.get("name")
        )
        if not isinstance(style_id_raw, str) or not style_id_raw.strip():
            continue
        style_id = style_id_raw.strip()
        name_raw = item.get("name")
        name = name_raw.strip() if isinstance(name_raw, str) and name_raw.strip() else style_id
        description_raw = item.get("description")
        description = description_raw.strip() if isinstance(description_raw, str) and description_raw.strip() else None
        normalized.append({"id": style_id, "name": name, "description": description})

    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for style in normalized:
        key = str(style.get("id") or style.get("name") or "").strip().lower()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(style)

    return deduped


def list_provider_image_styles() -> list[dict[str, Any]]:
    """
    Fetch available image style presets from provider (`GET /image/styles`).
    """
    api_key = get_jarvina_api_key()
    if not api_key:
        raise RuntimeError("Missing API key for provider style discovery.")

    base_url = validate_outbound_url(get_jarvina_base_url(), purpose="provider")
    endpoint_url = _build_provider_endpoint(base_url, "image/styles")
    validate_outbound_url(endpoint_url, purpose="provider")

    timeout = _env_int("PERCIVAL_IMAGE_MCP_PROVIDER_TIMEOUT_SECONDS", DEFAULT_PROVIDER_TIMEOUT_SECONDS)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.get(endpoint_url, headers=headers, timeout=timeout)
    try:
        response.raise_for_status()
    except Exception as exc:
        response_text = (getattr(response, "text", "") or "").strip()
        details = f"{exc}"
        if response_text:
            details = f"{details}; body={response_text[:400]}"
        raise RuntimeError(f"Provider styles request failed: {details}") from exc

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError("Provider styles endpoint returned non-JSON response.") from exc

    styles = _normalize_style_records(payload)
    if not styles:
        raise RuntimeError("Provider styles response did not contain style records.")
    return styles


def _normalize_native_generation_response(payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise ValueError("Native provider response must be a JSON object.")

    raw_items: list[Any] = []
    data_items = payload.get("data")
    if isinstance(data_items, list):
        raw_items = data_items
    elif isinstance(payload.get("images"), list):
        raw_items = payload["images"]
    elif payload.get("image") is not None:
        raw_items = [payload["image"]]

    normalized_items: list[Any] = []
    for item in raw_items:
        if isinstance(item, dict):
            b64_value = item.get("b64_json") or item.get("base64") or item.get("image_base64") or item.get("image")
            url_value = item.get("url")
            normalized_items.append(
                SimpleNamespace(
                    b64_json=b64_value if isinstance(b64_value, str) else None,
                    url=url_value if isinstance(url_value, str) else None,
                )
            )
            continue

        if isinstance(item, str):
            stripped = item.strip()
            if stripped.startswith("http://") or stripped.startswith("https://"):
                normalized_items.append(SimpleNamespace(b64_json=None, url=stripped))
            else:
                normalized_items.append(SimpleNamespace(b64_json=stripped, url=None))
            continue

    if not normalized_items:
        raise ValueError("Native provider response does not include image payloads.")

    return SimpleNamespace(data=normalized_items, raw=payload)


class _NativeUnrecognizedKeysError(RuntimeError):
    def __init__(self, keys: list[str], message: str):
        super().__init__(message)
        self.keys = keys


def _extract_unrecognized_keys_from_error_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []

    discovered: set[str] = set()
    issues = payload.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            code = issue.get("code")
            keys = issue.get("keys")
            if code == "unrecognized_keys" and isinstance(keys, list):
                for key in keys:
                    if isinstance(key, str) and key.strip():
                        discovered.add(key.strip())

    details = payload.get("details")
    if isinstance(details, dict):
        errors = details.get("_errors")
        if isinstance(errors, list):
            for message in errors:
                if not isinstance(message, str):
                    continue
                discovered.update(match.strip() for match in re.findall(r"'([^']+)'", message) if match.strip())

    return sorted(discovered)


def _get_native_retry_limit() -> int:
    raw = os.getenv("PERCIVAL_IMAGE_MCP_VENICE_NATIVE_RETRIES")
    if raw is None:
        return 1
    try:
        return max(0, int(raw.strip()))
    except Exception:
        return 1


def _generate_images_venice_native(openai_request: dict[str, Any]) -> tuple[Any, list[str]]:
    api_key = get_jarvina_api_key()
    if not api_key:
        raise RuntimeError("Missing API key for native Venice transport.")

    base_url = validate_outbound_url(get_jarvina_base_url(), purpose="provider")
    endpoint_url = _build_provider_endpoint(base_url, "image/generate")
    validate_outbound_url(endpoint_url, purpose="provider")

    timeout = _env_int("PERCIVAL_IMAGE_MCP_PROVIDER_TIMEOUT_SECONDS", DEFAULT_PROVIDER_TIMEOUT_SECONDS)
    payload: dict[str, Any] = {
        "model": openai_request.get("model"),
        "prompt": openai_request.get("prompt"),
    }

    size = openai_request.get("size")
    if isinstance(size, str) and size.strip():
        width, height = _parse_size_to_dimensions(size)
        if width and height:
            payload.setdefault("width", width)
            payload.setdefault("height", height)

    extra_body = openai_request.get("extra_body")
    if isinstance(extra_body, dict):
        payload.update(extra_body)

    # Venice supports this as explicit API contract in image/generate payload.
    payload.setdefault("return_binary", False)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    retry_limit = _get_native_retry_limit()
    dropped_keys: list[str] = []
    immutable_keys = {"model", "prompt"}

    for attempt in range(retry_limit + 1):
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=timeout)
        try:
            response.raise_for_status()
        except Exception as exc:
            response_text = (getattr(response, "text", "") or "").strip()
            details = f"{exc}"
            response_json = None
            try:
                response_json = response.json()
            except Exception:
                response_json = None

            unrecognized_keys = _extract_unrecognized_keys_from_error_payload(response_json)
            retryable_keys = [key for key in unrecognized_keys if key in payload and key not in immutable_keys]
            can_retry = bool(retryable_keys) and attempt < retry_limit
            if can_retry:
                for key in retryable_keys:
                    payload.pop(key, None)
                    if key not in dropped_keys:
                        dropped_keys.append(key)
                logger.warning(
                    "Native Venice transport rejected keys %s (attempt %s/%s). Retrying without them.",
                    retryable_keys,
                    attempt + 1,
                    retry_limit + 1,
                )
                continue

            if response_text:
                details = f"{details}; body={response_text[:400]}"
            if unrecognized_keys:
                raise _NativeUnrecognizedKeysError(
                    unrecognized_keys,
                    f"Native Venice generation request failed: {details}",
                ) from exc
            raise RuntimeError(f"Native Venice generation request failed: {details}") from exc

        try:
            payload_json = response.json()
        except Exception as exc:
            raise RuntimeError("Native Venice generation returned non-JSON response.") from exc

        return _normalize_native_generation_response(payload_json), dropped_keys

    raise RuntimeError("Native Venice generation request exhausted retries.")


def generate_images_with_transport(
    openai_request: dict[str, Any],
    *,
    openai_client: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Execute image generation with configured transport strategy.

    Modes:
    - openai_compat (default): client.images.generate(...)
    - venice_native: POST /image/generate, with optional fallback to openai_compat
    """
    requested_mode = get_image_transport_mode()
    base_url = get_jarvina_base_url()
    mode = requested_mode
    if mode == "auto":
        mode = "venice_native" if _is_venice_base_url(base_url) else "openai_compat"
    fallback_enabled = _env_bool("PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT_FALLBACK", True)
    effective_client = openai_client or jarvina_client

    if mode == "openai_compat":
        should_drop_extra_body = _is_venice_base_url(base_url)
        request_payload, dropped_keys = _sanitize_for_openai_compat(
            openai_request,
            drop_extra_body=should_drop_extra_body,
        )
        response = effective_client.images.generate(**request_payload)
        return response, {
            "transport_requested": requested_mode,
            "transport_used": "openai_compat",
            "fallback_used": False,
            "compat_dropped_keys": dropped_keys,
        }

    if mode == "venice_native":
        try:
            response, dropped_keys = _generate_images_venice_native(openai_request)
            return response, {
                "transport_requested": requested_mode,
                "transport_used": "venice_native",
                "fallback_used": False,
                "native_dropped_keys": dropped_keys,
            }
        except Exception as exc:
            if not fallback_enabled:
                raise
            logger.warning(
                "Native Venice transport failed, falling back to OpenAI-compatible transport: %s",
                exc,
            )
            record_security_event(
                "provider_transport_fallback",
                {"requested_transport": "venice_native", "reason": str(exc)},
            )
            request_payload, dropped_keys = _sanitize_for_openai_compat(
                openai_request,
                drop_extra_body=_is_venice_base_url(base_url),
            )
            response = effective_client.images.generate(**request_payload)
            return response, {
                "transport_requested": requested_mode,
                "transport_used": "openai_compat",
                "fallback_used": True,
                "fallback_reason": str(exc),
                "compat_dropped_keys": dropped_keys,
            }

    # Defensive fallback for unexpected values.
    response = effective_client.images.generate(**openai_request)
    return response, {
        "transport_requested": requested_mode,
        "transport_used": "openai_compat",
        "fallback_used": True,
        "fallback_reason": "unknown_transport_mode",
    }


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_valid_image_format(file_path: Path) -> bool:
    """Check if file is a valid image format supported by vision APIs."""
    valid_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    return file_path.suffix.lower() in valid_extensions


def get_image_info(image_path: Path) -> dict:
    """Get basic image information."""
    try:
        with Image.open(image_path) as img:
            return {
                "format": img.format,
                "size": img.size,
                "mode": img.mode,
                "file_size": image_path.stat().st_size,
            }
    except Exception as exc:
        return {"error": str(exc)}


def save_base64_image(base64_data: str, output_path: Path, image_format: str = "PNG") -> bool:
    """Save base64 encoded image data to file."""
    try:
        image_data = base64.b64decode(base64_data)
        with Image.open(io.BytesIO(image_data)) as img:
            img.save(output_path, format=image_format)
        return True
    except Exception as exc:
        print(f"Error saving image: {exc}")
        return False


def download_image_from_url(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download image from URL and save to file."""
    try:
        max_bytes = _env_int("PERCIVAL_IMAGE_MCP_DOWNLOAD_MAX_BYTES", DEFAULT_DOWNLOAD_MAX_BYTES)
        validated_url = validate_outbound_url(url, purpose="download")

        response = requests.get(validated_url, timeout=timeout, stream=True)
        response.raise_for_status()

        # Validate final redirect destination if provider redirected.
        final_url = response.url or validated_url
        validate_outbound_url(final_url, purpose="download")

        content_length_raw = response.headers.get("Content-Length")
        if content_length_raw:
            try:
                content_length = int(content_length_raw)
            except Exception:
                content_length = None
            else:
                if content_length > max_bytes:
                    record_security_event(
                        "upstream_download_blocked",
                        {
                            "reason": "content_length_exceeded",
                            "max_bytes": max_bytes,
                            "content_length": content_length,
                        },
                    )
                    raise ValueError(
                        f"Download blocked: content-length {content_length} exceeds limit {max_bytes}."
                    )

        bytes_written = 0
        with open(output_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    record_security_event(
                        "upstream_download_blocked",
                        {
                            "reason": "stream_size_exceeded",
                            "max_bytes": max_bytes,
                            "bytes_written": bytes_written,
                        },
                    )
                    raise ValueError(
                        f"Download blocked: stream exceeded limit {max_bytes} bytes."
                    )
                file_handle.write(chunk)
        record_security_event(
            "upstream_download_success",
            {"bytes_written": bytes_written, "max_bytes": max_bytes},
        )
        return True
    except Exception as exc:
        record_security_event("upstream_download_failed", {"error": str(exc)})
        logger.warning("Error downloading image: %s", exc)
        try:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False
