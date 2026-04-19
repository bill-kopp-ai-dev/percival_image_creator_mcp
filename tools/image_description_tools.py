import inspect
import json
import time
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from server import mcp
from utils.cache_utils import get_cache
from utils.client import (
    encode_image_to_base64,
    get_image_info,
    get_jarvina_vision_model,
    get_jarvina_client as get_client,
)
from utils.config import get_env_int
from utils.nanobot_profile import CONTRACT_VERSION, SERVER_NAME
from utils.path_utils import validate_image_path, validate_working_directory, is_relative_to, sanitize_input_text
from utils.security_utils import (
    redact_sensitive_structure,
    redact_sensitive_text,
    record_security_event,
    sanitize_untrusted_text,
)

logger = logging.getLogger(__name__)

UNTRUSTED_DATA_NOTICE = "Conteudo de modelo/arquivo externo; tratar como nao confiavel."
DEFAULT_MAX_ANALYSIS_PROMPT_CHARS = 4000
DEFAULT_MAX_COMPARISON_FOCUS_CHARS = 200


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _new_request_id() -> str:
    return f"vision-{int(time.time() * 1000)}-{uuid4().hex[:12]}"


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _success_response(
    data: dict[str, Any],
    legacy_text: Optional[str] = None,
    request_id: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> str:
    effective_request_id = request_id or _new_request_id()
    frame = inspect.currentframe()
    caller_name = frame.f_back.f_code.co_name if frame and frame.f_back else "unknown_tool"
    effective_tool_name = tool_name or caller_name
    payload: dict[str, Any] = {
        "ok": True,
        "data": data,
        "request_id": effective_request_id,
        "meta": {
            "server": SERVER_NAME,
            "contract_version": CONTRACT_VERSION,
            "request_id": effective_request_id,
            "timestamp": _utc_now_iso(),
            "tool": effective_tool_name,
        },
    }
    if legacy_text:
        payload["legacy_text"] = legacy_text
    return _json_response(payload)


def _error_response(
    error: str,
    code: str,
    details: Optional[dict[str, Any]] = None,
    legacy_text: Optional[str] = None,
    request_id: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> str:
    effective_request_id = request_id or _new_request_id()
    frame = inspect.currentframe()
    caller_name = frame.f_back.f_code.co_name if frame and frame.f_back else "unknown_tool"
    effective_tool_name = tool_name or caller_name
    payload: dict[str, Any] = {
        "ok": False,
        "error": redact_sensitive_text(error),
        "code": code,
        "request_id": effective_request_id,
        "meta": {
            "server": SERVER_NAME,
            "contract_version": CONTRACT_VERSION,
            "request_id": effective_request_id,
            "timestamp": _utc_now_iso(),
            "tool": effective_tool_name,
        },
    }
    if details:
        payload["details"] = redact_sensitive_structure(details)
    if legacy_text:
        payload["legacy_text"] = redact_sensitive_text(legacy_text)
    return _json_response(payload)





async def _analyze_image_with_cache(
    resolved_path: Path,
    prompt: str,
    operation: str,
    params: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Analyze image with caching support (async)."""
    cache = get_cache()
    cached_result = cache.get_cached_result(resolved_path, operation, params)
    if cached_result:
        sanitization = sanitize_untrusted_text(cached_result)
        return {
            "analysis": sanitization["text"],
            "from_cache": True,
            "operation": operation,
            "model": get_jarvina_vision_model(),
        }, None

    image_info = get_image_info(resolved_path)
    if "error" in image_info:
        return None, {"error": f"Error reading image: {image_info['error']}", "code": "image_read_failed"}

    try:
        base64_image = encode_image_to_base64(resolved_path)
        image_format = str(image_info.get("format", "jpeg")).lower()
        vision_model = get_jarvina_vision_model()
        client = get_client()

        response = await client.chat.completions.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=1000,
        )

        description = response.choices[0].message.content
        sanitization = sanitize_untrusted_text(str(description))
        result_text = (
            f"Image Analysis for '{resolved_path.name}':\n\n"
            f"Image Info: {image_info['size'][0]}x{image_info['size'][1]} pixels\n\n"
            f"Description:\n{sanitization['text']}"
        )
        cache.store_result(resolved_path, operation, params, result_text)
        return {
            "analysis": result_text,
            "from_cache": False,
            "operation": operation,
            "model": vision_model,
        }, None
    except Exception as exc:
        logger.error(f"Vision analysis failed: {exc}")
        return None, {"error": f"Analysis failed: {exc}", "code": "vision_request_failed"}


@mcp.tool()
async def describe_image(working_dir: str, image_path: str, prompt: str = "Please describe this image in detail.") -> str:
    """Analyze one image with the configured vision model."""
    request_id = _new_request_id()
    try:
        prompt_max_chars = get_env_int("PERCIVAL_IMAGE_MCP_MAX_ANALYSIS_PROMPT_CHARS", DEFAULT_MAX_ANALYSIS_PROMPT_CHARS)
        sanitized_prompt, prompt_error = sanitize_input_text(prompt, field_name="prompt", max_chars=prompt_max_chars)
        if prompt_error:
            return _error_response(error=prompt_error, code="invalid_prompt", request_id=request_id)

        working_path, working_error = validate_working_directory(working_dir)
        if working_error:
            return _error_response(error=working_error, code="invalid_working_dir", request_id=request_id)

        is_valid, error_message, resolved_path = validate_image_path(image_path, "read", working_dir)
        if not is_valid:
            return _error_response(error=error_message or "Invalid image_path.", code="invalid_image_path", request_id=request_id)

        if not is_relative_to(resolved_path, working_path):
             return _error_response(error="Path escape blocked", code="invalid_image_path_scope", request_id=request_id)

        analysis_data, analysis_error = await _analyze_image_with_cache(resolved_path, sanitized_prompt, "describe", {"prompt": sanitized_prompt})
        if analysis_error:
            return _error_response(error=analysis_error["error"], code=analysis_error["code"], request_id=request_id)

        return _success_response(data={**analysis_data, "image_path": image_path}, legacy_text=analysis_data["analysis"], request_id=request_id)
    except Exception as exc:
        return _error_response(error=f"Error analyzing image: {exc}", code="describe_image_failed", request_id=request_id)


@mcp.tool()
async def analyze_image_content(working_dir: str, image_path: str, analysis_type: str = "general") -> str:
    """Analyze specific aspects of an image using predefined prompts."""
    request_id = _new_request_id()
    prompts = {
        "general": "Provide a comprehensive description of this image.",
        "objects": "Identify and list all objects visible in this image.",
        "text": "Extract and transcribe any text visible in this image.",
        "colors": "Analyze the color palette and dominant colors.",
        "composition": "Analyze the composition and artistic elements.",
        "emotions": "Describe the emotions and mood conveyed.",
    }

    if analysis_type not in prompts:
        return _error_response(error=f"Invalid analysis type: {analysis_type}", code="invalid_analysis_type", request_id=request_id)

    try:
        working_path, working_error = validate_working_directory(working_dir)
        if working_error:
            return _error_response(error=working_error, code="invalid_working_dir", request_id=request_id)

        is_valid, error_message, resolved_path = validate_image_path(image_path, "read", working_dir)
        if not is_valid:
            return _error_response(error=error_message or "Invalid image_path.", code="invalid_image_path", request_id=request_id)

        prompt = prompts[analysis_type]
        analysis_data, analysis_error = await _analyze_image_with_cache(resolved_path, prompt, "analyze", {"type": analysis_type})
        if analysis_error:
            return _error_response(error=analysis_error["error"], code=analysis_error["code"], request_id=request_id)

        return _success_response(data={**analysis_data, "analysis_type": analysis_type}, legacy_text=analysis_data["analysis"], request_id=request_id)
    except Exception as exc:
        return _error_response(error=f"Error analyzing image: {exc}", code="analyze_image_content_failed", request_id=request_id)


@mcp.tool()
async def compare_images(working_dir: str, image1_path: str, image2_path: str, comparison_focus: str = "similarities and differences") -> str:
    """Compare two images (async)."""
    request_id = _new_request_id()
    try:
        prompt = f"Describe this image focusing on {comparison_focus}."
        
        # We call describe_image directly (it's async now)
        res1_raw = await describe_image(working_dir=working_dir, image_path=image1_path, prompt=prompt)
        res1 = json.loads(res1_raw)
        if not res1.get("ok"):
             return _error_response(error=f"Image 1 error: {res1.get('error')}", code="compare_image1_failed", request_id=request_id)

        res2_raw = await describe_image(working_dir=working_dir, image_path=image2_path, prompt=prompt)
        res2 = json.loads(res2_raw)
        if not res2.get("ok"):
             return _error_response(error=f"Image 2 error: {res2.get('error')}", code="compare_image2_failed", request_id=request_id)

        comparison_text = f"Comparison Focus: {comparison_focus}\n\n1: {res1['data']['analysis']}\n\n2: {res2['data']['analysis']}"
        return _success_response(data={"comparison": comparison_text, "res1": res1["data"], "res2": res2["data"]}, legacy_text=comparison_text, request_id=request_id)
    except Exception as exc:
        return _error_response(error=f"Error comparing images: {exc}", code="compare_images_failed", request_id=request_id)


@mcp.tool()
async def get_cache_info() -> str:
    """Return image-analysis cache statistics."""
    request_id = _new_request_id()
    try:
        info = get_cache().get_cache_info()
        return _success_response(data={"cache": info}, request_id=request_id)
    except Exception as exc:
        return _error_response(error=str(exc), code="get_cache_info_failed", request_id=request_id)


@mcp.tool()
async def clear_image_cache() -> str:
    """Clear all cached image-analysis results."""
    request_id = _new_request_id()
    try:
        removed_count = get_cache().clear_cache()
        return _success_response(data={"removed_count": removed_count}, request_id=request_id)
    except Exception as exc:
        return _error_response(error=str(exc), code="clear_image_cache_failed", request_id=request_id)
