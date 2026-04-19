from __future__ import annotations

from typing import Any


SERVER_NAME = "percival-image-creator-mcp"
CONTRACT_VERSION = "2026-03-s5"


def build_nanobot_profile() -> dict[str, Any]:
    """
    Return a compact machine-readable profile for nanobot orchestration.
    """
    return {
        "server": SERVER_NAME,
        "contract_version": CONTRACT_VERSION,
        "response_contract": {
            "success": {"ok": True, "data": {}, "meta": {}, "legacy_text": "optional"},
            "error": {
                "ok": False,
                "error": "message",
                "code": "error_code",
                "details": {},
                "meta": {},
                "legacy_text": "optional",
            },
            "notes": [
                "legacy_text is compatibility-only and may be omitted in future major versions.",
                "Agents should rely on structured fields (ok/data/error/code/details/meta).",
            ],
        },
        "recommended_workflows": {
            "text_to_image": [
                "recommend_model_for_intent(task_type='text_to_image', intent='<human_goal>')",
                "list_model_cards(task_type='text_to_image')",
                "get_model_card(model_id) [optional]",
                "list_image_styles() [optional when style_preset is desired]",
                "verify_model_availability(model_id, task_type='text_to_image')",
                "generate_image(...)",
            ],
            "image_edit": [
                "recommend_model_for_intent(task_type='image_edit', intent='<human_goal>')",
                "list_model_cards(task_type='image_edit')",
                "verify_model_availability(model_id, task_type='image_edit')",
                "edit_image(...)",
            ],
            "image_analysis": [
                "describe_image(...) or analyze_image_content(...)",
                "compare_images(...) [optional]",
                "get_image_metadata(...) [optional]",
            ],
        },
        "recommended_enabled_tools": [
            "recommend_model_for_intent",
            "list_model_cards",
            "get_model_card",
            "list_image_styles",
            "verify_model_availability",
            "get_security_metrics",
            "clear_security_metrics",
            "get_security_posture",
            "generate_image",
            "edit_image",
            "describe_image",
            "analyze_image_content",
            "compare_images",
            "get_image_metadata",
            "list_generated_images",
        ],
        "defaults": {
            "strict_model_check": True,
            "provider_model_cache_ttl_seconds": 300,
            "image_edit_max_n": 10,
            "default_output_dir": "~/Pictures",
        },
        "generation_parameter_contract": {
            "precedence": [
                "explicit_generate_image_args",
                "parameter_overrides_json",
                "model_card.recommended_api_params",
                "server_defaults",
            ],
            "supported_provider_params": [
                "aspect_ratio",
                "resolution",
                "cfg_scale",
                "negative_prompt",
                "steps",
                "style_preset",
                "safe_mode",
                "variants",
                "seed",
                "format",
                "lora_strength",
                "width",
                "height",
                "hide_watermark",
                "embed_exif_metadata",
                "enable_web_search",
            ],
            "style_discovery_tool": "list_image_styles",
            "style_validation_defaults": {
                "strict_style_check": True,
                "force_style_refresh": False,
            },
        },
        "transport_modes": {
            "env_var": "PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT",
            "allowed_values": ["auto", "openai_compat", "venice_native"],
            "fallback_env_var": "PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT_FALLBACK",
            "default_mode": "auto",
        },
    }
