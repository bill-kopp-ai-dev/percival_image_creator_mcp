import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.venice_image_payload import (  # noqa: E402
    build_venice_generation_request,
    parse_parameter_overrides_json,
)


class TestVeniceImagePayload(unittest.TestCase):
    def setUp(self) -> None:
        self._original_env = os.environ.get("PERCIVAL_IMAGE_MCP_GENERATION_OVERRIDES_JSON")
        os.environ.pop("PERCIVAL_IMAGE_MCP_GENERATION_OVERRIDES_JSON", None)

    def tearDown(self) -> None:
        if self._original_env is None:
            os.environ.pop("PERCIVAL_IMAGE_MCP_GENERATION_OVERRIDES_JSON", None)
        else:
            os.environ["PERCIVAL_IMAGE_MCP_GENERATION_OVERRIDES_JSON"] = self._original_env

    def test_precedence_explicit_overrides_catalog_and_runtime(self) -> None:
        payload = build_venice_generation_request(
            model="venice-sd35",
            prompt="a test prompt",
            size="1024x1024",
            explicit_params={"cfg_scale": 9.0, "format": "png"},
            card_recommended_params={"cfg_scale": 4.5, "format": "webp"},
            runtime_overrides={"cfg_scale": 6.0},
        )
        provider_params = payload["resolved_provider_params"]
        self.assertEqual(provider_params["cfg_scale"], 9.0)
        self.assertEqual(provider_params["format"], "png")
        self.assertEqual(payload["param_sources"]["cfg_scale"], "explicit_input")
        self.assertEqual(payload["param_sources"]["format"], "explicit_input")

    def test_runtime_overrides_json_is_applied(self) -> None:
        os.environ["PERCIVAL_IMAGE_MCP_GENERATION_OVERRIDES_JSON"] = '{"safe_mode": false, "steps": 12}'
        payload = build_venice_generation_request(
            model="venice-sd35",
            prompt="a test prompt",
            size="1024x1024",
        )
        provider_params = payload["resolved_provider_params"]
        self.assertFalse(provider_params["safe_mode"])
        self.assertEqual(provider_params["steps"], 12)
        self.assertIn("safe_mode", payload["runtime_override_keys"])
        self.assertIn("steps", payload["runtime_override_keys"])

    def test_parse_parameter_overrides_requires_object(self) -> None:
        with self.assertRaises(ValueError):
            parse_parameter_overrides_json('["not", "an", "object"]')

    def test_invalid_format_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_venice_generation_request(
                model="venice-sd35",
                prompt="a test prompt",
                size="1024x1024",
                explicit_params={"format": "bmp"},
            )


if __name__ == "__main__":
    unittest.main()
