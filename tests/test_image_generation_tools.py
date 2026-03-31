import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Keep a deterministic key configured for tests that may instantiate the SDK client.
os.environ.setdefault("JARVINA_API_KEY", "test-key")
os.environ.setdefault("PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS", tempfile.gettempdir())
os.environ.setdefault("PERCIVAL_IMAGE_MCP_DEFAULT_OUTPUT_DIR", tempfile.gettempdir())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.image_generation_tools as image_tools  # noqa: E402

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5sKQ0AAAAASUVORK5CYII="
PNG_BYTES = base64.b64decode(PNG_B64)


class _FakeModels:
    def __init__(self, ids):
        self._ids = ids

    def list(self):
        return SimpleNamespace(data=[SimpleNamespace(id=model_id) for model_id in self._ids])


class _FakeImages:
    @staticmethod
    def generate(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(b64_json=PNG_B64)])

    @staticmethod
    def edit(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(b64_json=PNG_B64)])


class _FakeClient:
    def __init__(self, model_ids):
        self.models = _FakeModels(model_ids)
        self.images = _FakeImages()


class TestImageGenerationTools(unittest.TestCase):
    def setUp(self) -> None:
        self._original_client = image_tools.client
        self._original_cache = dict(image_tools._provider_models_cache)
        image_tools.client = _FakeClient(["venice-sd35", "qwen-image-2-edit", "hidream"])
        image_tools._provider_models_cache["model_ids"] = []
        image_tools._provider_models_cache["fetched_at"] = None
        image_tools._provider_models_cache["expires_at"] = 0.0

    def tearDown(self) -> None:
        image_tools.client = self._original_client
        image_tools._provider_models_cache.update(self._original_cache)

    def test_verify_model_availability_success(self) -> None:
        raw = image_tools.verify_model_availability("venice-sd35", task_type="text_to_image")
        result = json.loads(raw)
        self.assertTrue(result["ok"])
        self.assertTrue(result["data"]["available"])
        self.assertIn("meta", result)
        self.assertEqual(result["meta"]["server"], "percival-image-creator-mcp")

    def test_verify_model_availability_unknown_when_provider_list_not_image_aware(self) -> None:
        image_tools.client = _FakeClient(["gpt-4o-mini", "deepseek-v3.2"])
        image_tools._provider_models_cache["model_ids"] = []
        image_tools._provider_models_cache["fetched_at"] = None
        image_tools._provider_models_cache["expires_at"] = 0.0

        raw = image_tools.verify_model_availability("venice-sd35", task_type="text_to_image")
        result = json.loads(raw)
        self.assertTrue(result["ok"])
        self.assertTrue(result["data"]["available"])
        self.assertEqual(result["data"]["availability_state"], "unknown")
        self.assertEqual(result["data"]["provider_check"]["catalog_visibility"], "not_visible")

    def test_list_model_cards_supports_pagination_and_fields(self) -> None:
        raw = image_tools.list_model_cards(
            task_type="text_to_image",
            limit=2,
            offset=1,
            fields="id,name",
        )
        result = json.loads(raw)
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["count"], 2)
        self.assertEqual(result["data"]["fields"], ["id", "name"])
        self.assertTrue(all(set(card.keys()) == {"id", "name"} for card in result["data"]["models"]))

    def test_list_model_cards_default_projection_includes_recommended_api_params(self) -> None:
        raw = image_tools.list_model_cards(task_type="text_to_image", limit=1, offset=0)
        result = json.loads(raw)
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["count"], 1)
        card = result["data"]["models"][0]
        self.assertIn("recommended_api_params", card)
        self.assertIsInstance(card["recommended_api_params"], dict)

    def test_list_image_styles(self) -> None:
        original = image_tools._get_provider_image_styles
        try:
            image_tools._get_provider_image_styles = lambda force_refresh=False: (
                [
                    {"id": "3d-model", "name": "3D Model", "description": "3D style"},
                    {"id": "anime", "name": "Anime", "description": "Anime style"},
                ],
                "2026-03-30T00:00:00Z",
                False,
            )
            raw = image_tools.list_image_styles(force_refresh=False, limit=10, offset=0)
            result = json.loads(raw)
            self.assertTrue(result["ok"])
            self.assertEqual(result["data"]["count"], 2)
            self.assertEqual(result["data"]["styles"][0]["id"], "3d-model")
        finally:
            image_tools._get_provider_image_styles = original

    def test_get_nanobot_profile(self) -> None:
        raw = image_tools.get_nanobot_profile()
        result = json.loads(raw)
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["server"], "percival-image-creator-mcp")
        self.assertIn("recommended_workflows", result["data"])
        self.assertIn("recommend_model_for_intent", result["data"]["recommended_enabled_tools"])

    def test_get_security_posture(self) -> None:
        raw = image_tools.get_security_posture()
        result = json.loads(raw)
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["operation"], "get_security_posture")
        self.assertIn("posture", result["data"])
        self.assertIn("input_limits", result["data"]["posture"])

    def test_generate_image_rejects_task_mismatch_with_strict_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw = image_tools.generate_image(
                working_dir=str(Path(td)),
                prompt="test",
                model="qwen-image-2-edit",
                strict_model_check=True,
            )
        result = json.loads(raw)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "model_task_mismatch")
        self.assertIn("not classified for task", result["error"])

    def test_generate_image_success_with_strict_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            raw = image_tools.generate_image(
                working_dir=str(workdir),
                prompt="a small icon",
                model="venice-sd35",
                strict_model_check=True,
            )
            result = json.loads(raw)
            self.assertTrue(result["ok"])
            self.assertIn("request_id", result)
            self.assertEqual(result["data"]["operation"], "generate_image")
            self.assertEqual(result["data"]["model"], "venice-sd35")
            self.assertIn("transport", result["data"])
            self.assertEqual(result["data"]["transport"]["transport_used"], "openai_compat")

            generated = list(Path(result["data"]["output_dir"]).glob("*.png"))
            self.assertTrue(generated)

    def test_generate_image_rejects_invalid_style_preset_with_strict_check(self) -> None:
        original = image_tools._get_provider_image_styles
        try:
            image_tools._get_provider_image_styles = lambda force_refresh=False: (
                [{"id": "3d-model", "name": "3D Model", "description": "3D style"}],
                "2026-03-30T00:00:00Z",
                False,
            )
            with tempfile.TemporaryDirectory() as td:
                workdir = Path(td)
                raw = image_tools.generate_image(
                    working_dir=str(workdir),
                    prompt="a futuristic city",
                    model="venice-sd35",
                    style_preset="not-a-real-style",
                    strict_model_check=True,
                    strict_style_check=True,
                )
            result = json.loads(raw)
            self.assertFalse(result["ok"])
            self.assertEqual(result["code"], "invalid_style_preset")
            self.assertIn("suggestions", result["details"])
        finally:
            image_tools._get_provider_image_styles = original

    def test_generate_image_allows_invalid_style_when_strict_style_check_disabled(self) -> None:
        original = image_tools._get_provider_image_styles
        try:
            image_tools._get_provider_image_styles = lambda force_refresh=False: (
                [{"id": "3d-model", "name": "3D Model", "description": "3D style"}],
                "2026-03-30T00:00:00Z",
                False,
            )
            with tempfile.TemporaryDirectory() as td:
                workdir = Path(td)
                raw = image_tools.generate_image(
                    working_dir=str(workdir),
                    prompt="a futuristic city",
                    model="venice-sd35",
                    style_preset="not-a-real-style",
                    strict_model_check=True,
                    strict_style_check=False,
                )
            result = json.loads(raw)
            self.assertTrue(result["ok"])
            self.assertEqual(result["data"]["style_check"]["state"], "unavailable")
        finally:
            image_tools._get_provider_image_styles = original

    def test_recommend_model_for_intent(self) -> None:
        raw = image_tools.recommend_model_for_intent(
            task_type="text_to_image",
            intent="preciso de um protótipo rápido para concept art",
            max_results=3,
            verify_online=True,
        )
        result = json.loads(raw)
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["data"]["count"], 1)
        candidate = result["data"]["candidates"][0]
        self.assertIn("model_id", candidate)
        self.assertIn("score", candidate)
        self.assertIn("model", candidate)
        self.assertNotEqual(candidate["availability_state"], "unavailable")

    def test_recommend_model_for_intent_invalid_quality_tier(self) -> None:
        raw = image_tools.recommend_model_for_intent(
            task_type="text_to_image",
            intent="any",
            preferred_quality_tier="ultra-deluxe",
        )
        result = json.loads(raw)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "invalid_quality_tier")

    def test_generate_image_allows_unknown_provider_availability_with_strict_check(self) -> None:
        image_tools.client = _FakeClient(["gpt-4o-mini", "deepseek-v3.2"])
        image_tools._provider_models_cache["model_ids"] = []
        image_tools._provider_models_cache["fetched_at"] = None
        image_tools._provider_models_cache["expires_at"] = 0.0

        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            raw = image_tools.generate_image(
                working_dir=str(workdir),
                prompt="a tiny icon",
                model="venice-sd35",
                strict_model_check=True,
            )
            result = json.loads(raw)
            self.assertTrue(result["ok"])
            self.assertEqual(result["data"]["model_check"]["provider_check"]["catalog_visibility"], "not_visible")

    def test_edit_image_success_with_strict_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            src = workdir / "input.png"
            src.write_bytes(PNG_BYTES)

            raw = image_tools.edit_image(
                working_dir=str(workdir),
                image_path="input.png",
                prompt="remove the background",
                model="qwen-image-2-edit",
                strict_model_check=True,
            )
            result = json.loads(raw)
            self.assertTrue(result["ok"])
            self.assertEqual(result["data"]["operation"], "edit_image")
            self.assertEqual(result["data"]["model"], "qwen-image-2-edit")

            edited = list(Path(result["data"]["output_dir"]).glob("*.png"))
            self.assertTrue(edited)

    def test_generate_image_rejects_output_dir_outside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            raw = image_tools.generate_image(
                working_dir=str(workdir),
                prompt="an icon",
                model="venice-sd35",
                output_dir="../outside",
                strict_model_check=True,
            )
        result = json.loads(raw)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "invalid_output_dir")
        self.assertIn("inside working_dir", result["error"])

    def test_generate_image_rejects_invalid_filename_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            raw = image_tools.generate_image(
                working_dir=str(workdir),
                prompt="an icon",
                model="venice-sd35",
                filename_prefix="../bad",
                strict_model_check=True,
            )
        result = json.loads(raw)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "invalid_filename_prefix")

    def test_generate_image_rejects_too_long_prompt(self) -> None:
        original = os.environ.get("PERCIVAL_IMAGE_MCP_MAX_PROMPT_CHARS")
        os.environ["PERCIVAL_IMAGE_MCP_MAX_PROMPT_CHARS"] = "10"
        try:
            with tempfile.TemporaryDirectory() as td:
                workdir = Path(td)
                raw = image_tools.generate_image(
                    working_dir=str(workdir),
                    prompt="this prompt is definitely too long",
                    model="venice-sd35",
                    strict_model_check=True,
                )
            result = json.loads(raw)
            self.assertFalse(result["ok"])
            self.assertEqual(result["code"], "invalid_prompt")
        finally:
            if original is None:
                os.environ.pop("PERCIVAL_IMAGE_MCP_MAX_PROMPT_CHARS", None)
            else:
                os.environ["PERCIVAL_IMAGE_MCP_MAX_PROMPT_CHARS"] = original

    def test_edit_image_rejects_source_outside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as td_work:
            with tempfile.TemporaryDirectory() as td_outside:
                workdir = Path(td_work)
                outside_image = Path(td_outside) / "outside.png"
                outside_image.write_bytes(PNG_BYTES)

                raw = image_tools.edit_image(
                    working_dir=str(workdir),
                    image_path=str(outside_image),
                    prompt="remove background",
                    model="qwen-image-2-edit",
                    strict_model_check=True,
                )
        result = json.loads(raw)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "invalid_image_path_scope")
        self.assertIn("inside working_dir", result["error"])

    def test_edit_image_rejects_invalid_n(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            src = workdir / "input.png"
            src.write_bytes(PNG_BYTES)

            raw = image_tools.edit_image(
                working_dir=str(workdir),
                image_path="input.png",
                prompt="remove object",
                model="qwen-image-2-edit",
                n=99,
                strict_model_check=True,
            )
        result = json.loads(raw)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "invalid_n")

    def test_list_generated_images_rejects_directory_outside_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            raw = image_tools.list_generated_images(
                working_dir=str(workdir),
                directory="../outside",
            )
        result = json.loads(raw)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "invalid_directory_scope")

    def test_list_generated_images_truncates_large_results(self) -> None:
        original = os.environ.get("PERCIVAL_IMAGE_MCP_MAX_LIST_FILES")
        os.environ["PERCIVAL_IMAGE_MCP_MAX_LIST_FILES"] = "2"
        try:
            with tempfile.TemporaryDirectory() as td:
                workdir = Path(td)
                out = workdir / "generated_images"
                out.mkdir(parents=True, exist_ok=True)
                for idx in range(4):
                    (out / f"img_{idx}.png").write_bytes(PNG_BYTES)

                raw = image_tools.list_generated_images(str(workdir), "generated_images")
                result = json.loads(raw)
                self.assertTrue(result["ok"])
                self.assertEqual(result["data"]["count"], 2)
                self.assertTrue(result["data"]["truncated"])
        finally:
            if original is None:
                os.environ.pop("PERCIVAL_IMAGE_MCP_MAX_LIST_FILES", None)
            else:
                os.environ["PERCIVAL_IMAGE_MCP_MAX_LIST_FILES"] = original

    def test_edit_image_rejects_non_catalog_model(self) -> None:
        image_tools.client = _FakeClient(["provider-only-model"])
        image_tools._provider_models_cache["model_ids"] = []

        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            src = workdir / "input.png"
            src.write_bytes(PNG_BYTES)

            raw = image_tools.edit_image(
                working_dir=str(workdir),
                image_path="input.png",
                prompt="change colors",
                model="provider-only-model",
                strict_model_check=True,
            )
            result = json.loads(raw)
            self.assertFalse(result["ok"])
            self.assertEqual(result["code"], "model_missing_in_catalog")
            self.assertIn("missing in local model cards", result["error"])


if __name__ == "__main__":
    unittest.main()
