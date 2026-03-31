import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JARVINA_API_KEY", "test-key")
os.environ.setdefault("PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS", tempfile.gettempdir())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.client as client_module  # noqa: E402


class _FakeResponse:
    def __init__(self, *, chunks: list[bytes], content_length: int | None = None, url: str):
        self._chunks = chunks
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.url = url

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 8192):
        del chunk_size
        for chunk in self._chunks:
            yield chunk


class _FakePostResponse:
    def __init__(self, *, payload: dict):
        self._payload = payload
        self.text = str(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeErrorPostResponse:
    def __init__(self, *, payload: dict, message: str = "400 Client Error: Bad Request"):
        self._payload = payload
        self.text = str(payload)
        self._message = message

    def raise_for_status(self) -> None:
        raise client_module.requests.HTTPError(self._message)

    def json(self):
        return self._payload


class _FakeGetJsonResponse:
    def __init__(self, *, payload: dict):
        self._payload = payload
        self.text = str(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeOpenAIClient:
    last_kwargs = None

    class images:
        @staticmethod
        def generate(**kwargs):
            _FakeOpenAIClient.last_kwargs = kwargs
            return SimpleNamespace(data=[SimpleNamespace(b64_json="ZmFrZQ==", url=None)])


class TestClientSecurity(unittest.TestCase):
    _ENV_KEYS = [
        "JARVINA_BASE_URL",
        "PERCIVAL_IMAGE_MCP_ALLOW_INSECURE_PROVIDER_URL",
        "PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL",
        "PERCIVAL_IMAGE_MCP_ALLOWED_PROVIDER_HOSTS",
        "PERCIVAL_IMAGE_MCP_ALLOW_HTTP_DOWNLOADS",
        "PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_DOWNLOADS",
        "PERCIVAL_IMAGE_MCP_ALLOWED_DOWNLOAD_HOSTS",
        "PERCIVAL_IMAGE_MCP_DOWNLOAD_MAX_BYTES",
        "PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT",
        "PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT_FALLBACK",
        "PERCIVAL_IMAGE_MCP_VENICE_NATIVE_RETRIES",
    ]

    def setUp(self) -> None:
        self._env_backup = {k: os.environ.get(k) for k in self._ENV_KEYS}
        self._original_get = client_module.requests.get
        self._original_post = client_module.requests.post
        self._original_is_venice = client_module._is_venice_base_url
        self._original_instance = client_module._client_instance
        client_module._client_instance = None
        _FakeOpenAIClient.last_kwargs = None

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        client_module.requests.get = self._original_get
        client_module.requests.post = self._original_post
        client_module._is_venice_base_url = self._original_is_venice
        client_module._client_instance = self._original_instance

    def test_validate_download_url_blocks_http_by_default(self) -> None:
        with self.assertRaises(ValueError):
            client_module.validate_outbound_url("http://example.com/image.png", purpose="download")

    def test_validate_download_url_blocks_private_ip_by_default(self) -> None:
        with self.assertRaises(ValueError):
            client_module.validate_outbound_url("https://127.0.0.1/image.png", purpose="download")

    def test_validate_download_url_allows_private_ip_with_opt_in(self) -> None:
        os.environ["PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_DOWNLOADS"] = "true"
        validated = client_module.validate_outbound_url("https://127.0.0.1/image.png", purpose="download")
        self.assertEqual(validated, "https://127.0.0.1/image.png")

    def test_get_jarvina_client_blocks_private_provider_url(self) -> None:
        os.environ["JARVINA_BASE_URL"] = "https://127.0.0.1/v1"
        os.environ.pop("PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL", None)
        with self.assertRaises(ValueError):
            client_module.get_jarvina_client()

    def test_download_image_from_url_blocks_when_stream_exceeds_limit(self) -> None:
        os.environ["PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_DOWNLOADS"] = "true"
        os.environ["PERCIVAL_IMAGE_MCP_DOWNLOAD_MAX_BYTES"] = "10"

        def _fake_get(url: str, timeout: int, stream: bool):
            self.assertTrue(stream)
            self.assertEqual(timeout, 30)
            return _FakeResponse(chunks=[b"12345", b"67890", b"x"], url=url)

        client_module.requests.get = _fake_get

        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "img.png"
            ok = client_module.download_image_from_url("https://127.0.0.1/image.png", output)
            self.assertFalse(ok)
            self.assertFalse(output.exists())

    def test_download_image_from_url_success(self) -> None:
        os.environ["PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_DOWNLOADS"] = "true"
        os.environ["PERCIVAL_IMAGE_MCP_DOWNLOAD_MAX_BYTES"] = "100"

        def _fake_get(url: str, timeout: int, stream: bool):
            self.assertTrue(stream)
            self.assertEqual(timeout, 30)
            return _FakeResponse(chunks=[b"123", b"456"], content_length=6, url=url)

        client_module.requests.get = _fake_get

        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "img.png"
            ok = client_module.download_image_from_url("https://127.0.0.1/image.png", output)
            self.assertTrue(ok)
            self.assertTrue(output.exists())
            self.assertEqual(output.read_bytes(), b"123456")

    def test_generate_images_with_transport_openai_compat(self) -> None:
        os.environ["PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT"] = "openai_compat"
        response, transport = client_module.generate_images_with_transport(
            {"model": "venice-sd35", "prompt": "test", "size": "1024x1024"},
            openai_client=_FakeOpenAIClient(),
        )
        self.assertEqual(transport["transport_used"], "openai_compat")
        self.assertFalse(transport["fallback_used"])
        self.assertEqual(len(response.data), 1)
        self.assertEqual(transport["compat_dropped_keys"], [])

    def test_generate_images_with_transport_venice_native(self) -> None:
        os.environ["PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT"] = "venice_native"
        os.environ["JARVINA_BASE_URL"] = "https://127.0.0.1/api/v1"
        os.environ["PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL"] = "true"

        def _fake_post(url: str, json: dict, headers: dict, timeout: int):
            self.assertTrue(url.endswith("/api/v1/image/generate"))
            self.assertIn("Authorization", headers)
            self.assertGreater(timeout, 0)
            self.assertEqual(json["model"], "venice-sd35")
            self.assertNotIn("size", json)
            self.assertEqual(json["width"], 1024)
            self.assertEqual(json["height"], 1024)
            return _FakePostResponse(payload={"images": ["ZmFrZQ=="]})

        client_module.requests.post = _fake_post

        response, transport = client_module.generate_images_with_transport(
            {"model": "venice-sd35", "prompt": "test", "size": "1024x1024"},
            openai_client=_FakeOpenAIClient(),
        )
        self.assertEqual(transport["transport_used"], "venice_native")
        self.assertFalse(transport["fallback_used"])
        self.assertEqual(response.data[0].b64_json, "ZmFrZQ==")
        self.assertEqual(transport["native_dropped_keys"], [])

    def test_generate_images_with_transport_venice_native_retries_unrecognized_keys(self) -> None:
        os.environ["PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT"] = "venice_native"
        os.environ["PERCIVAL_IMAGE_MCP_VENICE_NATIVE_RETRIES"] = "1"
        os.environ["JARVINA_BASE_URL"] = "https://127.0.0.1/api/v1"
        os.environ["PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL"] = "true"
        call_count = {"value": 0}

        def _fake_post(url: str, json: dict, headers: dict, timeout: int):
            self.assertTrue(url.endswith("/api/v1/image/generate"))
            self.assertIn("Authorization", headers)
            self.assertGreater(timeout, 0)
            self.assertNotIn("size", json)
            call_count["value"] += 1
            if call_count["value"] == 1:
                self.assertIn("cfg_scale", json)
                return _FakeErrorPostResponse(
                    payload={
                        "details": {"_errors": ["Unrecognized key(s) in object: 'cfg_scale'"]},
                        "issues": [
                            {
                                "code": "unrecognized_keys",
                                "keys": ["cfg_scale"],
                                "path": [],
                                "message": "Unrecognized key(s) in object: 'cfg_scale'",
                            }
                        ],
                    }
                )
            self.assertNotIn("cfg_scale", json)
            return _FakePostResponse(payload={"images": ["ZmFrZQ=="]})

        client_module.requests.post = _fake_post

        response, transport = client_module.generate_images_with_transport(
            {
                "model": "venice-sd35",
                "prompt": "test",
                "size": "1024x1024",
                "extra_body": {"cfg_scale": 1.0},
            },
            openai_client=_FakeOpenAIClient(),
        )
        self.assertEqual(call_count["value"], 2)
        self.assertEqual(transport["transport_used"], "venice_native")
        self.assertFalse(transport["fallback_used"])
        self.assertEqual(transport["native_dropped_keys"], ["cfg_scale"])
        self.assertEqual(response.data[0].b64_json, "ZmFrZQ==")

    def test_generate_images_with_transport_fallback(self) -> None:
        os.environ["PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT"] = "venice_native"
        os.environ["PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT_FALLBACK"] = "true"
        os.environ["JARVINA_BASE_URL"] = "https://127.0.0.1/api/v1"
        os.environ["PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL"] = "true"

        def _fake_post(url: str, json: dict, headers: dict, timeout: int):
            del url, json, headers, timeout
            raise RuntimeError("simulated native failure")

        client_module.requests.post = _fake_post

        response, transport = client_module.generate_images_with_transport(
            {"model": "venice-sd35", "prompt": "test", "size": "1024x1024"},
            openai_client=_FakeOpenAIClient(),
        )
        self.assertEqual(transport["transport_requested"], "venice_native")
        self.assertEqual(transport["transport_used"], "openai_compat")
        self.assertTrue(transport["fallback_used"])
        self.assertEqual(len(response.data), 1)

    def test_generate_images_with_transport_auto_prefers_venice_native(self) -> None:
        os.environ.pop("PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT", None)
        os.environ["JARVINA_BASE_URL"] = "https://127.0.0.1/api/v1"
        os.environ["PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL"] = "true"
        client_module._is_venice_base_url = lambda _url: True

        def _fake_post(url: str, json: dict, headers: dict, timeout: int):
            self.assertTrue(url.endswith("/api/v1/image/generate"))
            self.assertIn("Authorization", headers)
            self.assertGreater(timeout, 0)
            self.assertEqual(json["model"], "venice-sd35")
            return _FakePostResponse(payload={"images": ["ZmFrZQ=="]})

        client_module.requests.post = _fake_post

        response, transport = client_module.generate_images_with_transport(
            {"model": "venice-sd35", "prompt": "test", "size": "1024x1024"},
            openai_client=_FakeOpenAIClient(),
        )
        self.assertEqual(transport["transport_requested"], "auto")
        self.assertEqual(transport["transport_used"], "venice_native")
        self.assertFalse(transport["fallback_used"])
        self.assertEqual(response.data[0].b64_json, "ZmFrZQ==")

    def test_openai_compat_drops_extra_body_for_venice_base(self) -> None:
        os.environ["PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT"] = "openai_compat"
        os.environ["JARVINA_BASE_URL"] = "https://api.venice.ai/api/v1"
        response, transport = client_module.generate_images_with_transport(
            {
                "model": "venice-sd35",
                "prompt": "test",
                "size": "1024x1024",
                "extra_body": {"cfg_scale": 7.5, "steps": 8},
            },
            openai_client=_FakeOpenAIClient(),
        )
        self.assertEqual(transport["transport_used"], "openai_compat")
        self.assertEqual(sorted(transport["compat_dropped_keys"]), ["cfg_scale", "steps"])
        self.assertNotIn("extra_body", _FakeOpenAIClient.last_kwargs or {})
        self.assertEqual(len(response.data), 1)

    def test_list_provider_image_styles_success(self) -> None:
        os.environ["JARVINA_BASE_URL"] = "https://127.0.0.1/api/v1"
        os.environ["PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL"] = "true"

        def _fake_get(url: str, headers: dict, timeout: int):
            self.assertTrue(url.endswith("/api/v1/image/styles"))
            self.assertIn("Authorization", headers)
            self.assertGreater(timeout, 0)
            return _FakeGetJsonResponse(
                payload={
                    "data": [
                        {"id": "3d-model", "name": "3D Model", "description": "desc"},
                        "anime",
                    ]
                }
            )

        client_module.requests.get = _fake_get
        styles = client_module.list_provider_image_styles()
        self.assertEqual(len(styles), 2)
        self.assertEqual(styles[0]["id"], "3d-model")
        self.assertEqual(styles[1]["id"], "anime")


if __name__ == "__main__":
    unittest.main()
