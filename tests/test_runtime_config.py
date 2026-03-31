import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

os.environ.setdefault("JARVINA_API_KEY", "test-key")
os.environ.setdefault("PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS", tempfile.gettempdir())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module  # noqa: E402
from server import configure_runtime_settings, mcp  # noqa: E402


class TestRuntimeConfig(unittest.TestCase):
    def test_configure_runtime_settings(self) -> None:
        info = configure_runtime_settings(
            host="127.0.0.1",
            port=8123,
            log_level="DEBUG",
            json_response=True,
            stateless_http=True,
            mount_path="/mcp",
        )
        self.assertEqual(info["host"], "127.0.0.1")
        self.assertEqual(info["port"], 8123)
        self.assertEqual(info["log_level"], "DEBUG")
        self.assertTrue(mcp.settings.json_response)
        self.assertTrue(mcp.settings.stateless_http)
        self.assertEqual(mcp.settings.mount_path, "/mcp")

    def test_main_print_profile(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            main_module.main(["--print-profile"])
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["server"], "percival-image-creator-mcp")
        self.assertEqual(payload["profile"]["server"], "percival-image-creator-mcp")


if __name__ == "__main__":
    unittest.main()
