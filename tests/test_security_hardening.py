import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("JARVINA_API_KEY", "test-key")
os.environ.setdefault("PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS", tempfile.gettempdir())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module  # noqa: E402
from utils.path_utils import validate_working_directory  # noqa: E402


class TestSecurityHardening(unittest.TestCase):
    def test_http_remote_requires_opt_in(self) -> None:
        with self.assertRaises(ValueError):
            main_module._validate_http_runtime_security(
                mode="sse",
                host="0.0.0.0",
                allow_remote_http=False,
                auth_token=None,
                auth_token_env="PERCIVAL_IMAGE_MCP_AUTH_TOKEN",
            )

    def test_http_remote_requires_auth_token(self) -> None:
        with self.assertRaises(ValueError):
            main_module._validate_http_runtime_security(
                mode="streamable-http",
                host="0.0.0.0",
                allow_remote_http=True,
                auth_token=None,
                auth_token_env="PERCIVAL_IMAGE_MCP_AUTH_TOKEN",
            )

    def test_http_remote_with_auth_and_opt_in_is_allowed(self) -> None:
        # Should not raise.
        main_module._validate_http_runtime_security(
            mode="streamable-http",
            host="0.0.0.0",
            allow_remote_http=True,
            auth_token="secret-token",
            auth_token_env="PERCIVAL_IMAGE_MCP_AUTH_TOKEN",
        )

    def test_working_dir_restricted_to_allowed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as allowed_root, tempfile.TemporaryDirectory() as outside_root:
            original = os.environ.get("PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS")
            os.environ["PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS"] = allowed_root
            os.environ.pop("PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX", None)
            try:
                inside = Path(allowed_root) / "project"
                inside.mkdir()
                resolved_inside, error_inside = validate_working_directory(str(inside))
                self.assertIsNotNone(resolved_inside)
                self.assertIsNone(error_inside)

                resolved_outside, error_outside = validate_working_directory(str(Path(outside_root)))
                self.assertIsNone(resolved_outside)
                self.assertIsNotNone(error_outside)
                self.assertIn("outside allowed roots", error_outside or "")
            finally:
                if original is None:
                    os.environ.pop("PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS", None)
                else:
                    os.environ["PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS"] = original

    def test_working_dir_sandbox_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as outside_root:
            original_disable = os.environ.get("PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX")
            try:
                os.environ["PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX"] = "true"
                resolved, error = validate_working_directory(str(Path(outside_root)))
                self.assertIsNotNone(resolved)
                self.assertIsNone(error)
            finally:
                if original_disable is None:
                    os.environ.pop("PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX", None)
                else:
                    os.environ["PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX"] = original_disable


if __name__ == "__main__":
    unittest.main()
