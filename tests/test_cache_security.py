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

from utils.cache_utils import ImageAnalysisCache  # noqa: E402
from utils.security_utils import (  # noqa: E402
    get_security_metrics_snapshot,
    reset_security_metrics_for_tests,
)


class TestCacheSecurity(unittest.TestCase):
    def setUp(self) -> None:
        reset_security_metrics_for_tests()

    def test_cache_permissions_and_roundtrip(self) -> None:
        subdir = f"cache_test_{os.getpid()}_{id(self)}"
        cache = ImageAnalysisCache(cache_subdir=subdir)
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "image.bin"
            img.write_bytes(b"abc123")

            cache.store_result(img, "describe", {"prompt": "x"}, "result-value")
            value = cache.get_cached_result(img, "describe", {"prompt": "x"})
            self.assertEqual(value, "result-value")

            if os.name != "nt":
                dir_mode = cache.cache_dir.stat().st_mode & 0o777
                self.assertEqual(dir_mode, 0o700)
                cache_key = cache._get_cache_key(img, "describe", {"prompt": "x"})  # noqa: SLF001
                cache_file = cache._get_cache_file_path(cache_key)  # noqa: SLF001
                file_mode = cache_file.stat().st_mode & 0o777
                self.assertEqual(file_mode, 0o600)

    def test_symlink_cache_file_is_blocked(self) -> None:
        if os.name == "nt":
            self.skipTest("symlink test is not portable on this platform")

        subdir = f"cache_test_link_{os.getpid()}_{id(self)}"
        cache = ImageAnalysisCache(cache_subdir=subdir)
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "image.bin"
            img.write_bytes(b"abc123")

            key = cache._get_cache_key(img, "describe", {"prompt": "x"})  # noqa: SLF001
            cache_file = cache._get_cache_file_path(key)  # noqa: SLF001
            target = Path(td) / "target.json"
            target.write_text("{}", encoding="utf-8")
            cache_file.symlink_to(target)

            result = cache.get_cached_result(img, "describe", {"prompt": "x"})
            self.assertIsNone(result)

            metrics = get_security_metrics_snapshot()
            self.assertGreater(metrics["counters"].get("cache_path_blocked", 0), 0)


if __name__ == "__main__":
    unittest.main()
