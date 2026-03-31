import json
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

import tools.image_generation_tools as gen_tools  # noqa: E402
from utils.security_utils import (  # noqa: E402
    get_security_metrics_snapshot,
    record_security_event,
    reset_security_metrics_for_tests,
    sanitize_untrusted_text,
)


def _load_corpus() -> list[dict[str, object]]:
    fixture = Path(__file__).parent / "fixtures" / "prompt_injection_corpus.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


class TestPromptInjectionRegression(unittest.TestCase):
    def setUp(self) -> None:
        reset_security_metrics_for_tests()

    def test_prompt_injection_corpus_is_sanitized(self) -> None:
        for case in _load_corpus():
            payload = str(case["payload"])
            expected_absent = list(case["expected_absent"])
            sanitized = sanitize_untrusted_text(payload)
            rendered = sanitized["text"]
            for forbidden in expected_absent:
                self.assertNotIn(str(forbidden), rendered)

    def test_security_metrics_tool_returns_snapshot(self) -> None:
        record_security_event("custom_security_event_for_test", {"sample": "ok"})
        payload = json.loads(gen_tools.get_security_metrics())
        self.assertTrue(payload["ok"])
        metrics = payload["data"]["security_metrics"]
        self.assertEqual(metrics["counters"].get("custom_security_event_for_test", 0), 1)
        self.assertGreaterEqual(metrics["total_events"], 1)
        self.assertTrue(isinstance(metrics["recent_events"], list))

    def test_token_like_content_is_redacted_in_metrics(self) -> None:
        record_security_event(
            "token_redaction_test",
            {"raw": "Authorization: Bearer token-super-secret"},
        )
        metrics = get_security_metrics_snapshot()
        recent = metrics["recent_events"][-1]
        self.assertEqual(recent["event"], "token_redaction_test")
        rendered = str(recent["details"])
        self.assertIn("Bearer [REDACTED]", rendered)
        self.assertNotIn("token-super-secret", rendered)

    def test_clear_security_metrics_tool(self) -> None:
        record_security_event("event_before_clear", {"sample": "value"})
        metrics_before = get_security_metrics_snapshot()
        self.assertGreater(metrics_before["total_events"], 0)

        payload = json.loads(gen_tools.clear_security_metrics())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["operation"], "clear_security_metrics")

        metrics_after = get_security_metrics_snapshot()
        self.assertEqual(metrics_after["total_events"], 0)

    def test_security_posture_reports_warnings_for_insecure_flags(self) -> None:
        original = os.environ.get("PERCIVAL_IMAGE_MCP_ALLOW_HTTP_DOWNLOADS")
        os.environ["PERCIVAL_IMAGE_MCP_ALLOW_HTTP_DOWNLOADS"] = "true"
        try:
            payload = json.loads(gen_tools.get_security_posture())
            self.assertTrue(payload["ok"])
            warnings = payload["data"]["posture"]["warnings"]
            self.assertTrue(any("HTTP download URLs allowed" in warning for warning in warnings))
        finally:
            if original is None:
                os.environ.pop("PERCIVAL_IMAGE_MCP_ALLOW_HTTP_DOWNLOADS", None)
            else:
                os.environ["PERCIVAL_IMAGE_MCP_ALLOW_HTTP_DOWNLOADS"] = original


if __name__ == "__main__":
    unittest.main()
