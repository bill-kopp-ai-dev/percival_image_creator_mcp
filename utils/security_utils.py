from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any
import re


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_BEARER_TOKEN_RE = re.compile(r"(?i)\b(bearer\s+)([a-z0-9._\-~+/]+=*)")
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*([^\s,;]+)"
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{10,}\b")

_PROMPT_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "override_instructions",
        re.compile(
            r"(?is)\b(ignore|disregard|forget|override)\b.{0,80}\b(previous|prior|above|all)\b.{0,80}\b"
            r"(instruction|instructions|prompt|message|messages)\b"
        ),
        "[redacted:override_instructions]",
    ),
    (
        "system_prompt_reference",
        re.compile(r"(?is)\b(system prompt|developer message|hidden prompt)\b"),
        "[redacted:system_prompt_reference]",
    ),
    (
        "override_rules_instruction",
        re.compile(r"(?is)\boverride\b.{0,40}\b(all|any)\b.{0,40}\b(rule|rules|policy|policies)\b"),
        "[redacted:override_rules_instruction]",
    ),
    (
        "tool_invocation_instruction",
        re.compile(r"(?is)\b(call|invoke|use)\b.{0,40}\b(tool|function)\b"),
        "[redacted:tool_invocation_instruction]",
    ),
    (
        "role_tag_injection",
        re.compile(r"(?is)<\s*/?\s*(system|assistant|developer|tool)\s*>"),
        "[redacted:role_tag]",
    ),
    (
        "secret_exfiltration_instruction",
        re.compile(
            r"(?is)\b(exfiltrate|leak|dump|print)\b.{0,80}\b(secret|credential|token|api key|password)\b"
        ),
        "[redacted:secret_exfiltration_instruction]",
    ),
)

_SECURITY_LOCK = Lock()
_SECURITY_COUNTERS: Counter[str] = Counter()
_RECENT_SECURITY_EVENTS: deque[dict[str, Any]] = deque(maxlen=100)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_detail_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_sensitive_text(value, max_len=600)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def record_security_event(event: str, details: dict[str, Any] | None = None) -> None:
    event_name = (event or "").strip() or "unknown_event"
    with _SECURITY_LOCK:
        _SECURITY_COUNTERS[event_name] += 1
        _RECENT_SECURITY_EVENTS.append(
            {
                "event": event_name,
                "timestamp": _utc_now_iso(),
                "details": {k: _safe_detail_value(v) for k, v in (details or {}).items()},
            }
        )


def get_security_metrics_snapshot() -> dict[str, Any]:
    with _SECURITY_LOCK:
        counters = dict(_SECURITY_COUNTERS)
        recent = list(_RECENT_SECURITY_EVENTS)
    return {
        "counters": counters,
        "recent_events": recent,
        "total_events": int(sum(counters.values())),
    }


def reset_security_metrics_for_tests() -> None:
    with _SECURITY_LOCK:
        _SECURITY_COUNTERS.clear()
        _RECENT_SECURITY_EVENTS.clear()


def clear_security_metrics() -> dict[str, int]:
    with _SECURITY_LOCK:
        counters_total = int(sum(_SECURITY_COUNTERS.values()))
        events_total = len(_RECENT_SECURITY_EVENTS)
        _SECURITY_COUNTERS.clear()
        _RECENT_SECURITY_EVENTS.clear()
    return {
        "cleared_counters_total": counters_total,
        "cleared_recent_events_total": events_total,
    }


def redact_sensitive_text(text: str, max_len: int = 1200) -> str:
    value = _CONTROL_CHAR_RE.sub(" ", str(text))
    value = _BEARER_TOKEN_RE.sub(r"\1[REDACTED]", value)
    value = _ASSIGNMENT_SECRET_RE.sub(r"\1=[REDACTED]", value)
    value = _OPENAI_KEY_RE.sub("[REDACTED_OPENAI_KEY]", value)
    if len(value) > max_len:
        return value[:max_len] + "...[truncated]"
    return value


def redact_sensitive_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: redact_sensitive_structure(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_sensitive_structure(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_structure(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def sanitize_untrusted_text(text: str, max_len: int = 8000) -> dict[str, Any]:
    original = str(text)
    sanitized = _CONTROL_CHAR_RE.sub(" ", original)
    findings: list[str] = []

    for finding_name, pattern, replacement in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(sanitized):
            findings.append(finding_name)
            sanitized = pattern.sub(replacement, sanitized)

    truncated = False
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "...[truncated]"
        truncated = True

    return {
        "text": sanitized,
        "findings": sorted(set(findings)),
        "truncated": truncated,
        "modified": sanitized != original,
    }
