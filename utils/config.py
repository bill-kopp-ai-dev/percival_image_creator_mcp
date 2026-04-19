import os
from typing import Optional

def get_env_bool(var_name: str, default: bool = False) -> bool:
    """Retrieve an environment variable as a boolean."""
    raw = os.getenv(var_name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default

def get_env_int(var_name: str, default: int, minimum: int = 1) -> int:
    """Retrieve an environment variable as an integer."""
    raw = os.getenv(var_name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except Exception:
        return default
    return max(minimum, value)

def get_env_str(var_name: str, default: str = "") -> str:
    """Retrieve an environment variable as a stripped string."""
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.strip()
