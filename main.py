import argparse
import asyncio
import hmac
import json
import os
from ipaddress import ip_address
from typing import Sequence

from server import configure_runtime_settings, mcp
from utils.nanobot_profile import CONTRACT_VERSION, SERVER_NAME, build_nanobot_profile
from utils.security_utils import record_security_event

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import uvicorn

# Import tools for registration side effects.
import tools.image_description_tools  # noqa: F401
import tools.image_generation_tools  # noqa: F401

DEFAULT_AUTH_TOKEN_ENV_VAR = "PERCIVAL_IMAGE_MCP_AUTH_TOKEN"


def _env_bool(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _is_loopback_host(host: str) -> bool:
    normalized_host = host.strip().lower()
    if normalized_host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ip_address(normalized_host).is_loopback
    except ValueError:
        return False


class BearerTokenAuthMiddleware(BaseHTTPMiddleware):
    """Require bearer auth for HTTP transports when token is configured."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self._token = token.strip()

    async def dispatch(self, request: Request, call_next):
        authorization = request.headers.get("authorization", "")
        header_token = request.headers.get("x-mcp-auth-token", "")

        provided_token = ""
        if authorization.lower().startswith("bearer "):
            provided_token = authorization[7:].strip()
        elif header_token:
            provided_token = header_token.strip()

        if not provided_token or not hmac.compare_digest(provided_token, self._token):
            record_security_event(
                "auth_rejected",
                {
                    "path": request.url.path,
                    "method": request.method,
                    "client": request.client.host if request.client else "unknown",
                },
            )
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


def _validate_http_runtime_security(
    *,
    mode: str,
    host: str,
    allow_remote_http: bool,
    auth_token: str | None,
    auth_token_env: str,
) -> None:
    if mode not in {"sse", "streamable-http"}:
        return

    loopback_host = _is_loopback_host(host)
    if not loopback_host and not allow_remote_http:
        record_security_event(
            "remote_bind_blocked",
            {"mode": mode, "host": host, "reason": "remote_http_not_allowed"},
        )
        raise ValueError(
            "Refusing to bind HTTP transport to a non-loopback host without --allow-remote-http."
        )

    if not loopback_host and not auth_token:
        record_security_event(
            "remote_bind_blocked",
            {"mode": mode, "host": host, "reason": "missing_auth_token"},
        )
        raise ValueError(
            "Remote HTTP mode requires authentication. "
            f"Set {auth_token_env} (or change token env var with --auth-token-env)."
        )

    if loopback_host and not auth_token:
        record_security_event(
            "loopback_http_without_auth",
            {"mode": mode, "host": host},
        )
        print(
            "[security-warning] HTTP mode on loopback without authentication token. "
            "Use only for local development."
        )


def _create_http_transport_app(
    *,
    mode: str,
    mount_path: str,
    auth_token: str | None,
) -> Starlette:
    if mode == "sse":
        app = mcp.sse_app(mount_path=mount_path)
    elif mode == "streamable-http":
        app = mcp.streamable_http_app()
    else:
        raise ValueError(f"Unsupported HTTP mode: {mode}")

    if auth_token:
        app.add_middleware(BearerTokenAuthMiddleware, token=auth_token)

    return app


async def _run_http_transport(
    *,
    mode: str,
    host: str,
    port: int,
    log_level: str,
    mount_path: str,
    auth_token: str | None,
) -> None:
    app = _create_http_transport_app(
        mode=mode,
        mount_path=mount_path,
        auth_token=auth_token,
    )
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Percival Image Creator MCP Server")
    parser.add_argument(
        "--mode",
        choices=("stdio", "sse", "streamable-http"),
        default=os.getenv("PERCIVAL_IMAGE_MCP_MODE", "stdio"),
        help="MCP transport mode (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("PERCIVAL_IMAGE_MCP_HOST", "127.0.0.1"),
        help="Host for HTTP-based modes (sse/streamable-http).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PERCIVAL_IMAGE_MCP_PORT", "8000")),
        help="Port for HTTP-based modes (sse/streamable-http).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=os.getenv("PERCIVAL_IMAGE_MCP_LOG_LEVEL", "INFO").upper(),
        help="Runtime log level for FastMCP HTTP transports.",
    )
    parser.add_argument(
        "--mount-path",
        default=os.getenv("PERCIVAL_IMAGE_MCP_MOUNT_PATH", "/"),
        help="Mount path when running in SSE mode.",
    )
    parser.add_argument(
        "--json-response",
        action="store_true",
        default=_env_bool("PERCIVAL_IMAGE_MCP_JSON_RESPONSE", False),
        help="Enable JSON response mode in FastMCP HTTP transport.",
    )
    parser.add_argument(
        "--stateless-http",
        action="store_true",
        default=_env_bool("PERCIVAL_IMAGE_MCP_STATELESS_HTTP", False),
        help="Enable stateless HTTP mode (streamable-http).",
    )
    parser.add_argument(
        "--print-profile",
        action="store_true",
        help="Print nanobot integration profile JSON and exit.",
    )
    parser.add_argument(
        "--allow-remote-http",
        action="store_true",
        default=_env_bool("PERCIVAL_IMAGE_MCP_ALLOW_REMOTE_HTTP", False),
        help="Allow HTTP transports to bind on non-loopback hosts.",
    )
    parser.add_argument(
        "--auth-token-env",
        default=os.getenv("PERCIVAL_IMAGE_MCP_AUTH_TOKEN_ENV", DEFAULT_AUTH_TOKEN_ENV_VAR),
        help="Environment variable containing bearer token for HTTP transport authentication.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.port < 1 or args.port > 65535:
        raise SystemExit("Invalid --port. Must be between 1 and 65535.")

    if args.print_profile:
        payload = {
            "server": SERVER_NAME,
            "contract_version": CONTRACT_VERSION,
            "profile": build_nanobot_profile(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    auth_token_raw = os.getenv(args.auth_token_env, "").strip()
    auth_token = auth_token_raw or None

    try:
        _validate_http_runtime_security(
            mode=args.mode,
            host=args.host,
            allow_remote_http=args.allow_remote_http,
            auth_token=auth_token,
            auth_token_env=args.auth_token_env,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    runtime_info = configure_runtime_settings(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        json_response=args.json_response,
        stateless_http=args.stateless_http,
        mount_path=args.mount_path,
    )
    print(
        "[mcp-startup] "
        f"server={runtime_info['server']} "
        f"mode={args.mode} "
        f"host={runtime_info['host']} "
        f"port={runtime_info['port']} "
        f"log_level={runtime_info['log_level']} "
        f"contract={CONTRACT_VERSION} "
        f"auth_enabled={bool(auth_token)}"
    )

    if args.mode == "stdio":
        mcp.run(transport=args.mode)
        return

    asyncio.run(
        _run_http_transport(
            mode=args.mode,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            mount_path=args.mount_path,
            auth_token=auth_token,
        )
    )


if __name__ == "__main__":
    main()
