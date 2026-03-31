from mcp.server.fastmcp import FastMCP

from utils.nanobot_profile import SERVER_NAME


mcp = FastMCP(SERVER_NAME)


def configure_runtime_settings(
    *,
    host: str,
    port: int,
    log_level: str,
    json_response: bool = False,
    stateless_http: bool = False,
    mount_path: str = "/",
) -> dict[str, object]:
    """
    Apply runtime FastMCP settings before `mcp.run(...)`.
    """
    mcp.settings.host = host
    mcp.settings.port = int(port)
    mcp.settings.log_level = log_level.upper()
    mcp.settings.json_response = bool(json_response)
    mcp.settings.stateless_http = bool(stateless_http)
    mcp.settings.mount_path = mount_path
    return {
        "server": SERVER_NAME,
        "host": mcp.settings.host,
        "port": mcp.settings.port,
        "log_level": mcp.settings.log_level,
        "json_response": mcp.settings.json_response,
        "stateless_http": mcp.settings.stateless_http,
        "mount_path": mcp.settings.mount_path,
    }
