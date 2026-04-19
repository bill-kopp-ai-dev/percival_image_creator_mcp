# 🎨 Percival — Image MCP Server

**Percival** is a provider-agnostic [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for AI image generation and analysis. It was designed for native compatibility with any OpenAI-compatible provider — including **Venice.ai**, **OpenAI**, and others — and integrates seamlessly with the [nanobot](https://github.com/HKUDS/nanobot) autonomous agent ecosystem as part of [percival.OS](https://github.com/bill-kopp-ai-dev/percival.OS_Dev).

---

## 🙏 Credits & Original Repository

This project is a refactored fork of the excellent **[ai-image-mcp](https://github.com/kareemaly/ai-image-mcp)**, originally created by **Kareem Aly** (`kareemaly`).

The original work's robust MCP tool architecture, `uv`-based dependency management, and SHA-256/MD5 image analysis cache system are all from the upstream project. Our refactoring focused exclusively on removing vendor lock-in, expanding provider flexibility, and optimizing tool descriptions for autonomous agent orchestration.

---

## 🛠️ What Changed? (Refactoring Details)

The following architectural changes were made to transform `ai-image-mcp` into **Percival**:

### 1. Fully Asynchronous Architecture (Performance)

The original project used synchronous I/O and blocking requests, which could degrade MCP server responsiveness during large generations.

- **Change:** Migrated the entire server to an async-first model. Core transport now uses `httpx.AsyncClient` and `AsyncOpenAI`.
- **Change:** All tools (generation, editing, vision, list, metadata) are now `async def` and non-blocking.
- **Benefit:** Full compatibility with FastMCP's async runtime and significantly higher throughput for concurrent requests.

### 2. Decoupled OpenAI Client (Provider-Agnostic)

- **Change:** A centralized async client module (`utils/client.py`) was implemented, utilizing Pydantic models for response validation. It targets variables like `JARVINA_BASE_URL` and `JARVINA_API_KEY`.
- **Benefit:** Robust, type-safe communication with any OpenAI-compatible provider (e.g. Venice.ai).

### 3. Modernized Model Catalog (Schema 3.0)

- **Change:** Updated the local model catalog (`image_models.json`) to schema version `3.0` and optimized catalog utilities (`utils/model_catalog.py`) for efficient normalization.
- **Benefit:** Richer metadata support for specialized tasks like vector clear generation and 4K upscaling.

### 4. Flexible & Agnostic Sandbox (User Workspace)

- **Change:** Centralized path validation in `utils/path_utils.py` and implemented a broad, language-agnostic sandbox policy.
- **Change:** The user's **Home directory** (`Path.home()`) is now an allowed root by default, ensuring immediate compatibility with Nanobot workspaces across different OS languages (e.g. Portuguese, English).
- **Benefit:** Eliminates "outside allowed roots" errors while maintaining security guardrails against arbitrary system escapes.

### 5. Centralized Configuration & Standardized Logging

- **Change:** Replaced all scattered `os.getenv` and `print` statements with a unified `utils/config.py` module and standard Python `logging`.
- **Benefit:** Improved observability and easier deployment without interfering with MCP's `stdio` protocol.

### 6. Venice Payload Builder (Deterministic Parameter Resolution)

The original `generate_image` function did not support advanced parameters from non-OpenAI providers.

- **Change:** The tool now builds and validates provider-specific payloads via a dedicated Venice payload layer, then forwards to OpenAI-compatible transport using `extra_body`.
- **Resolution Order:** explicit tool parameters > runtime override JSON > model-card `recommended_api_params` > server defaults.
- **Benefit:** Full utilization of Venice.ai's extended image generation capabilities with deterministic parameter precedence.

### 5. Native Venice Transport Hardening (Compatibility + Self-Healing)

- **Change:** Native requests to `POST /image/generate` now avoid sending OpenAI-style `size` and derive `width`/`height` from it when applicable.
- **Change:** Added bounded retry logic for native mode when the provider returns `unrecognized_keys`; rejected keys are pruned and retried automatically.
- **Change:** Transport metadata now exposes diagnostic fields (`native_dropped_keys`, `compat_dropped_keys`, `fallback_reason`) for agent observability.
- **Benefit:** Lower fallback frequency, fewer 400 errors in production, and clearer diagnostics for nanobot tool reasoning.

### 6. Provider-Aware Cache

The original cache system generated hashes based only on prompt, model, and image size.

- **Change:** The `_get_cache_key()` method now includes the domain of `JARVINA_BASE_URL` in the hash computation.
- **Benefit:** Prevents cache collisions across providers. The same prompt sent to OpenAI vs Venice.ai now produces separate, independent cache entries.

### 7. Agent-Optimized Docstrings

- **Change:** Tool descriptions were rewritten for autonomous agent clarity.
- **Benefit:** Agent orchestrators now understand precisely *when* to query the model list and *how* to pass provider-specific parameters, reducing hallucinated function calls.

### 8. Vision Model Configuration

- **Change:** A dedicated `JARVINA_VISION_MODEL` environment variable was added for the image analysis (vision) model, separate from the generation model.
- **Benefit:** Allows independent configuration of generation vs. analysis models (e.g. using a multimodal model like `qwen-2.5-vl` for vision tasks).

---

## 🔌 MCP Tools

### Image Generation

| Tool | Description |
|---|---|
| `recommend_model_for_intent` | Rank best-fit models for a human goal using model cards (intent, cost, quality, speed, availability) |
| `list_model_cards` | List structured model cards from local JSON catalog (recommended first step) |
| `get_model_card` | Get one model card with capabilities, tiers, and pricing metadata |
| `list_image_styles` | List available `style_preset` entries directly from provider (`/image/styles`) |
| `verify_model_availability` | Confirm selected model is still active in Venice before execution |
| `get_nanobot_profile` | Return machine-readable integration profile and recommended workflow for nanobot |
| `list_available_models` | List provider models directly (online, with short cache) |
| `generate_image` | Generate an image from a text prompt with optional provider-specific parameters |
| `edit_image` | Edit an existing image using image-edit capable models |
| `list_generated_images` | List all generated images in a directory with metadata |

> **Note:** `create_image_variations` remains disabled until provider compatibility is validated.

### Image Analysis (Vision)

| Tool | Description |
|---|---|
| `describe_image` | Generate a detailed description of an image using the vision model |
| `analyze_image_content` | Targeted analysis by type: `general`, `objects`, `text`, `colors`, `composition`, `emotions` |
| `compare_images` | Compare two images and highlight similarities and differences |
| `get_image_metadata` | Get technical metadata (dimensions, format, file size, EXIF) without an API call |

### Cache Management

| Tool | Description |
|---|---|
| `get_cache_info` | View cache statistics (number of files, total size) |
| `clear_image_cache` | Remove all cached analysis results |
| `get_security_metrics` | Inspect in-memory security counters/events (auth/path escape/prompt-injection detections) |
| `clear_security_metrics` | Reset in-memory security counters/events |
| `get_security_posture` | Show effective runtime security configuration and warnings |

---

## 🚀 Requirements

- Python 3.10+
- [`httpx`](https://github.com/encode/httpx) & [`pydantic`](https://github.com/pydantic/pydantic)
- [`uv`](https://github.com/astral-sh/uv) package manager
- Shared workspace virtual environment (`percival.OS_Dev/.venv`) for runtime and tests

---

## 📦 Installation

```bash
cd ~/Documentos/percival.OS_Dev
export UV_PROJECT_ENVIRONMENT=~/Documentos/percival.OS_Dev/.venv
uv sync --directory mcp_servers/percival_image_creator_mcp
```

> This server intentionally does **not** keep a local `.venv` inside `mcp_servers/percival_image_creator_mcp`.
> Version control is managed by the monorepo root (`percival.OS_Dev/.git`), with no nested Git metadata in this server directory.

---

## ⚙️ Configuration

Percival is configured via environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `JARVINA_API_KEY` | ✅ | — | API key for the provider. Falls back to `VENICE_API_KEY` or `OPENAI_API_KEY` |
| `JARVINA_BASE_URL` | ✅ | `https://api.openai.com/v1` | Base URL for the OpenAI-compatible API endpoint |
| `JARVINA_VISION_MODEL` | ❌ | `qwen-2.5-vl` | Model ID to use for vision/analysis tasks |
| `PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT` | ❌ | `auto` | Generation transport mode: `auto`, `openai_compat`, or `venice_native` |
| `PERCIVAL_IMAGE_MCP_IMAGE_TRANSPORT_FALLBACK` | ❌ | `true` | Allow fallback to `openai_compat` if native Venice transport fails |
| `PERCIVAL_IMAGE_MCP_VENICE_NATIVE_RETRIES` | ❌ | `1` | Number of native retry attempts after provider `unrecognized_keys` errors |
| `PERCIVAL_IMAGE_MCP_PROVIDER_TIMEOUT_SECONDS` | ❌ | `90` | Timeout for provider generation requests |
| `PERCIVAL_IMAGE_MCP_GENERATION_OVERRIDES_JSON` | ❌ | — | JSON object with runtime generation parameter overrides |
| `PERCIVAL_IMAGE_MCP_DEFAULT_OUTPUT_DIR` | ❌ | `~/Pictures` | Default output directory for generated/edited images |
| `PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS` | ❌ | Home + CWD | Comma-separated absolute roots allowed for `working_dir` |
| `PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX` | ❌ | `false` | Disable `working_dir` root containment (development only) |
| `PERCIVAL_IMAGE_MCP_AUTH_TOKEN` | ❌ | — | Bearer token used by HTTP auth middleware |
| `PERCIVAL_IMAGE_MCP_ALLOW_REMOTE_HTTP` | ❌ | `false` | Permit non-loopback HTTP bind (requires auth token) |
| `PERCIVAL_IMAGE_MCP_ALLOW_INSECURE_PROVIDER_URL` | ❌ | `false` | Allow non-HTTPS `JARVINA_BASE_URL` |
| `PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_PROVIDER_URL` | ❌ | `false` | Allow provider URL resolving to local/private IP |
| `PERCIVAL_IMAGE_MCP_ALLOWED_PROVIDER_HOSTS` | ❌ | — | Comma-separated provider host allowlist |
| `PERCIVAL_IMAGE_MCP_ALLOW_HTTP_DOWNLOADS` | ❌ | `false` | Allow `http://` download URLs |
| `PERCIVAL_IMAGE_MCP_ALLOW_PRIVATE_DOWNLOADS` | ❌ | `false` | Allow download URLs resolving to local/private IP |
| `PERCIVAL_IMAGE_MCP_ALLOWED_DOWNLOAD_HOSTS` | ❌ | — | Comma-separated host allowlist for download URLs |
| `PERCIVAL_IMAGE_MCP_DOWNLOAD_MAX_BYTES` | ❌ | `26214400` | Maximum bytes allowed when downloading image URL payloads |
| `PERCIVAL_IMAGE_MCP_MAX_PROMPT_CHARS` | ❌ | `4000` | Max characters accepted in generation/edit prompts |
| `PERCIVAL_IMAGE_MCP_MAX_NEGATIVE_PROMPT_CHARS` | ❌ | `2000` | Max characters accepted in `negative_prompt` |
| `PERCIVAL_IMAGE_MCP_MAX_FILENAME_PREFIX_CHARS` | ❌ | `80` | Max characters accepted in image filename prefix |
| `PERCIVAL_IMAGE_MCP_MAX_MODEL_ID_CHARS` | ❌ | `128` | Max length for `model`/`model_id` |
| `PERCIVAL_IMAGE_MCP_MAX_LIST_FILES` | ❌ | `200` | Max files returned by `list_generated_images` |
| `PERCIVAL_IMAGE_MCP_MAX_ANALYSIS_PROMPT_CHARS` | ❌ | `4000` | Max length for `describe_image.prompt` |
| `PERCIVAL_IMAGE_MCP_MAX_COMPARISON_FOCUS_CHARS` | ❌ | `200` | Max length for `compare_images.comparison_focus` |

---

## ▶️ Running

Canonical stdio runtime:

```bash
export UV_PROJECT_ENVIRONMENT=~/Documentos/percival.OS_Dev/.venv
uv run --no-sync --directory ~/Documentos/percival.OS_Dev/mcp_servers/percival_image_creator_mcp python main.py --mode stdio
```

HTTP transports:

```bash
export UV_PROJECT_ENVIRONMENT=~/Documentos/percival.OS_Dev/.venv
PERCIVAL_IMAGE_MCP_AUTH_TOKEN=change-me uv run --no-sync --directory ~/Documentos/percival.OS_Dev/mcp_servers/percival_image_creator_mcp python main.py --mode sse --host 127.0.0.1 --port 8000
PERCIVAL_IMAGE_MCP_AUTH_TOKEN=change-me uv run --no-sync --directory ~/Documentos/percival.OS_Dev/mcp_servers/percival_image_creator_mcp python main.py --mode streamable-http --host 127.0.0.1 --port 8000 --stateless-http
```

Print integration profile:

```bash
export UV_PROJECT_ENVIRONMENT=~/Documentos/percival.OS_Dev/.venv
uv run --no-sync --directory ~/Documentos/percival.OS_Dev/mcp_servers/percival_image_creator_mcp python main.py --print-profile
```

## ✅ Tests

```bash
export UV_PROJECT_ENVIRONMENT=~/Documentos/percival.OS_Dev/.venv
uv run --no-sync --directory ~/Documentos/percival.OS_Dev/mcp_servers/percival_image_creator_mcp pytest -q
```

---

## 🤖 Integrating with nanobot / Claude Desktop

Add the following entry to your agent's `config.json`:

```json
"percival-image": {
  "command": "uv",
  "args": [
    "run",
    "--no-sync",
    "--directory",
    "~/Documentos/percival.OS_Dev/mcp_servers/percival_image_creator_mcp",
    "python",
    "main.py",
    "--mode",
    "stdio"
  ],
  "enabledTools": [
    "recommend_model_for_intent",
    "list_model_cards",
    "get_model_card",
    "list_image_styles",
    "verify_model_availability",
    "get_security_metrics",
    "clear_security_metrics",
    "get_security_posture",
    "generate_image",
    "edit_image",
    "describe_image",
    "analyze_image_content",
    "compare_images",
    "get_image_metadata",
    "list_generated_images"
  ],
  "toolTimeout": 45,
  "env": {
    "UV_PROJECT_ENVIRONMENT": "~/Documentos/percival.OS_Dev/.venv",
    "JARVINA_API_KEY": "your-api-key-here",
    "JARVINA_BASE_URL": "https://api.venice.ai/api/v1",
    "JARVINA_VISION_MODEL": "qwen-2.5-vl"
  }
}
```

### Example Usage

```
User: Generate a 16:9 cyberpunk cityscape image.

Agent: [calls list_model_cards(task_type="text_to_image")]
       [or calls recommend_model_for_intent(task_type="text_to_image", intent="<goal>")]
       [selects model by model card metadata]
       [calls list_image_styles() when LoRA style preset is needed]
       [calls verify_model_availability(model_id="venice-sd35", task_type="text_to_image")]
       [calls generate_image(model="venice-sd35", aspect_ratio="16:9", negative_prompt="blur, low quality")]

User: Edit that image to remove a background element.

Agent: [calls list_model_cards(task_type="image_edit")]
       [calls verify_model_availability(model_id="qwen-image-2-edit", task_type="image_edit")]
       [calls edit_image(image_path="...", model="qwen-image-2-edit", prompt="remove background element")]

User: Describe what's in that image.

Agent: [calls describe_image with the generated file path]
       [returns cached result on subsequent identical requests]
```

---

## 📁 Project Structure

```
percival_image_creator_mcp/
├── main.py                          # Entry point
├── server.py                        # MCP server instance
├── pyproject.toml                   # Project metadata & dependencies
├── image_models.json                # Structured model cards (schema v2.1)
├── tests/                           # Unit and mocked integration tests
│   └── fixtures/prompt_injection_corpus.json  # Security regression corpus
├── tools/
│   ├── image_generation_tools.py    # catalog tools, availability check, generate/edit tools
│   └── image_description_tools.py  # describe_image, analyze_image_content, compare_images, metadata, cache tools
└── utils/
    ├── model_catalog.py             # Catalog load/validation/query layer
    ├── client.py                    # Provider-agnostic OpenAI client (Percival singleton)
    ├── nanobot_profile.py           # Machine-readable nanobot integration profile + contract version
    ├── cache_utils.py               # SHA-256 + MD5 provider-aware image analysis cache
    ├── path_utils.py                # Image path validation utilities
    └── security_utils.py            # Prompt/input sanitization + security telemetry
```

### Nanobot Contract (Recommended Orchestration)

1. `list_model_cards(task_type=...)`
2. `get_model_card(model_id)` (optional deep inspection)
3. `verify_model_availability(model_id, task_type=...)`
4. `generate_image(...)` or `edit_image(...)`

`generate_image` and `edit_image` run strict preflight checks by default (`strict_model_check=True`).

### Stable Tool Output Contract

Tools return a compact JSON envelope (generation + vision + cache):

```json
{
  "ok": true,
  "data": {
    "...": "...",
    "transport": {
      "transport_requested": "auto",
      "transport_used": "venice_native",
      "fallback_used": false,
      "native_dropped_keys": []
    }
  },
  "meta": {
    "server": "percival-image-creator-mcp",
    "contract_version": "2026-03-s3",
    "request_id": "img-...",
    "timestamp": "2026-03-29T21:31:00Z",
    "tool": "generate_image"
  },
  "legacy_text": "human-readable compatibility text"
}
```

Error shape:

```json
{
  "ok": false,
  "error": "message",
  "code": "error_code",
  "details": { "...": "..." },
  "meta": {
    "server": "percival-image-creator-mcp",
    "contract_version": "2026-03-s3",
    "request_id": "img-...",
    "timestamp": "2026-03-29T21:31:00Z",
    "tool": "generate_image"
  },
  "legacy_text": "Error: ..."
}
```

For payload control in large catalogs, `list_model_cards` supports:
- `limit`
- `offset`
- `fields` (comma-separated)

### Security Hardening (PR-SEC-3 / PR-SEC-4 / PR-SEC-5 / PR-SEC-6 / PR-SEC-7 / PR-SEC-8)

- Vision outputs and EXIF text are treated as **untrusted external data**.
- Prompt-injection patterns in model/metadata output are sanitized and flagged in structured `security` fields.
- Security telemetry is recorded in-memory:
  - `auth_rejected`
  - `remote_bind_blocked`
  - `loopback_http_without_auth`
  - `path_escape_blocked`
  - `prompt_injection_detected`
- Use `get_security_metrics` for runtime inspection.
- Use `clear_security_metrics` to reset counters/events between incident analysis windows.
- Use `get_security_posture` to inspect effective runtime policy + hardening warnings.
- Outbound URL guardrails:
  - `JARVINA_BASE_URL` is validated (scheme/host/private-network policy).
  - provider-returned image URLs are validated before download (SSRF controls).
  - download size is capped via `PERCIVAL_IMAGE_MCP_DOWNLOAD_MAX_BYTES`.
- Cache hardening:
  - cache dir/file permissions forced to `0700`/`0600` where supported;
  - cache writes use atomic replace;
  - cache symlink/path-escape attempts are blocked and audited.
- Input/abuse guardrails:
  - prompt/model/prefix lengths are bounded by env-configurable limits;
  - filename prefix and model IDs are validated by safe character policy;
  - very large `list_generated_images` result sets are truncated by policy.

---

## 📖 Attribution

This project is built upon the work of:

- **[kareemaly/ai-image-mcp](https://github.com/kareemaly/ai-image-mcp)** — The direct upstream project providing the original MCP tool architecture and cache system.

---

## 📄 License

This project maintains the MIT License from the original repository. See [LICENSE](LICENSE) for details.
