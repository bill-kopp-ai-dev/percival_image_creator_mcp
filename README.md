# 🎨 Jarvina — Image MCP Server

**Jarvina** is a provider-agnostic [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for AI image generation and analysis. It was designed for native compatibility with any OpenAI-compatible provider — including **Venice.ai**, **OpenAI**, and others — and integrates seamlessly with the [nanobot](https://github.com/HKUDS/nanobot) autonomous agent ecosystem as part of [percival.OS](https://github.com/bill-kopp-ai-dev/percival.OS_Dev).

---

## 🙏 Credits & Original Repository

This project is a refactored fork of the excellent **[ai-image-mcp](https://github.com/kareemaly/ai-image-mcp)**, originally created by **Kareem Aly** (`kareemaly`).

The original work's robust MCP tool architecture, `uv`-based dependency management, and SHA-256/MD5 image analysis cache system are all from the upstream project. Our refactoring focused exclusively on removing vendor lock-in, expanding provider flexibility, and optimizing tool descriptions for autonomous agent orchestration.

---

## 🛠️ What Changed? (Refactoring Details)

The following architectural changes were made to transform `ai-image-mcp` into **Jarvina**:

### 1. Decoupled OpenAI Client (Provider-Agnostic)

The original project required `OPENAI_API_KEY` and pointed strictly to OpenAI's servers.

- **Change:** A centralized client module (`utils/client.py`) was implemented, reading custom environment variables `JARVINA_BASE_URL` and `JARVINA_API_KEY`. It also supports graceful fallbacks to `VENICE_API_KEY` and `OPENAI_API_KEY` to ease migration.
- **Benefit:** The server can now target any OpenAI-compatible provider (e.g. Venice.ai) without modifying any tool code.

### 2. Dynamic Model Discovery

The original functions hardcoded model validation (e.g. `if model not in ["dall-e-2", "dall-e-3"]`).

- **Change:** Restrictive validations were removed and a new `list_available_models` tool was created.
- **Benefit:** The LLM agent can now dynamically query the provider's API to discover available image models before attempting generation.

### 3. Flexible Payload via `extra_body`

The original `generate_image` function did not support advanced parameters from non-OpenAI providers.

- **Change:** The tool now accepts provider-specific parameters via the OpenAI SDK's `extra_body` argument, passing configuration options such as `aspect_ratio`, `resolution`, `cfg_scale`, and `negative_prompt`.
- **Benefit:** Full utilization of Venice.ai's extended image generation capabilities.

### 4. Provider-Aware Cache

The original cache system generated hashes based only on prompt, model, and image size.

- **Change:** The `_get_cache_key()` method now includes the domain of `JARVINA_BASE_URL` in the hash computation.
- **Benefit:** Prevents cache collisions across providers. The same prompt sent to OpenAI vs Venice.ai now produces separate, independent cache entries.

### 5. Agent-Optimized Docstrings

- **Change:** Tool descriptions were rewritten for autonomous agent clarity.
- **Benefit:** Agent orchestrators now understand precisely *when* to query the model list and *how* to pass provider-specific parameters, reducing hallucinated function calls.

### 6. Vision Model Configuration

- **Change:** A dedicated `JARVINA_VISION_MODEL` environment variable was added for the image analysis (vision) model, separate from the generation model.
- **Benefit:** Allows independent configuration of generation vs. analysis models (e.g. using a multimodal model like `qwen-2.5-vl` for vision tasks).

---

## 🔌 MCP Tools

### Image Generation

| Tool | Description |
|---|---|
| `list_available_models` | Dynamically lists all image models available from the configured provider |
| `generate_image` | Generate an image from a text prompt with optional provider-specific parameters |
| `list_generated_images` | List all generated images in a directory with metadata |

> **Note:** `edit_image` and `create_image_variations` are currently disabled as they depend on OpenAI-specific APIs not universally supported by other providers.

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

---

## 🚀 Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager

---

## 📦 Installation

```bash
git clone https://github.com/bill-kopp-ai-dev/percival_image_creator_mcp.git
cd percival_image_creator_mcp
uv sync
```

---

## ⚙️ Configuration

Jarvina is configured via environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `JARVINA_API_KEY` | ✅ | — | API key for the provider. Falls back to `VENICE_API_KEY` or `OPENAI_API_KEY` |
| `JARVINA_BASE_URL` | ✅ | `https://api.openai.com/v1` | Base URL for the OpenAI-compatible API endpoint |
| `JARVINA_VISION_MODEL` | ❌ | `qwen-2.5-vl` | Model ID to use for vision/analysis tasks |

---

## ▶️ Running

```bash
uv run main.py
```

---

## 🤖 Integrating with nanobot / Claude Desktop

Add the following entry to your agent's `config.json`:

```json
"percival-image": {
  "command": "/path/to/.venv/bin/python",
  "args": ["/path/to/percival_image_creator_mcp/main.py"],
  "env": {
    "JARVINA_API_KEY": "your-api-key-here",
    "JARVINA_BASE_URL": "https://api.venice.ai/api/v1",
    "JARVINA_VISION_MODEL": "qwen-2.5-vl"
  }
}
```

### Example Usage

```
User: Generate a 16:9 cyberpunk cityscape image.

Agent: [calls list_available_models to discover available models]
       [calls generate_image with model="fluently-xl", aspect_ratio="16:9", negative_prompt="blur, low quality"]

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
├── tools/
│   ├── image_generation_tools.py    # generate_image, list_available_models, list_generated_images
│   └── image_description_tools.py  # describe_image, analyze_image_content, compare_images, metadata, cache tools
└── utils/
    ├── client.py                    # Provider-agnostic OpenAI client (Jarvina singleton)
    ├── cache_utils.py               # SHA-256 + MD5 provider-aware image analysis cache
    └── path_utils.py                # Image path validation utilities
```

---

## 📖 Attribution

This project is built upon the work of:

- **[kareemaly/ai-image-mcp](https://github.com/kareemaly/ai-image-mcp)** — The direct upstream project providing the original MCP tool architecture and cache system.

---

## 📄 License

This project maintains the MIT License from the original repository. See [LICENSE](LICENSE) for details.
