# 🤖 Percival Image Creator - percival.OS MCP

**Version 0.0.2**

[![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)]()
[![MCP](https://img.shields.io/badge/mcp-server-blue.svg)]()
[![percival.OS](https://img.shields.io/badge/percival.OS-ecosystem-orange.svg)](https://github.com/bill-kopp-ai-dev/percival.OS)

## 📋 Description
**Percival Image Creator** is a provider-agnostic MCP server for AI image generation and analysis. It was designed for native compatibility with any OpenAI-compatible provider (including Venice.ai) and seamless integration with Nanobot.

This server is part of the **percival.OS** ecosystem, a Personal Agentic Operating System designed for autonomy, security, and absolute privacy.

---

## 🛡️ percival.OS Principles
Like all components of `percival.OS`, this MCP server strictly follows our core principles:

- **Privacy & Flexibility**: You can choose your preferred image generation provider, maintaining sovereignty over your keys and data.
- **Data Sovereignty**: Generated and analyzed images are stored locally in your infrastructure.
- **Hardened Security**: We implement a strict sandbox for working directories, prompt-injection protection in vision metadata, and real-time security telemetry.
- **Transparency**: Based on the `ai-image-mcp` project by Kareem Aly, but extensively refactored to be asynchronous, agnostic, and secure.

---

## 🚀 Features & Tools

### Image Generation
- `image_generate`: Generate an image from a text prompt.
- `image_edit`: Edit an existing image using image-edit capable models.
- `image_upscale`: Upscale an image to higher resolution.
- `image_list_models`: List available models and their capabilities.
- `image_recommend_model`: Rank best-fit models for a specific task.

### Image Analysis (Vision)
- `image_describe`: Generate a detailed description of an image.
- `image_analyze`: Targeted analysis by type (objects, text, colors, etc.).
- `image_compare`: Compare two images and highlight differences.
- `image_get_metadata`: Get technical metadata (EXIF, dimensions) locally.

### Management & System
- `image_get_status`: Return server operational status.
- `image_list_recent`: List recently generated images.
- `image_get_security_metrics`: Inspect in-memory security counters/events.

---

## ⚙️ Configuration in percival.OS (Nanobot)
Add the following configuration to your `~/.nanobot/config.json`:

```json
{
  "tools": {
    "mcpServers": {
      "percival-image": {
        "command": "uv",
        "args": [
          "run",
          "--no-sync",
          "--directory",
          "/path/to/percival_image_creator_mcp",
          "python",
          "main.py",
          "--mode",
          "stdio"
        ],
        "env": {
          "JARVINA_API_KEY": "YOUR_API_KEY",
          "JARVINA_BASE_URL": "https://api.venice.ai/api/v1",
          "JARVINA_VISION_MODEL": "qwen-2.5-vl"
        }
      }
    }
  }
}
```

---

## 🛠️ Development & Testing
This project uses `uv` for dependency management in the shared environment.

```bash
# Manual execution in stdio
uv run --no-sync --directory ./mcp_servers/percival_image_creator_mcp python main.py --mode stdio

# Run tests
uv run --no-sync --directory ./mcp_servers/percival_image_creator_mcp pytest -q
```

---

## 📚 About the Project
This server is an integral module of the **percival.OS** project. It expands Nanobot's creativity by allowing the agent to "see" and "create" visual content.

- **Main Repository**: [https://github.com/bill-kopp-ai-dev/percival.OS](https://github.com/bill-kopp-ai-dev/percival.OS)
- **License**: MIT

---
*Developed with ❤️ by the percival.OS Team*
