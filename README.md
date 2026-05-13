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

---

## 🚀 Features & Tools

### Image Generation & Modification
- `image_generate`: Generate high-quality images from text prompts (Default: `flux-2-pro`).
- `image_edit`: Modify existing images using specialized models (`qwen-edit`, `nano-banana-2-edit`).
- `image_upscale`: Increase image resolution using local or cloud-based upscalers.
- `image_list_models`: List the curated catalog of verified models and their costs.
- `image_recommend_model`: Get an agentic recommendation of the best model for a specific intent.
- `image_verify_model`: Verify if a specific model ID is available and active.

### Image Analysis & Metadata
- `image_describe`: Generate detailed, structured descriptions using Vision LLMs.
- `image_analyze_content`: Targeted analysis (object detection, OCR, color palette).
- `image_compare`: Pixel-level and semantic comparison between two images.
- `image_get_metadata`: Extract technical metadata (dimensions, EXIF, prompt history) locally.
- `image_list_recent`: List recently generated images in the working directory.

### Observability & Security
- `image_get_status`: Real-time security metrics and event counters.
- `image_get_system_info`: Detailed audit of runtime security posture and environment limits.
- `image_get_contract_info`: Machine-readable integration profile for Nanobot orchestration.
- `image_clear_status`: Reset security metrics for a new diagnostic window.

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
This project uses `uv` for dependency management.

```bash
# Manual execution in stdio
uv run python main.py --mode stdio

# Validation
python3 -m py_compile tools/image_generation_tools.py
```

---

## 📚 About the Project
This server is an integral module of the **percival.OS** project. It expands Nanobot's creativity by allowing the agent to "see" and "create" visual content.

- **Main Repository**: [https://github.com/bill-kopp-ai-dev/percival.OS](https://github.com/bill-kopp-ai-dev/percival.OS)
- **License**: MIT

---
*Developed with ❤️ by the percival.OS Team*
