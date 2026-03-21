from pathlib import Path
from typing import Optional
from server import mcp
from utils.client import jarvina_client as client
from utils.client import (
    save_base64_image,
    download_image_from_url
)

@mcp.tool()
def list_available_models() -> str:
    """
    Lista todos os modelos de geração de imagem disponíveis no provedor atual (ex: Venice.ai).
    O LLM DEVE chamar esta função antes de gerar uma imagem se não souber o nome exato do modelo.
    """
    try:
        models = client.models.list()
        model_names = [model.id for model in models.data]
        if not model_names:
            return "Nenhum modelo encontrado no provedor atual."
        return "Modelos disponíveis para uso:\n- " + "\n- ".join(model_names)
    except Exception as e:
        return f"Erro ao buscar modelos: {str(e)}"

@mcp.tool()
def generate_image(
    working_dir: str,
    prompt: str,
    model: str = "fluently-xl",
    size: str = "1024x1024",
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    output_dir: str = "generated_images",
    filename_prefix: str = "jarvina_gen"
) -> str:
    """
    Gera uma imagem a partir de um prompt de texto usando o provedor configurado.
    
    Args:
        working_dir: Absolute path to the working directory for file operations
        prompt: A descrição detalhada da imagem desejada.
        model: O ID do modelo a ser usado (use list_available_models para descobrir).
        size: Dimensões básicas (ex: "1024x1024").
        aspect_ratio: Opcional. Proporção da imagem (ex: "1:1", "16:9").
        resolution: Opcional. Resolução desejada (ex: "1K", "2K", "4K").
        cfg_scale: Opcional. Quão estritamente o modelo deve seguir o prompt (ex: 7.5).
        negative_prompt: Opcional. O que NÃO deve aparecer na imagem.
        output_dir: Directory relative to working_dir to save generated images
        filename_prefix: Prefix for generated image filenames
    """
    try:
        working_path = Path(working_dir)
        if not working_path.is_absolute():
            return f"Error: working_dir must be an absolute path, got: {working_dir}"
        if not working_path.exists():
            return f"Error: working_dir does not exist: {working_dir}"
        if not working_path.is_dir():
            return f"Error: working_dir is not a directory: {working_dir}"
            
        output_path = working_path / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        extra_params = {}
        if aspect_ratio:
            extra_params["aspect_ratio"] = aspect_ratio
        if resolution:
            extra_params["resolution"] = resolution
        if cfg_scale is not None:
            extra_params["cfg_scale"] = cfg_scale
        if negative_prompt:
            extra_params["negative_prompt"] = negative_prompt

        response = client.images.generate(
            prompt=prompt,
            model=model,
            size=size,
            extra_body=extra_params if extra_params else None,
            response_format="b64_json"
        )
        
        saved_files = []
        import time
        timestamp = int(time.time())
        
        for i, image_data in enumerate(response.data):
            filename = f"{filename_prefix}_{timestamp}_{i+1}.png"
            file_path = output_path / filename
            
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                if save_base64_image(image_data.b64_json, file_path):
                    saved_files.append(str(file_path))
                else:
                    return f"Error: Failed to save image {i+1}"
            elif hasattr(image_data, 'url') and image_data.url:
                if download_image_from_url(image_data.url, file_path):
                    saved_files.append(str(file_path))
                else:
                    return f"Error: Failed to download image {i+1} from URL"
            else:
                return f"Error: No image data found in response for image {i+1}"
        
        result = f"Imagem gerada com sucesso! (Modelo: {model})\n\n"
        result += f"Prompt: {prompt}\n"
        if negative_prompt:
            result += f"Negative Prompt: {negative_prompt}\n"
        result += f"Parameters: size={size}"
        if aspect_ratio:
            result += f", aspect_ratio={aspect_ratio}"
        if resolution:
            result += f", resolution={resolution}"
        if cfg_scale:
            result += f", cfg_scale={cfg_scale}"
        result += "\n\nGenerated files:\n"
        
        for file_path in saved_files:
            result += f"- {file_path}\n"
            
        return result
        
    except Exception as e:
        return f"Erro ao gerar imagem com o provedor: {str(e)}"

@mcp.tool()
def edit_image(
    working_dir: str,
    image_path: str,
    prompt: str,
    mask_path: Optional[str] = None,
    model: str = "gpt-image-1",
    size: Optional[str] = None,
    quality: Optional[str] = None,
    n: int = 1,
    output_dir: str = "./edited_images",
    filename_prefix: str = "edited"
) -> str:
    """
    [AVISO: Ferramenta temporariamente desativada]
    Edit or extend existing images using OpenAI's image editing capabilities.
    """
    return "Aviso: A edição de imagem (inpaint/outpaint) pode não ser totalmente suportada pelo provedor atual (ex: Venice.ai). Esta função foi temporariamente desativada para focar na estabilidade."

@mcp.tool()
def create_image_variations(
    working_dir: str,
    image_path: str,
    n: int = 2,
    size: Optional[str] = "1024x1024",
    output_dir: str = "./image_variations",
    filename_prefix: str = "variation"
) -> str:
    """
    [AVISO: Ferramenta temporariamente desativada]
    Create variations of an existing image using DALL-E 2.
    """
    return "Aviso: A criação de variações de imagem pode não ser totalmente suportada pelo provedor atual (ex: Venice.ai). Esta função foi temporariamente desativada."

@mcp.tool()
def list_generated_images(working_dir: str, directory: str = "generated_images") -> str:
    """
    List all generated images in a directory with metadata.
    
    Args:
        working_dir: Absolute path to the working directory for file operations
        directory: Directory relative to working_dir to scan for generated images
    
    Returns:
        List of generated images with their metadata
    """
    try:
        # Validate working directory
        working_path = Path(working_dir)
        if not working_path.is_absolute():
            return f"Error: working_dir must be an absolute path, got: {working_dir}"
        if not working_path.exists():
            return f"Error: working_dir does not exist: {working_dir}"
        if not working_path.is_dir():
            return f"Error: working_dir is not a directory: {working_dir}"
        
        # Create directory path relative to working directory
        dir_path = working_path / directory
        if not dir_path.exists():
            return f"Directory '{directory}' does not exist"
        
        if not dir_path.is_dir():
            return f"'{directory}' is not a directory"
        
        # Find image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        image_files = []
        
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        if not image_files:
            return f"No image files found in '{directory}'"
        
        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        result = f"Generated Images in '{directory}':\n"
        result += f"Found {len(image_files)} image file(s)\n\n"
        
        from utils.openai_client import get_image_info
        import time
        
        for i, file_path in enumerate(image_files, 1):
            file_stats = file_path.stat()
            image_info = get_image_info(file_path)
            
            result += f"{i}. {file_path.name}\n"
            result += f"   Path: {file_path}\n"
            result += f"   Size: {file_stats.st_size:,} bytes ({file_stats.st_size / 1024 / 1024:.2f} MB)\n"
            result += f"   Modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_stats.st_mtime))}\n"
            
            if "error" not in image_info:
                result += f"   Dimensions: {image_info['size'][0]}x{image_info['size'][1]} pixels\n"
                result += f"   Format: {image_info.get('format', 'Unknown')}\n"
            
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error listing generated images: {str(e)}" 