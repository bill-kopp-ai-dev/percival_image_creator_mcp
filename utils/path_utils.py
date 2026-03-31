from pathlib import Path
from typing import Tuple, Optional
import os


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


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def get_allowed_working_roots() -> list[Path]:
    """
    Return allowed roots for working_dir containment.

    Environment:
      - PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS: comma-separated absolute paths.
      - PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX: when true, disables root containment.
    """
    if _env_bool("PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX", False):
        return []

    raw = os.getenv("PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS", "").strip()
    roots: list[Path] = []

    if raw:
        candidates = [item.strip() for item in raw.split(",") if item.strip()]
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                continue
            try:
                roots.append(path.resolve(strict=True))
            except Exception:
                # Ignore invalid roots; validation function will fail if no valid roots remain.
                continue
        return roots

    # Secure-by-default fallback: confine working_dir to current process working directory.
    try:
        return [Path(os.getcwd()).resolve(strict=True)]
    except Exception:
        return []


def validate_working_directory(working_dir: str) -> Tuple[Optional[Path], Optional[str]]:
    """
    Validate working_dir and enforce root containment policy.
    """
    working_path = Path(working_dir).expanduser()
    if not working_path.is_absolute():
        return None, f"Error: working_dir must be an absolute path, got: {working_dir}"
    if not working_path.exists():
        return None, f"Error: working_dir does not exist: {working_dir}"
    if not working_path.is_dir():
        return None, f"Error: working_dir is not a directory: {working_dir}"

    try:
        resolved_working = working_path.resolve(strict=True)
    except Exception as exc:
        return None, f"Error: failed to resolve working_dir '{working_dir}': {exc}"

    if _env_bool("PERCIVAL_IMAGE_MCP_DISABLE_ROOT_SANDBOX", False):
        return resolved_working, None

    allowed_roots = get_allowed_working_roots()
    if not allowed_roots:
        return None, (
            "Error: no valid allowed roots configured for working_dir sandbox. "
            "Set PERCIVAL_IMAGE_MCP_ALLOWED_ROOTS with absolute existing directories."
        )

    if not any(_is_relative_to(resolved_working, root) for root in allowed_roots):
        allowed_display = ", ".join(str(root) for root in allowed_roots)
        return None, (
            "Error: working_dir is outside allowed roots.\n"
            f"• working_dir: '{resolved_working}'\n"
            f"• allowed_roots: '{allowed_display}'"
        )

    return resolved_working, None


def resolve_path(file_path: str, base_dir: str) -> Path:
    """
    Handle both relative and absolute paths for image files.
    
    Args:
        file_path: Path to the image file, can be:
            - Relative path: "test_data/image.jpg" (resolved relative to base_dir)  
            - Absolute path: "/home/user/images/image.jpg" (used as-is)
        base_dir: Base directory for resolving relative paths (required)
    
    Returns:
        Path: Resolved absolute Path object
        
    Examples:
        # Relative path (resolved from client's working directory)
        resolve_path("test_data/logo.jpg", "/client/working/directory")
        # Returns: /client/working/directory/test_data/logo.jpg
        
        # Absolute path (used as-is)
        resolve_path("/home/user/images/logo.jpg", "/any/base/dir") 
        # Returns: /home/user/images/logo.jpg
        
        # Relative path with explicit base directory
        resolve_path("../images/logo.jpg", "/some/base/dir")
        # Returns: /some/base/images/logo.jpg
    """
    path = Path(file_path)
    if path.is_absolute():
        return path
    
    # For relative paths, use provided base_dir
    base = Path(base_dir)
    return base / path

def get_client_working_directory() -> str:
    """
    Attempt to determine the client's working directory.
    
    Returns:
        String path of the most likely client working directory
    """
    # Try PWD environment variable first (more likely to be client's PWD)
    pwd = os.environ.get('PWD')
    if pwd and Path(pwd).exists():
        return pwd
    
    # Fallback to process working directory
    return os.getcwd()

def validate_image_path(file_path: str, operation: str = "access", base_dir: str = None) -> Tuple[bool, Optional[str], Optional[Path]]:
    """
    Validate image path with clear error messages for AI agents.
    
    Args:
        file_path: Path to validate (relative or absolute)
        operation: Type of operation ("read", "write", "access")
        base_dir: Base directory for resolving relative paths
    
    Returns:
        Tuple of (is_valid, error_message, resolved_path)
        - is_valid: True if path is valid for the operation
        - error_message: None if valid, detailed error message if invalid
        - resolved_path: Resolved Path object if valid, None if invalid
    """
    if not file_path or not file_path.strip():
        return False, "Error: Empty or invalid file path provided. Please provide a valid image file path.", None
    
    # Determine if path is absolute or relative
    is_absolute = file_path.startswith('/')
    path_type = "absolute" if is_absolute else "relative"
    
    # Get the effective base directory for relative paths
    if not is_absolute:
        effective_base = base_dir or get_client_working_directory()
    else:
        effective_base = None
    
    try:
        resolved_path = resolve_path(file_path.strip(), effective_base)
        
        # For read operations, check if file exists
        if operation == "read" or operation == "access":
            if not resolved_path.exists():
                error_msg = (
                    f"Error: Image file not found.\n"
                    f"• Provided path: '{file_path}' ({path_type})\n"
                    f"• Resolved to: '{resolved_path}'\n"
                )
                
                if not is_absolute:
                    error_msg += f"• Base directory: '{effective_base}'\n"
                    error_msg += f"• Note: Relative paths are resolved from the directory where you're running the client\n"
                
                error_msg += (
                    f"• Suggestion: Verify the file exists and path is correct. "
                    f"Use absolute paths (starting with '/') for files outside your current directory."
                )
                
                return False, error_msg, None
            
            if not resolved_path.is_file():
                error_msg = (
                    f"Error: Path exists but is not a file.\n"
                    f"• Provided path: '{file_path}' ({path_type})\n"
                    f"• Resolved to: '{resolved_path}'\n"
                    f"• Path type: {'Directory' if resolved_path.is_dir() else 'Other'}\n"
                )
                
                if not is_absolute:
                    error_msg += f"• Base directory: '{effective_base}'\n"
                
                error_msg += f"• Suggestion: Provide a path to an image file, not a directory."
                
                return False, error_msg, None
        
        # For write operations, check if parent directory exists and is writable
        elif operation == "write":
            parent_dir = resolved_path.parent
            if not parent_dir.exists():
                error_msg = (
                    f"Error: Output directory does not exist.\n"
                    f"• Provided path: '{file_path}' ({path_type})\n"
                    f"• Output directory: '{parent_dir}'\n"
                )
                
                if not is_absolute:
                    error_msg += f"• Base directory: '{effective_base}'\n"
                
                error_msg += f"• Suggestion: Create the directory first or use an existing directory path."
                
                return False, error_msg, None
            
            if not parent_dir.is_dir():
                error_msg = (
                    f"Error: Parent path is not a directory.\n"
                    f"• Provided path: '{file_path}' ({path_type})\n"
                    f"• Parent path: '{parent_dir}'\n"
                )
                
                if not is_absolute:
                    error_msg += f"• Base directory: '{effective_base}'\n"
                
                error_msg += f"• Suggestion: Ensure the parent path is a valid directory."
                
                return False, error_msg, None
        
        return True, None, resolved_path
        
    except Exception as e:
        error_msg = (
            f"Error: Invalid file path format.\n"
            f"• Provided path: '{file_path}'\n"
            f"• Error details: {str(e)}\n"
        )
        
        if not is_absolute:
            error_msg += f"• Base directory: '{effective_base}'\n"
        
        error_msg += (
            f"• Suggestion: Use a valid file path. Examples:\n"
            f"  - Relative: 'images/photo.jpg' or './images/photo.jpg'\n"
            f"  - Absolute: '/home/user/images/photo.jpg'"
        )
        
        return False, error_msg, None 
