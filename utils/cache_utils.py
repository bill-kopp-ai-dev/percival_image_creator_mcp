import json
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import time
from urllib.parse import urlparse

from utils.client import get_jarvina_base_url
from utils.security_utils import record_security_event

logger = logging.getLogger(__name__)

class ImageAnalysisCache:
    """
    Cache for OpenAI image analysis responses using file hashes for cache invalidation.
    Cache files are stored in the OS temporary directory.
    """
    
    def __init__(self, cache_subdir: str = "ai_image_analysis_cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_subdir: Subdirectory name within the temp directory for cache files
        """
        self.cache_dir = Path(tempfile.gettempdir()) / cache_subdir
        self.cache_dir.mkdir(mode=0o700, exist_ok=True)
        try:
            os.chmod(self.cache_dir, 0o700)
        except Exception:
            # Non-fatal on platforms/filesystems without chmod support.
            pass
        
    def _get_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of the file content.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_cache_key(self, file_path: Path, operation: str, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for the file and operation, tied to the provider.
        """
        provider_domain = urlparse(get_jarvina_base_url()).netloc or "unknown-provider"
        
        # Create a unique identifier based on file path and parameters
        param_str = json.dumps(params, sort_keys=True)
        key_content = f"{provider_domain}_{file_path.absolute()}_{operation}_{param_str}"
        
        # Hash the key content to create a safe filename
        key_hash = hashlib.md5(key_content.encode('utf-8')).hexdigest()
        return f"{operation}_{key_hash}.json"
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the full path to a cache file."""
        return self.cache_dir / cache_key

    def _is_relative_to(self, path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    def _is_safe_cache_file(self, cache_file: Path) -> bool:
        try:
            resolved_cache = cache_file.resolve(strict=False)
            resolved_base = self.cache_dir.resolve(strict=True)
            if not self._is_relative_to(resolved_cache, resolved_base):
                record_security_event(
                    "cache_path_blocked",
                    {"reason": "outside_cache_dir", "path": str(resolved_cache)},
                )
                return False

            if cache_file.exists() and cache_file.is_symlink():
                record_security_event(
                    "cache_path_blocked",
                    {"reason": "symlink_cache_file", "path": str(cache_file)},
                )
                return False
            return True
        except Exception as exc:
            record_security_event(
                "cache_path_blocked",
                {"reason": "cache_validation_error", "error": str(exc)},
            )
            return False
    
    def get_cached_result(self, file_path: Path, operation: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Retrieve cached result if available and valid.
        
        Args:
            file_path: Path to the image file
            operation: Type of operation
            params: Operation parameters
            
        Returns:
            Cached result string if valid cache exists, None otherwise
        """
        try:
            if not file_path.exists():
                return None
                
            cache_key = self._get_cache_key(file_path, operation, params)
            cache_file = self._get_cache_file_path(cache_key)

            if not self._is_safe_cache_file(cache_file):
                return None

            if not cache_file.exists():
                return None
            
            # Load cache data
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Verify cache is still valid by comparing file hash
            current_hash = self._get_file_hash(file_path)
            if cache_data.get('file_hash') != current_hash:
                # File has changed, cache is invalid
                cache_file.unlink(missing_ok=True)  # Remove invalid cache
                return None
            
            # Check cache age (optional - you can set max age if desired)
            cache_age = time.time() - cache_data.get('timestamp', 0)
            max_age = 30 * 24 * 3600  # 30 days in seconds
            if cache_age > max_age:
                cache_file.unlink(missing_ok=True)  # Remove expired cache
                return None
            
            return cache_data.get('result')
            
        except Exception as e:
            # If anything goes wrong with cache retrieval, just return None
            # This ensures the cache never breaks the main functionality
            logger.warning("Cache retrieval error (non-fatal): %s", e)
            return None
    
    def store_result(self, file_path: Path, operation: str, params: Dict[str, Any], result: str) -> None:
        """
        Store analysis result in cache.
        
        Args:
            file_path: Path to the image file
            operation: Type of operation
            params: Operation parameters
            result: Analysis result to cache
        """
        try:
            if not file_path.exists():
                return
                
            cache_key = self._get_cache_key(file_path, operation, params)
            cache_file = self._get_cache_file_path(cache_key)

            if not self._is_safe_cache_file(cache_file):
                return

            file_hash = self._get_file_hash(file_path)
            file_path_hash = hashlib.sha256(str(file_path.absolute()).encode("utf-8")).hexdigest()
            
            cache_data = {
                'file_path_hash': file_path_hash,
                'file_hash': file_hash,
                'operation': operation,
                'params': params,
                'result': result,
                'timestamp': time.time(),
                'file_size': file_path.stat().st_size,
                'file_mtime': file_path.stat().st_mtime
            }

            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                delete=False,
                dir=str(self.cache_dir),
            ) as tmp_file:
                json.dump(cache_data, tmp_file, indent=2, ensure_ascii=False)
                tmp_path = Path(tmp_file.name)

            try:
                os.chmod(tmp_path, 0o600)
            except Exception:
                pass

            os.replace(tmp_path, cache_file)
            try:
                os.chmod(cache_file, 0o600)
            except Exception:
                pass
                
        except Exception as e:
            # Cache storage errors should not break the main functionality
            logger.warning("Cache storage error (non-fatal): %s", e)
    
    def clear_cache(self) -> int:
        """
        Clear all cache files.
        
        Returns:
            Number of cache files removed
        """
        removed_count = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                if not self._is_safe_cache_file(cache_file):
                    continue
                cache_file.unlink()
                removed_count += 1
        except Exception as e:
            logger.warning("Cache clearing error: %s", e)
        return removed_count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())
            
            return {
                'cache_dir': str(self.cache_dir),
                'cache_files_count': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {'error': str(e)}

# Global cache instance
_cache_instance = None

def get_cache() -> ImageAnalysisCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ImageAnalysisCache()
    return _cache_instance 
