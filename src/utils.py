import os
from pathlib import Path
import hashlib
from typing import Dict, Any

def setup_directories():
    """Create necessary directories for the application"""
    directories = [
        "data",
        "chunks", 
        "vectordb",
        "notebooks",
        ".streamlit"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def collect_generator_response(response_generator):
    """Collect streaming response into complete text - GENERATOR FIX"""
    if isinstance(response_generator, str):
        return response_generator
    
    response_text = ""
    try:
        for chunk in response_generator:
            if chunk:
                response_text += str(chunk)
    except Exception as e:
        print(f"Error collecting response: {e}")
        response_text = "Sorry, I encountered an error generating the response."
    
    return response_text

def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content"""
    return hashlib.md5(file_content).hexdigest()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names)-1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported"""
    supported_extensions = {'.txt', '.pdf', '.docx', '.doc'}
    file_extension = Path(filename).suffix.lower()
    return file_extension in supported_extensions

def clean_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    return {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "platform": os.name,
        "current_directory": str(Path.cwd())
    }

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def count_tokens_estimate(text: str) -> int:
    """Rough estimate of token count (4 chars per token average)"""
    return len(text) // 4

def sanitize_input(text: str) -> str:
    """Basic input sanitization"""
    # Remove potential harmful characters
    text = text.replace('<script>', '').replace('</script>', '')
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    return text.strip()