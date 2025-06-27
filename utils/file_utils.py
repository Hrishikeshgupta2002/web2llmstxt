#!/usr/bin/env python3
"""
File utility functions for managing output files and directories.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from dotenv import set_key

logger = logging.getLogger(__name__)


def ensure_output_dir(output_dir: str = "output") -> None:
    """Ensure the output directory exists"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise


def write_safe_file(filename: str, content: str, encoding: str = 'utf-8') -> None:
    """Write content to file safely with proper error handling"""
    try:
        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Write the file
        with open(filename, 'w', encoding=encoding) as f:
            f.write(content)
        
        logger.debug(f"Successfully wrote file: {filename}")
        
    except Exception as e:
        logger.error(f"Failed to write file {filename}: {e}")
        raise


def read_safe_file(filename: str, encoding: str = 'utf-8') -> Optional[str]:
    """Read file content safely with proper error handling"""
    try:
        with open(filename, 'r', encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"File not found: {filename}")
        return None
    except Exception as e:
        logger.error(f"Failed to read file {filename}: {e}")
        return None


def file_exists(filename: str) -> bool:
    """Check if file exists"""
    return os.path.isfile(filename)


def get_file_size(filename: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(filename)
    except OSError:
        return 0


def create_sample_env_file(env_file: str = ".env") -> None:
    """Create a sample .env file with configuration templates"""
    
    sample_content = """# LLMs.txt Generator Configuration
# Copy this file to .env and fill in your actual values

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Ollama Configuration  
OLLAMA_BASE_URL=http://localhost:11434

# Generation Settings
MAX_GEN_OUTPUT_TOKENS=1024
CACHE_DESCRIPTIONS=true
DEFAULT_PARALLEL_WORKERS=3

# Output Configuration
OUTPUT_DIR=./output

# Logging Level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Model Selection Preferences
PREFERRED_PROVIDER=gemini
PREFERRED_MODEL=gemini-1.5-flash

# Crawling Settings
DEFAULT_MAX_PAGES=50
DEFAULT_BATCH_SIZE=10
DEFAULT_PARALLEL_WORKERS=5

# Performance Settings
LOCAL_MODEL_BATCH_SIZE=3
LOCAL_MODEL_TIMEOUT=180
LOCAL_MODEL_RETRY_DELAY=2

# Cache Settings
ENABLE_MODEL_CACHE=true
CACHE_EXPIRY_HOURS=24
"""
    
    try:
        if not file_exists(env_file):
            write_safe_file(env_file, sample_content)
            logger.info(f"Created sample environment file: {env_file}")
            print(f"✅ Created sample .env file: {env_file}")
            print("📝 Please edit this file with your actual API keys and preferences.")
        else:
            logger.info(f"Environment file already exists: {env_file}")
            print(f"ℹ️  Environment file already exists: {env_file}")
            
    except Exception as e:
        logger.error(f"Failed to create sample .env file: {e}")
        print(f"❌ Failed to create sample .env file: {e}")


def save_api_key_to_env(api_key: str, key_name: str = "GEMINI_API_KEY", env_file: str = ".env") -> None:
    """Save API key to .env file"""
    try:
        set_key(env_file, key_name, api_key)
        logger.info(f"Saved {key_name} to {env_file}")
        print(f"✅ Saved {key_name} to {env_file}")
    except Exception as e:
        logger.error(f"Failed to save API key to {env_file}: {e}")
        print(f"❌ Failed to save API key: {e}")


def load_json_file(filename: str) -> Optional[dict]:
    """Load JSON file safely"""
    try:
        content = read_safe_file(filename)
        if content:
            return json.loads(content)
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        return None


def save_json_file(filename: str, data: dict, indent: int = 2) -> None:
    """Save data as JSON file"""
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        write_safe_file(filename, content)
    except Exception as e:
        logger.error(f"Failed to save JSON file {filename}: {e}")
        raise


def backup_file(filename: str, backup_suffix: str = ".bak") -> Optional[str]:
    """Create a backup of a file"""
    if not file_exists(filename):
        return None
    
    backup_filename = filename + backup_suffix
    try:
        content = read_safe_file(filename)
        if content:
            write_safe_file(backup_filename, content)
            logger.info(f"Created backup: {backup_filename}")
            return backup_filename
    except Exception as e:
        logger.error(f"Failed to create backup of {filename}: {e}")
    
    return None


def clean_filename(filename: str) -> str:
    """Clean filename by removing/replacing invalid characters"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = "untitled"
    
    return filename


def get_available_filename(base_filename: str, extension: str = "") -> str:
    """Get an available filename by adding a number suffix if needed"""
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    full_filename = base_filename + extension
    
    if not file_exists(full_filename):
        return full_filename
    
    # Try with number suffixes
    counter = 1
    while True:
        numbered_filename = f"{base_filename}_{counter}{extension}"
        if not file_exists(numbered_filename):
            return numbered_filename
        counter += 1
        
        # Safety check to avoid infinite loop
        if counter > 1000:
            raise ValueError("Cannot find available filename after 1000 attempts")


def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()


def is_text_file(filename: str) -> bool:
    """Check if file is likely a text file based on extension"""
    text_extensions = {'.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.js', '.py', '.log'}
    return get_file_extension(filename) in text_extensions 