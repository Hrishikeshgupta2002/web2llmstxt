#!/usr/bin/env python3
"""
Configuration module for LLMs.txt Generator.

This module handles configuration settings, environment variables,
and logging setup for the entire package.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants
MAX_GEN_OUTPUT_TOKENS = int(os.getenv('MAX_GEN_OUTPUT_TOKENS', '1024'))
CACHE_DESCRIPTIONS = os.getenv('CACHE_DESCRIPTIONS', 'true').lower() == 'true'
DEFAULT_PARALLEL_WORKERS = int(os.getenv('DEFAULT_PARALLEL_WORKERS', '3'))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')

# Local model optimization constants
LOCAL_MODEL_BATCH_SIZE = int(os.getenv('LOCAL_MODEL_BATCH_SIZE', '3'))
LOCAL_MODEL_TIMEOUT = int(os.getenv('LOCAL_MODEL_TIMEOUT', '180'))
LOCAL_MODEL_RETRY_DELAY = int(os.getenv('LOCAL_MODEL_RETRY_DELAY', '2'))

# Ollama configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

# Gemini configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Playwright configuration
USE_LOCAL_PLAYWRIGHT = os.getenv('USE_LOCAL_PLAYWRIGHT', 'true').lower() == 'true'
LOCAL_PLAYWRIGHT_BROWSERS = os.getenv('LOCAL_PLAYWRIGHT_BROWSERS', 'auto')  # 'auto' to detect automatically

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

def setup_logging(verbose: bool = False, log_file: str = 'llmstxt_generator.log'):
    """
    Setup logging configuration for the package.
    
    Args:
        verbose: Enable debug level logging
        log_file: Log file path
    """
    # Determine log level
    if verbose:
        level = logging.DEBUG
    else:
        level = getattr(logging, LOG_LEVEL, logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # File handler with detailed format
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file {log_file}: {e}")
    
    # Console handler with simple format for non-debug levels
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if verbose:
        console_handler.setFormatter(detailed_formatter)
    else:
        console_handler.setFormatter(simple_formatter)
    
    root_logger.addHandler(console_handler)
    
    # Fix Windows UTF-8 encoding for emojis
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except (TypeError, AttributeError):
            root_logger.info("Could not reconfigure stdout for UTF-8. Emojis may not render correctly.")
    
    return root_logger

# Initialize default logger
logger = setup_logging()

def get_config_summary() -> dict:
    """Get a summary of current configuration settings"""
    return {
        'max_gen_output_tokens': MAX_GEN_OUTPUT_TOKENS,
        'cache_descriptions': CACHE_DESCRIPTIONS,
        'default_parallel_workers': DEFAULT_PARALLEL_WORKERS,
        'output_dir': OUTPUT_DIR,
        'local_model_batch_size': LOCAL_MODEL_BATCH_SIZE,
        'local_model_timeout': LOCAL_MODEL_TIMEOUT,
        'local_model_retry_delay': LOCAL_MODEL_RETRY_DELAY,
        'ollama_base_url': OLLAMA_BASE_URL,
        'gemini_api_key_configured': bool(GEMINI_API_KEY),
        'log_level': LOG_LEVEL
    }

def validate_config() -> list:
    """Validate configuration and return list of issues"""
    issues = []
    
    # Check if output directory is writable
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        test_file = os.path.join(OUTPUT_DIR, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        issues.append(f"Output directory {OUTPUT_DIR} is not writable: {e}")
    
    # Check if at least one AI provider is configured
    has_ollama = True  # We can check Ollama availability at runtime
    has_gemini = bool(GEMINI_API_KEY)
    
    if not has_gemini:
        issues.append("No Gemini API key configured. Set GEMINI_API_KEY in .env file or environment.")
    
    # Validate numeric settings
    if MAX_GEN_OUTPUT_TOKENS <= 0:
        issues.append("MAX_GEN_OUTPUT_TOKENS must be positive")
    
    if DEFAULT_PARALLEL_WORKERS <= 0:
        issues.append("DEFAULT_PARALLEL_WORKERS must be positive")
    
    if LOCAL_MODEL_TIMEOUT <= 0:
        issues.append("LOCAL_MODEL_TIMEOUT must be positive")
    
    return issues

def print_config_info():
    """Print configuration information"""
    config = get_config_summary()
    issues = validate_config()
    
    print("🔧 Configuration Summary:")
    print("=" * 50)
    
    for key, value in config.items():
        key_display = key.replace('_', ' ').title()
        print(f"{key_display}: {value}")
    
    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration is valid")
    
    print("=" * 50) 