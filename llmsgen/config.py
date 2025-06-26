"""
Configuration module for the LLMs.txt Generator.

Handles loading of environment variables, defining constants,
and setting up the application logger.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants
MAX_GEN_OUTPUT_TOKENS = int(os.getenv('MAX_GEN_OUTPUT_TOKENS', '1024'))
CACHE_DESCRIPTIONS = os.getenv('CACHE_DESCRIPTIONS', 'true').lower() == 'true'
DEFAULT_PARALLEL_WORKERS = int(os.getenv('DEFAULT_PARALLEL_WORKERS', '3'))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')

# Local model optimization constants
LOCAL_MODEL_BATCH_SIZE = 3  # Smaller batches for local models
LOCAL_MODEL_TIMEOUT = 180   # Longer timeout for local inference
LOCAL_MODEL_RETRY_DELAY = 2 # Longer delay between retries for local models

# Configure logging
LOG_FILE_PATH = 'llmstxt_generator.log'

def get_logger(name: str) -> logging.Logger:
    """Initializes and returns a logger instance."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if logger is already configured
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream handler (console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # Attempt to set stdout encoding to UTF-8 for emojis in Windows terminals.
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except (TypeError, AttributeError):
                # This can fail in non-interactive shells or older Python versions.
                logger.info("Could not reconfigure stdout for UTF-8. Emojis may not render correctly in the console.")

    return logger

# Default logger for the application
# This can be imported by other modules: from llmsgen.config import logger
logger = get_logger(__name__.split('.')[0]) # Use the root package name for the logger

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
