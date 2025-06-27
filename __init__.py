#!/usr/bin/env python3
"""
LLMsGen SDK - AI-Powered Website Content Extraction for LLMs

A production-ready Python SDK for generating llms.txt files from websites using 
advanced web crawling and AI-powered content analysis.

Basic Usage:
    from llmsgen import LLMsGenerator
    
    # Initialize the generator
    generator = LLMsGenerator()
    
    # Generate llms.txt from a website
    await generator.generate_from_url("https://example.com")

Advanced Usage:
    from llmsgen import LLMsGenerator, WebCrawler, ModelManager
    
    # Custom configuration
    generator = LLMsGenerator()
    
    # Set up AI model
    model_manager = ModelManager()
    generator.setup_ai_model("gemini-1.5-flash")
    
    # Advanced crawling options
    await generator.generate_from_url(
        "https://example.com",
        max_pages=1000,
        crawl_mode="comprehensive",
        output_format="json",
        include_full_text=True
    )
"""

__version__ = "1.0.0"
__author__ = "LLMsGen Team"
__email__ = "hrishikeshgupta007@gmail.com"
__description__ = "AI-Powered Website Content Extraction for LLMs"

# Core SDK Classes - use flexible imports
try:
    # Try relative imports first (when installed as package)
    from .generator.llms_generator import LLMsTxtGenerator as LLMsGenerator
    from .crawler.web_crawler import WebCrawler
    from .models.client import ModelManager, AIClient
    from .config import *
    from .utils.file_utils import ensure_output_dir, save_api_key_to_env
    from .utils.text_utils import clean_text, extract_key_sentences
except ImportError:
    # Fall back to absolute imports (when running from source)
    from generator.llms_generator import LLMsTxtGenerator as LLMsGenerator
    from crawler.web_crawler import WebCrawler
    from models.client import ModelManager, AIClient
    from config import *
    from utils.file_utils import ensure_output_dir, save_api_key_to_env
    from utils.text_utils import clean_text, extract_key_sentences

__all__ = [
    # Main SDK class
    'LLMsGenerator',
    
    # Core components
    'WebCrawler',
    'ModelManager', 
    'AIClient',
    
    # Utilities
    'ensure_output_dir',
    'save_api_key_to_env',
    'clean_text',
    'extract_key_sentences',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__description__'
] 