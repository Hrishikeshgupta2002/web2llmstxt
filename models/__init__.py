"""
Models package for LLMs.txt Generator.

This package contains model management and AI client functionality.
"""

try:
    from .config_types import ModelConfig
    from .client import ModelManager, AIClient
except ImportError:
    from models.config_types import ModelConfig
    from models.client import ModelManager, AIClient

__all__ = ['ModelConfig', 'ModelManager', 'AIClient'] 