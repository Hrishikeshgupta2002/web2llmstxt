"""
Generator package for LLMs.txt Generator.

This package contains the main LLMs.txt generation functionality.
"""

try:
    from .llms_generator import LLMsTxtGenerator
except ImportError:
    from generator.llms_generator import LLMsTxtGenerator

__all__ = ['LLMsTxtGenerator'] 