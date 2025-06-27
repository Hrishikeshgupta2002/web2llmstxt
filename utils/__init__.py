"""
Utilities package for LLMs.txt Generator.

This package contains various utility functions for text processing and file operations.
"""

try:
    from .text_utils import clean_text, extract_key_sentences, truncate_text
    from .file_utils import ensure_output_dir, write_safe_file, create_sample_env_file
except ImportError:
    from utils.text_utils import clean_text, extract_key_sentences, truncate_text
    from utils.file_utils import ensure_output_dir, write_safe_file, create_sample_env_file

__all__ = [
    'clean_text', 
    'extract_key_sentences', 
    'truncate_text',
    'ensure_output_dir', 
    'write_safe_file', 
    'create_sample_env_file'
] 