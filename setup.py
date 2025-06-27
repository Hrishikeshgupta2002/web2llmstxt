#!/usr/bin/env python3
"""
Setup script for LLMsGen SDK
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AI-Powered Website Content Extraction for LLMs"

# Read version from package
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return "1.0.0"

setup(
    name="llmsgen",
    version=get_version(),
    author="LLMsGen Team",
    author_email="hrishikeshgupta007@gmail.com",
    description="AI-Powered Website Content Extraction for LLMs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llmsgen",
    packages=["llmsgen", "llmsgen.crawler", "llmsgen.generator", "llmsgen.models", "llmsgen.utils"],
    package_dir={"llmsgen": "."},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Text Processing :: Markup :: HTML",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "crawl4ai[all]>=0.6.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "lxml>=4.6.0",
        
        # AI/ML dependencies
        "google-generativeai>=0.3.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        
        # Utility dependencies
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "tenacity>=8.0.0",
        "tqdm>=4.62.0",
        
        # Optional dependencies for enhanced features
        "playwright>=1.20.0",
        "selenium>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "full": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "llmsgen=scripts.generate_llms:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llmsgen": [
            "playwright/.local-browsers/**/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "llm", "ai", "web-scraping", "content-extraction", 
        "crawling", "chatgpt", "claude", "gemini", "ollama",
        "llms-txt", "machine-learning", "nlp", "automation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llmsgen/issues",
        "Source": "https://github.com/yourusername/llmsgen",
        "Documentation": "https://llmsgen.readthedocs.io/",
    },
) 