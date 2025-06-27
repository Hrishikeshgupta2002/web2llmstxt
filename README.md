# 🚀 LLMsGen SDK

**AI-Powered Website Content Extraction for LLMs**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Production Ready](https://img.shields.io/badge/status-production%20ready-green.svg)]()

LLMsGen is a production-ready Python SDK for generating `llms.txt` files from websites using advanced web crawling and AI-powered content analysis. Extract clean, structured content from any website for training or fine-tuning large language models.

## ✨ Features

- 🕷️ **Advanced Web Crawling** - Multi-strategy crawling (systematic, comprehensive, sitemap-based)
- 🤖 **AI-Powered Analysis** - Support for Ollama, Gemini, OpenAI, and Anthropic models
- 🗺️ **Sitemap Integration** - Automatic sitemap discovery and processing
- 📊 **Multiple Output Formats** - Text, JSON, YAML with full-text options
- 🔒 **Production Security** - Rate limiting, memory management, error recovery
- ⚡ **High Performance** - Parallel processing with adaptive batch sizing
- 🎯 **Flexible API** - Use as CLI tool or integrate into your applications

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (when published)
pip install llmsgen

# Or install from source
git clone https://github.com/yourusername/llmsgen.git
cd llmsgen
pip install -e .
```

### Basic Usage

```python
from llmsgen import LLMsGenerator
import asyncio

async def main():
    # Initialize the generator
    generator = LLMsGenerator()
    
    # Generate llms.txt from a website
    await generator.generate_llmstxt(
        base_url="https://docs.python.org",
        max_pages=50,
        export_format='text'
    )

asyncio.run(main())
```

### Command Line Usage

```bash
# Interactive mode
llmsgen

# Direct URL
llmsgen https://docs.python.org

# Advanced options
llmsgen https://docs.python.org --max-pages 100 --format json --workers 10
```

## 📖 API Reference

### LLMsGenerator

The main SDK class for generating llms.txt files.

```python
from llmsgen import LLMsGenerator

generator = LLMsGenerator()

# Generate with custom options
await generator.generate_llmstxt(
    base_url="https://example.com",
    max_pages=100,                    # Number of pages to crawl
    export_format='text',             # 'text', 'json', 'yaml'
    crawl_strategy='comprehensive',   # 'systematic', 'comprehensive', 'sitemap'
    include_full_text=True,          # Include full content
    parallel_workers=8,              # Concurrent workers
    batch_size=15,                   # Batch processing size
    sitemap_url='auto'               # Sitemap URL or 'auto'
)
```

### WebCrawler

Advanced web crawler with multiple strategies.

```python
from llmsgen import WebCrawler

crawler = WebCrawler()

# Systematic crawling
pages = await crawler.crawl_website(
    base_url="https://example.com",
    max_pages=50,
    comprehensive=False
)

# Sitemap-based crawling
pages = await crawler.crawl_from_sitemap(
    base_url="https://example.com",
    sitemap_url="auto",
    max_pages=1000
)
```

### ModelManager

AI model management and integration.

```python
from llmsgen import ModelManager, AIClient

# Initialize model manager
model_manager = ModelManager()

# List available models
models = model_manager.list_models()

# Setup AI client
ai_client = AIClient(model_manager)
ai_client.set_model(selected_model)
```

## 🎯 Usage Examples

### Comprehensive Website Crawling

```python
from llmsgen import LLMsGenerator

async def comprehensive_crawl():
    generator = LLMsGenerator()
    
    # Crawl entire domain with unlimited pages
    await generator.generate_llmstxt(
        base_url="https://fastapi.tiangolo.com",
        max_pages=999999,  # Unlimited
        crawl_strategy='comprehensive',
        export_format='json',
        include_full_text=True
    )
```

### Sitemap-Based Extraction

```python
async def sitemap_extraction():
    generator = LLMsGenerator()
    
    # Extract all pages from sitemap
    await generator.generate_llmstxt(
        base_url="https://www.alternates.ai",
        crawl_strategy='sitemap',
        sitemap_url='https://www.alternates.ai/sitemap.xml',
        export_format='text'
    )
```

### Custom Processing Pipeline

```python
from llmsgen import WebCrawler

async def custom_pipeline():
    crawler = WebCrawler()
    
    # Step 1: Crawl website
    pages = await crawler.crawl_website("https://example.com")
    
    # Step 2: Custom filtering
    quality_pages = [
        page for page in pages 
        if page['word_count'] > 200
    ]
    
    # Step 3: Custom processing
    # Your custom logic here...
```

### Batch Processing

```python
async def batch_process():
    generator = LLMsGenerator()
    
    websites = [
        "https://docs.python.org",
        "https://fastapi.tiangolo.com",
        "https://www.djangoproject.com"
    ]
    
    for website in websites:
        await generator.generate_llmstxt(
            base_url=website,
            max_pages=50,
            export_format='json'
        )
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file for configuration:

```env
# AI Model API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Output Configuration
OUTPUT_DIR=./output
USE_LOCAL_PLAYWRIGHT=true
LOCAL_PLAYWRIGHT_BROWSERS=auto

# Performance Settings
DEFAULT_WORKERS=5
DEFAULT_BATCH_SIZE=10
MAX_PAGES_DEFAULT=50
```

### Model Configuration

The SDK supports multiple AI providers:

- **Ollama** (local models) - llama3.2, codellama, etc.
- **Google Gemini** - gemini-1.5-flash, gemini-1.5-pro
- **OpenAI** - gpt-4, gpt-3.5-turbo
- **Anthropic** - claude-3-sonnet, claude-3-haiku

## 🛠️ Advanced Features

### Custom Crawling Strategies

```python
# Systematic: Main page + direct links (default)
crawl_strategy='systematic'

# Comprehensive: Deep recursive crawling
crawl_strategy='comprehensive'

# Sitemap: Use website's sitemap for discovery
crawl_strategy='sitemap'
```

### Output Formats

```python
# Plain text format (default)
export_format='text'

# Structured JSON with metadata
export_format='json'

# YAML format for configuration
export_format='yaml'
```

### Performance Tuning

```python
await generator.generate_llmstxt(
    base_url="https://example.com",
    parallel_workers=10,      # More workers = faster crawling
    batch_size=20,           # Larger batches = better throughput
    max_pages=1000,          # Adjust based on site size
    crawl_strategy='sitemap' # Fastest for large sites
)
```

## 📊 Output Structure

### Text Format (`llms.txt`)

```
# Website: https://example.com
# Generated: 2024-01-20T10:30:00Z
# Pages: 156
# Strategy: comprehensive

## Page 1: Homepage | https://example.com
This is the main content of the homepage...

## Page 2: About Us | https://example.com/about
Information about the company...
```

### JSON Format

```json
{
  "metadata": {
    "base_url": "https://example.com",
    "generation_timestamp": "2024-01-20T10:30:00Z",
    "total_pages": 156,
    "crawl_strategy": "comprehensive",
    "model_used": "gemini-1.5-flash"
  },
  "pages": [
    {
      "url": "https://example.com",
      "title": "Homepage",
      "description": "AI-generated description...",
      "word_count": 1250,
      "content": "Full page content...",
      "crawl_timestamp": "2024-01-20T10:30:00Z"
    }
  ]
}
```

## 🔧 Development

### Setup Development Environment

```bash
git clone https://github.com/yourusername/llmsgen.git
cd llmsgen

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/llmsgen/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/llmsgen/discussions)
- 📧 **Email**: hrishikeshgupta007@gmail.com

## 🚀 What's Next?

- [ ] GraphQL API support
- [ ] Real-time streaming crawling
- [ ] Advanced content filtering
- [ ] Multi-language support
- [ ] Docker containers
- [ ] Cloud deployment options

## 📧 Contact & Feedback

For questions, suggestions, or collaboration opportunities:
- **Email**: hrishikeshgupta007@gmail.com
- **GitHub**: [Create an Issue](https://github.com/yourusername/llmsgen/issues)

## 🙏 Acknowledgments

- Built with [Crawl4AI](https://github.com/unclecode/crawl4ai) for advanced web crawling
- Supports multiple AI providers: Ollama, Google Gemini, OpenAI, Anthropic
- Inspired by the llms.txt specification for LLM training data

---

**Made with ❤️ by the LLMsGen Team** 
