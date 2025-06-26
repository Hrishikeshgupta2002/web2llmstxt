# 🚀 Enhanced LLMs.txt Generator with Crawl4AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Crawl4AI](https://img.shields.io/badge/crawl4ai-0.6.0+-green.svg)](https://github.com/unclecode/crawl4ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, production-ready tool for generating comprehensive `llms.txt` and `llms-full.txt` files from any website using **Crawl4AI** and **local/cloud AI models**. Built with 20+ years of software engineering expertise for enterprise-grade performance and reliability.

## 🌟 Key Features

### 🎯 **Deep Website Crawling**
- **Multi-Wave Discovery**: 5-wave deep crawling with intelligent link prioritization
- **Enhanced Link Extraction**: Navigation menus, JavaScript data attributes, structured content
- **Documentation-Aware**: Special handling for docs sites with deeper nesting (up to 10 levels)
- **Smart Filtering**: Automatically prioritizes documentation, guides, and valuable content

### 🧠 **AI-Powered Content Processing**
- **Dual AI Support**: Ollama (local) and Gemini (cloud) models
- **Anti-Hallucination System**: Extracts real content instead of generating fictional descriptions
- **Smart Caching**: Intelligent description caching with content change detection
- **Fallback Mechanisms**: Robust error handling with 5 retry attempts

### 🔧 **Production-Grade Features**
- **Parallel Processing**: Configurable batch processing with worker pools
- **Memory Optimization**: RAM usage warnings and conservative settings for large models
- **Timeout Handling**: Adaptive timeouts (45s local, 30s cloud) with exponential backoff
- **Session Management**: Proper browser session cleanup and resource management

### 📊 **Enhanced Content Cleaning**
- **Multi-Method Title Extraction**: HTML tags, headers, URL slugs with normalization
- **Advanced Sentence Filtering**: Removes navigation text, UI elements, and irrelevant content
- **Content Type Detection**: Automatic categorization (docs, pricing, features, etc.)
- **Text Polishing**: LLM-based text cleaning without content generation

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python 3.8+
python --version

# Install required packages
pip install crawl4ai asyncio requests python-dotenv pydantic tenacity psutil
```

### Basic Usage

```bash
# Interactive mode (recommended for first-time users)
python generate-llmstxt-crawl4ai.py

# Direct command line usage
python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com/

# Deep crawling with more pages
python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com/ --max-urls 100
```

### Advanced Usage

```bash
# Use specific AI model
python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com/ \
  --llm-provider ollama \
  --ollama-model deepseek-r1:latest \
  --max-urls 50

# Export as JSON with parallel processing
python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com/ \
  --format json \
  --batch-size 15 \
  --max-urls 100

# Full text only mode (no AI descriptions)
python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com/ \
  --no-full-text \
  --max-urls 200
```

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in your project directory:

```env
# AI Model Configuration
OLLAMA_BASE_URL=http://localhost:11434
GEMINI_API_KEY=your_api_key_here

# Performance Settings
MAX_GEN_OUTPUT_TOKENS=1024
DEFAULT_PARALLEL_WORKERS=3
CACHE_DESCRIPTIONS=true

# Output Configuration
OUTPUT_DIR=./output
```

### Model Selection

#### Ollama (Local Models)
```bash
# List available models
python generate-llmstxt-crawl4ai.py --list-models --llm-provider ollama

# Popular models for different use cases:
# - Fast: phi:2.7b (1.5GB RAM)
# - Balanced: gemma3:4b (3.1GB RAM)
# - High Quality: deepseek-r1:latest (4.4GB RAM)
```

#### Gemini (Cloud Models)
```bash
# List available Gemini models
python generate-llmstxt-crawl4ai.py --list-models --llm-provider gemini

# Set API key
export GEMINI_API_KEY="your_api_key_here"
```

## 📁 Output Files

The tool generates multiple output formats:

### Standard Output (`llms.txt`)
```
# https://docs.crawl4ai.com/ llms.txt
# Generated on 2025-06-26T20:25:44 using ollama:deepseek-r1:latest
# Total pages: 47
# Processing time: 156.3s

- [Quick Start](https://docs.crawl4ai.com/quick-start): Getting started guide for Crawl4AI
- [API Reference](https://docs.crawl4ai.com/api): Complete API documentation
- [Deep Crawling](https://docs.crawl4ai.com/core/deep-crawling): Advanced crawling techniques
```

### Full Content (`llms-full.txt`)
Complete page content with metadata for each discovered page.

### Clean Version (`llms-full-clean.txt`)
Polished content with enhanced formatting and navigation removal.

### Structured Data (`llms.json`, `llms.yaml`)
Machine-readable formats with metadata and statistics.

## 🎛️ Command Line Options

### Basic Options
```bash
--max-urls NUMBER          # Maximum pages to crawl (default: 20)
--llm-provider PROVIDER     # AI provider: ollama, gemini (default: ollama)
--format FORMAT             # Output format: text, json, yaml (default: text)
--output-dir DIR            # Output directory (default: ./output)
```

### AI Model Options
```bash
--ollama-model MODEL        # Specific Ollama model
--ollama-url URL           # Ollama server URL
--gemini-api-key KEY       # Gemini API key
--gemini-model MODEL       # Specific Gemini model
```

### Performance Options
```bash
--batch-size SIZE          # Pages per batch (default: 10)
--max-full-pages NUMBER    # Limit full text pages
--no-parallel-crawl        # Disable parallel processing
--no-cache                 # Disable description caching
```

### Advanced Options
```bash
--verbose                  # Detailed logging
--quiet                    # Minimal output
--no-full-text            # Skip full text generation
--create-env               # Create sample .env file
```

## 🏗️ Architecture

### Deep Crawling System
1. **Initial Discovery**: Extract links from main page using 4 methods
2. **Wave-Based Crawling**: Process pages in waves, discovering new links
3. **Link Prioritization**: Score and sort links by content value
4. **Smart Filtering**: Documentation-aware URL validation

### Content Processing Pipeline
1. **Enhanced Extraction**: Multi-method title and content extraction
2. **Content Cleaning**: Remove navigation, UI elements, and noise
3. **AI Processing**: Use LLM for text polishing (not generation)
4. **Quality Validation**: Anti-hallucination checks and content verification

### Performance Optimizations
- **Parallel Processing**: Configurable worker pools for concurrent operations
- **Intelligent Caching**: Content-based caching with change detection
- **Memory Management**: RAM usage monitoring and optimization
- **Timeout Handling**: Adaptive timeouts with exponential backoff

## 🔧 Troubleshooting

### Common Issues

#### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama server
ollama serve
```

#### Memory Issues
```bash
# Use smaller models for limited RAM
python generate-llmstxt-crawl4ai.py URL --ollama-model phi:2.7b

# Reduce batch size
python generate-llmstxt-crawl4ai.py URL --batch-size 5
```

#### Slow Performance
```bash
# Reduce parallel workers
python generate-llmstxt-crawl4ai.py URL --batch-size 5

# Disable caching for testing
python generate-llmstxt-crawl4ai.py URL --no-cache
```

### Debug Mode
```bash
# Enable verbose logging
python generate-llmstxt-crawl4ai.py URL --verbose

# Check log file
tail -f llmstxt_generator.log
```

## 🧪 Examples

### Documentation Sites
```bash
# Crawl API documentation
python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com/ --max-urls 100

# Python documentation
python generate-llmstxt-crawl4ai.py https://docs.python.org/ --max-urls 150
```

### Company Websites
```bash
# Corporate site with product info
python generate-llmstxt-crawl4ai.py https://openai.com/ --max-urls 50

# Export as structured data
python generate-llmstxt-crawl4ai.py https://anthropic.com/ --format json
```

### Blog and Content Sites
```bash
# Technical blogs
python generate-llmstxt-crawl4ai.py https://blog.openai.com/ --max-urls 75

# News sites
python generate-llmstxt-crawl4ai.py https://techcrunch.com/ --max-urls 100
```

## 🛡️ Features for Enterprise Use

### Security & Privacy
- **Local Processing**: Keep sensitive data on-premises with Ollama
- **No API Lock-in**: Switch between providers without vendor dependency
- **Session Isolation**: Proper cleanup and resource management

### Scalability
- **Configurable Limits**: Adjust crawling depth and breadth
- **Batch Processing**: Handle large sites efficiently
- **Resource Monitoring**: RAM usage warnings and optimization

### Reliability
- **Comprehensive Error Handling**: 5-level retry mechanism
- **Fallback Systems**: Multiple extraction methods for robustness
- **Quality Assurance**: Anti-hallucination validation

## 📈 Performance Benchmarks

### Typical Performance (Local Models)
- **Small Sites (10-20 pages)**: 30-60 seconds
- **Medium Sites (50-100 pages)**: 2-5 minutes  
- **Large Sites (200+ pages)**: 10-20 minutes

### Resource Usage
- **RAM**: 2-8GB depending on model size
- **CPU**: Moderate usage with efficient parallel processing
- **Network**: Optimized with intelligent caching

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-repo/llmstxt-generator
cd llmstxt-generator

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### Code Style
- Follow PEP 8 standards
- Use type hints for all functions
- Include comprehensive docstrings
- Add logging for debugging

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Crawl4AI](https://github.com/unclecode/crawl4ai)**: Powerful web crawling engine
- **[Ollama](https://ollama.ai/)**: Local AI model serving
- **[Google Gemini](https://gemini.google.com/)**: Cloud AI services
- **Community**: Contributors and users providing feedback

---

## 🚀 Getting Started Today

1. **Install Dependencies**: `pip install crawl4ai asyncio requests python-dotenv`
2. **Run Interactive Mode**: `python generate-llmstxt-crawl4ai.py`
3. **Choose Your Site**: Enter any website URL
4. **Select AI Model**: Pick from available Ollama/Gemini models
5. **Generate Content**: Watch as it discovers and processes pages

**Ready to transform any website into AI-friendly content? Start crawling!** 🕷️✨ 
