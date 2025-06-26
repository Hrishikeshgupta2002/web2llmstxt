# 🤖 LLMs.txt Generator with Crawl4AI

An advanced Python tool that generates **specification-compliant** [llms.txt](https://llmstxt.org/) files using Crawl4AI for comprehensive website crawling and AI-powered content analysis.

## ✨ Features

### 🎯 **Official llms.txt Specification Compliance**
- ✅ **H1 Header** with clean site name extraction
- ✅ **Blockquote Summary** with intelligent site type detection
- ✅ **Categorized H2 Sections** (Documentation, Products & Services, Resources, API & Technical)
- ✅ **Proper Link Format**: `- [title](url): description`

### 🚀 **Advanced Crawling Capabilities**
- **Comprehensive Multi-Level Crawling** with intelligent link discovery (enabled by default)
- **Smart URL Scoring** based on content relevance and importance
- **Depth-Limited Exploration** (up to 5 levels deep)
- **Intelligent Safety Limits** to prevent infinite crawling (auto-calculated)
- **Domain-Specific Filtering** with content type detection
- **Enhanced Coverage**: Up to 50+ pages per crawl with quality filtering

### 🤖 **AI-Powered Content Processing**
- **Multiple AI Models**: Gemini (Cloud) and Ollama (Local)
- **Intelligent Text Cleaning** with HTML/markdown removal
- **Content Summarization** with hallucination detection
- **Smart Fallbacks** for reliable content extraction
- **Batch Processing** with parallel workers

### 📄 **Multiple Output Formats**
- **Standard llms.txt**: Specification-compliant summary
- **Full Content**: Complete crawled content
- **Clean Version**: Processed and cleaned text
- **JSON/YAML**: Structured data export

## 🛠 Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Setup Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv crawl4ai_env

# Activate (Windows)
crawl4ai_env\Scripts\activate

# Activate (macOS/Linux)
source crawl4ai_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### AI Model Configuration

#### Option 1: Gemini (Cloud) - Recommended
1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set environment variable:
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# macOS/Linux
export GEMINI_API_KEY=your_api_key_here
```

#### Option 2: Ollama (Local)
1. Install [Ollama](https://ollama.ai/)
2. Download models:
```bash
ollama pull gemma3:latest
ollama pull phi:2.7b
```

## 🚀 Quick Start

### Basic Usage
```bash
# Generate llms.txt for a website
python generate-llmstxt-crawl4ai.py https://example.com

# Specify maximum pages
python generate-llmstxt-crawl4ai.py https://example.com --max-pages 20

# Full content extraction
python generate-llmstxt-crawl4ai.py https://example.com --full-text-only
```

### Advanced Examples
```bash
# Comprehensive crawling with custom safety limit
python generate-llmstxt-crawl4ai.py https://docs.example.com \
    --max-pages 15 \
    --safety-limit 100 \
    --crawl-strategy systematic

# Export as JSON with parallel processing
python generate-llmstxt-crawl4ai.py https://api.example.com \
    --format json \
    --parallel-workers 5 \
    --batch-size 15

# Generate only text files without AI descriptions
python generate-llmstxt-crawl4ai.py https://example.com \
    --full-text-only \
    --max-pages 25
```

## 📖 Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `url` | Required | Target website URL |
| `--max-pages` | 50 | Maximum pages to crawl |
| `--format` | text | Output format (text/json/yaml) |
| `--full-text-only` | False | Skip AI descriptions, full content only |
| `--parallel-workers` | 3 | Number of parallel workers |
| `--batch-size` | 10 | AI processing batch size |
| `--crawl-strategy` | systematic | Crawling strategy (systematic/bestfirst) |
| `--safety-limit` | auto | Maximum pages safety limit |
| `--include-full-text` | True | Generate full content files |

## 📂 Output Files

The tool generates multiple files in the `output/` directory:

### 📄 **domain-llms.txt** (Main File)
```markdown
# Site Name

> Intelligent site summary based on content analysis.

Generated from X pages on YYYY-MM-DD using automated crawling.

## Documentation

- [API Reference](https://example.com/api): Complete API documentation with endpoints, parameters, and examples...
- [User Guide](https://example.com/guide): Step-by-step tutorials and best practices...

## Products & Services

- [Pricing Plans](https://example.com/pricing): Flexible pricing options for individuals and enterprises...
```

### 📄 **domain-llms-full.txt** (Complete Content)
Raw crawled content with AI-generated descriptions for each page.

### 📄 **domain-llms-full-clean.txt** (Processed Content)
Cleaned and formatted version without HTML/markdown artifacts.

## ⚙️ Configuration

### Environment Variables
```bash
# AI Configuration
GEMINI_API_KEY=your_gemini_api_key
OLLAMA_BASE_URL=http://localhost:11434  # For custom Ollama instance

# Crawling Configuration
CRAWL4AI_MAX_TIMEOUT=30
CRAWL4AI_USER_AGENT="LLMsTxtGenerator/1.0"
```

### Model Selection
The tool automatically detects available models and prompts for selection:
```
📋 Available AI Models:
🌟 Gemini Models (Cloud):
  1. 🟢 🌟 Gemini 2.5 Flash (Experimental)
  2. 🟢 💫 Gemini 1.5 Flash
  
🤖 Ollama Models (Local):
  3. 🟢 🔬 Phi 2.7B (5.4GB RAM)
  4. 🟢 💎 Gemma 4B (8.0GB RAM)

Select a model (1-4):
```

## 🎯 Use Cases

### 📚 **Documentation Sites**
```bash
python generate-llmstxt-crawl4ai.py https://docs.python.org --max-pages 30
```
Perfect for creating comprehensive documentation indexes.

### 🏢 **Corporate Websites**
```bash
python generate-llmstxt-crawl4ai.py https://company.com --max-pages 20 --format json
```
Generate structured company information for analysis.

### 🛒 **E-commerce Platforms**
```bash
python generate-llmstxt-crawl4ai.py https://shop.example.com --safety-limit 200
```
Catalog products and services with intelligent categorization.

### 🔧 **API Documentation**
```bash
python generate-llmstxt-crawl4ai.py https://api.example.com --crawl-strategy systematic
```
Create comprehensive API reference summaries.

## 🏗 Architecture

### Crawling Strategy
1. **Discovery Phase**: Crawl main page and extract all internal links
2. **Scoring Phase**: Rate URLs based on content relevance and importance
3. **Multi-Level Crawling**: Systematically crawl by depth with quality filtering
4. **Content Processing**: AI-powered text extraction and summarization
5. **Categorization**: Intelligent grouping into logical sections

### Content Processing Pipeline
```
Raw HTML → Content Extraction → Text Cleaning → AI Summarization → Categorization → Output Generation
```

## 🔧 Troubleshooting

### Common Issues

**❌ "No pages could be crawled"**
- Check if the website is accessible
- Verify URL format (include https://)
- Try increasing `--safety-limit`

**❌ "Model not available"**
- For Gemini: Verify `GEMINI_API_KEY` is set
- For Ollama: Ensure Ollama is running (`ollama serve`)

**❌ "Low content quality"**
- Increase `--max-pages` for better coverage
- Use `--crawl-strategy systematic` for deeper analysis
- Check if site has anti-bot protection

### Performance Optimization

**🚀 Faster Processing**
```bash
# Increase parallel workers
--parallel-workers 8 --batch-size 20

# Use local models for consistent performance
# (Requires Ollama setup)
```

**💾 Memory Management**
```bash
# For large sites, use smaller batches
--batch-size 5 --max-pages 100
```

## 📊 Example Output

### Input
```bash
python generate-llmstxt-crawl4ai.py https://alternates.ai --max-pages 10
```

### Generated llms.txt
```markdown
# Alternates

> Software and service marketplace with pricing information and tool comparisons.

Generated from 50 pages on 2025-06-26 using automated crawling.

## Products & Services

- [AI Agents Marketplace](https://alternates.ai/agents): Your workspace to explore alternates and orchestrate AI Agents — from idea to execution. Explore AI AgentsShowing all 479 agents...
- [Pricing Plans](https://alternates.ai/pricing): Compare different subscription tiers and features for individuals and enterprises...

## Resources

- [Knowledge Hub](https://alternates.ai/knowledge-hub): Comprehensive documentation and learning resources for AI automation...
- [Blog](https://alternates.ai/blog): Latest insights on AI agents, automation trends, and industry developments...

## API & Technical

- [MCP Servers](https://alternates.ai/mcp-servers): Meet MCP Servers — product execution environments built for multi-agent automation...
- [Developer API](https://alternates.ai/api): Complete API documentation with endpoints, authentication, and code examples...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [llmstxt.org](https://llmstxt.org/) - Official llms.txt specification
- [Crawl4AI](https://github.com/unclecode/crawl4ai) - Advanced web crawling framework
- [Ollama](https://ollama.ai/) - Local AI model hosting

## 🙏 Acknowledgments

- Built with [Crawl4AI](https://github.com/unclecode/crawl4ai) for robust web crawling
- Supports [Gemini AI](https://ai.google.dev/) and [Ollama](https://ollama.ai/) models
- Follows [llms.txt specification](https://llmstxt.org/) for compatibility

---

**⭐ Star this repo if it helps you create better llms.txt files!** 
