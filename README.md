# Crawl4AI LLMs.txt Generator

A powerful Python script that generates `llms.txt` and `llms-full.txt` files for any website using **Crawl4AI** and either **Ollama** (local) or **Gemini** (cloud) AI models.

This is an enhanced, reverse-engineered version based on [Firecrawl LLMs.txt Generator](https://github.com/mendableai/create-llmstxt-py) with significant improvements and modern AI integration.

## 🚀 Features

### 🔧 Enhanced AI Integration
- **Ollama Support**: Full local LLM support with automatic model detection
- **Gemini Support**: Google's latest Gemini models with advanced configuration
- **Smart Model Selection**: Automatically detects and uses the best available model
- **Model Listing**: View all available models for your chosen provider

### 🕸️ Advanced Web Crawling
- **Crawl4AI Powered**: Uses the latest Crawl4AI for robust, browser-based scraping
- **Deep Crawling**: Automatic website discovery with BFS (Breadth-First Search)
- **JavaScript Support**: Full support for dynamic, JavaScript-heavy websites
- **Smart Content Extraction**: AI-optimized markdown generation

### 🎯 Professional Output
- **Clean Markdown**: LLM-ready content with intelligent filtering
- **Structured Data**: JSON-based title and description generation
- **Batch Processing**: Efficient concurrent processing of multiple URLs
- **Error Handling**: Robust error recovery and logging

## 📋 Requirements

- Python 3.8+
- Crawl4AI v0.6.0+
- Either Ollama (local) or Gemini API access

## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd crawl4ai-llmstxt-generator
pip install -r requirements-crawl4ai.txt
```

### 2. Install Crawl4AI

```bash
# Install Crawl4AI
pip install crawl4ai

# Setup browsers (required)
crawl4ai-setup

# Verify installation
crawl4ai-doctor
```

### 3. Setup AI Provider

#### Option A: Ollama (Local, Free)

```bash
# Install Ollama
# Visit https://ollama.ai for installation instructions

# Pull a model (choose one or more)
ollama pull llama3.2          # Recommended: Latest Llama
ollama pull qwen2.5:7b        # Alternative: Qwen 2.5
ollama pull gemma2            # Alternative: Google Gemma
ollama pull mistral           # Alternative: Mistral
```

#### Option B: Gemini (Cloud, Paid)

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set your API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
# Or create a .env file (see env.example.crawl4ai)
```

## 🚀 Usage

### Basic Examples

```bash
# Using Ollama (auto-detects best model)
python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com

# Using Gemini
python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com --llm-provider gemini

# Process more URLs
python generate-llmstxt-crawl4ai.py https://python.org --max-urls 50
```

### Advanced Usage

```bash
# List available models
python generate-llmstxt-crawl4ai.py --list-models --llm-provider ollama
python generate-llmstxt-crawl4ai.py --list-models --llm-provider gemini

# Specify exact model
python generate-llmstxt-crawl4ai.py https://github.com \
  --llm-provider ollama \
  --ollama-model "qwen2.5:7b"

# Custom Gemini model
python generate-llmstxt-crawl4ai.py https://github.com \
  --llm-provider gemini \
  --gemini-model "gemini-1.5-pro"

# Custom output directory
python generate-llmstxt-crawl4ai.py https://fastapi.tiangolo.com \
  --output-dir ./crawled-sites \
  --max-urls 30 \
  --verbose

# Only generate index (no full text)
python generate-llmstxt-crawl4ai.py https://example.com --no-full-text
```

## 🧠 Supported Models

### Ollama Models (Local)
- **llama3.2** ⭐ (Recommended)
- **llama3.1**
- **qwen2.5** series (1.5B, 7B, 14B)
- **mistral**
- **mixtral**
- **codellama**
- **phi3**
- **gemma2**
- *And many more...*

### Gemini Models (Cloud)
- **gemini-1.5-flash** ⭐ (Recommended - fast & affordable)
- **gemini-1.5-flash-8b**
- **gemini-1.5-pro** (More capable, higher cost)
- **gemini-2.0-flash-exp** (Latest experimental)
- **gemini-exp-1114**
- **gemini-exp-1121**

## 📊 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `url` | Website URL to process | Required |
| `--max-urls` | Maximum URLs to process | 20 |
| `--output-dir` | Output directory | Current directory |
| `--llm-provider` | AI provider (ollama/gemini) | ollama |
| `--ollama-model` | Specific Ollama model | Auto-detect |
| `--ollama-url` | Ollama server URL | http://localhost:11434 |
| `--gemini-api-key` | Gemini API key | From GEMINI_API_KEY env |
| `--gemini-model` | Specific Gemini model | gemini-1.5-flash |
| `--no-full-text` | Skip full text generation | False |
| `--verbose` | Enable detailed logging | False |
| `--list-models` | List available models | False |

## 📁 Output Format

### llms.txt (Index)
```
# https://example.com llms.txt

- [Getting Started Guide](https://example.com/guide): Complete tutorial for new users getting started
- [API Documentation](https://example.com/api): Comprehensive reference for developers using the API
- [Advanced Features](https://example.com/advanced): In-depth coverage of advanced functionality and configuration
```

### llms-full.txt (Complete Content)
```
# https://example.com llms-full.txt

<|crawl4ai-page-1-lllmstxt|>
## Getting Started Guide
Full markdown content of the getting started guide...

<|crawl4ai-page-2-lllmstxt|>
## API Documentation
Complete API documentation in markdown format...
```

## 🔧 Configuration

Create a `.env` file based on `env.example.crawl4ai`:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Ollama Configuration (optional)
OLLAMA_BASE_URL=http://localhost:11434

# Output Configuration
OUTPUT_DIR=./output
```

## 🎯 GitHub Extensions and Ollama

**Yes!** This tool works excellently with GitHub/VS Code extensions:

### Recommended Extensions:
1. **Continue.dev** - Works with both Ollama and cloud models
2. **GitHub Copilot** - Can be supplemented with local Ollama
3. **Codeium** - Supports various model providers
4. **Ollama Extension** - Direct VS Code integration with Ollama

### Integration Tips:
- Use `ollama list` to see your installed models
- Most extensions auto-detect local Ollama installation
- The generated `llms.txt` files work perfectly as context for any LLM

## 🚀 Performance Tips

### For Ollama
- **Model Size**: Larger models (7B+) give better results but need more RAM
- **GPU**: Use CUDA/Metal for faster inference
- **Multiple Models**: Install 2-3 different models for comparison

### For Gemini
- **Rate Limits**: Gemini Flash has generous limits for most use cases
- **Cost**: Flash is very affordable (~$0.075/1M tokens)
- **Quality**: Pro models give better results for complex content

### General
- **Batch Size**: Increase `--max-urls` for comprehensive site mapping
- **Parallel Processing**: Script automatically handles concurrent requests
- **Caching**: Crawl4AI automatically caches results for efficiency

## 🔍 Troubleshooting

### Common Issues

#### Ollama Connection Failed
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

#### Crawl4AI Browser Issues
```bash
# Reinstall browsers
crawl4ai-setup

# Manual browser installation
python -m playwright install chromium --with-deps
```

#### Permission/Path Issues
```bash
# Check Python path
which python

# Install in virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements-crawl4ai.txt
```

## 🆚 Comparison with Original

| Feature | Original (Firecrawl) | This Version (Crawl4AI) |
|---------|---------------------|------------------------|
| Web Scraping | Firecrawl API | Crawl4AI (Local) |
| AI Provider | OpenAI only | Ollama + Gemini |
| Cost | API costs for both | Free (Ollama) or Low (Gemini) |
| Models | GPT-4 only | 10+ model options |
| Privacy | Cloud-based | Can be fully local |
| Setup | API keys required | Works offline with Ollama |
| Performance | Network dependent | Local processing available |

## 📄 License

MIT License - Feel free to use, modify, and distribute.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 🙏 Credits

- Powered by [Crawl4AI](https://github.com/unclecode/crawl4ai)
- Enhanced with [Ollama](https://ollama.ai) and [Google Gemini](https://ai.google.dev)

---

**Happy Crawling!** 🕸️✨ 
