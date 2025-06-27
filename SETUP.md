# 🛠️ Setup Guide for LLMsGen SDK

This guide helps you set up LLMsGen SDK after cloning from GitHub.

## 📋 Prerequisites

- Python 3.8 or higher
- Git (for cloning)
- Internet connection (for downloading dependencies)

## 🚀 Quick Setup (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/llmsgen.git
   cd llmsgen
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python start.py
   ```

That's it! The application should start and guide you through the setup process.

## 🔧 Alternative Setup Methods

### Method 1: Development Installation
```bash
# Install as editable package
pip install -e .

# Then run with:
python -m llmsgen
# or
llmsgen
```

### Method 2: Direct Execution
```bash
# Run different entry points
python start.py      # Recommended - handles all scenarios
python run.py        # Alternative runner
python __main__.py   # Module execution
```

## 🔑 API Key Configuration

LLMsGen supports multiple AI providers. You'll need at least one API key:

### Google Gemini (Recommended)
1. Visit: https://aistudio.google.com/app/apikey
2. Create a new API key
3. The application will prompt you to enter it on first run

### Local Models (Ollama)
1. Install Ollama: https://ollama.ai/
2. Start Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3.2`

### Other Providers (Optional)
- **OpenAI:** Get API key from https://platform.openai.com/
- **Anthropic:** Get API key from https://console.anthropic.com/

## 📁 Project Structure

```
llmsgen/
├── start.py           # Main entry point (use this!)
├── run.py             # Alternative runner
├── __main__.py        # Module runner
├── requirements.txt   # Dependencies
├── setup.py          # Package installation
├── README.md         # Main documentation
├── SETUP.md          # This file
├── models/           # AI model management
├── generator/        # Core generation logic
├── crawler/          # Web crawling functionality
├── utils/            # Utility functions
└── scripts/          # Additional scripts
```

## 🧪 Testing Your Setup

1. **Check dependencies:**
   ```bash
   python -c "import requests, bs4, yaml, dotenv; print('✅ Dependencies OK')"
   ```

2. **Test import:**
   ```bash
   python -c "from models.client import ModelManager; print('✅ Imports OK')"
   ```

3. **Run application:**
   ```bash
   python start.py
   ```

## 🐛 Troubleshooting

### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# If still failing, install individually:
pip install requests beautifulsoup4 lxml python-dotenv pyyaml tenacity tqdm psutil
```

### Playwright Issues
```bash
# Install Playwright browsers
playwright install chromium
```

### Permission Issues
```bash
# Use user installation
pip install --user -r requirements.txt
```

### Windows-Specific Issues
```bash
# If encoding issues occur
chcp 65001
```

## 🎯 Usage Examples

### Basic Website Crawling
1. Run: `python start.py`
2. Enter URL: `https://docs.python.org`
3. Select crawling mode: `systematic`
4. Choose AI model (Gemini recommended)
5. Wait for generation to complete

### Advanced Configuration
- Modify `config.py` for default settings
- Create `.env` file for API keys
- Adjust worker count for your system

## 📞 Getting Help

If you encounter issues:

1. **Check logs:** Look for `llmstxt_generator.log`
2. **Verify setup:** Run the test commands above
3. **Update dependencies:** `pip install -r requirements.txt --upgrade`
4. **Report issues:** Create a GitHub issue with error details

## 🌟 Next Steps

After successful setup:
- Read the full README.md for advanced features
- Explore the API documentation
- Try different crawling strategies
- Experiment with different AI models

Happy crawling! 🕷️✨ 