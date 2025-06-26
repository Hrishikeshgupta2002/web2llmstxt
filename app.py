#!/usr/bin/env python3
"""
Generate llms.txt and llms-full.txt files for a website using Crawl4AI and Ollama/Gemini.

This script:
1. Maps all URLs from a website using Crawl4AI's built-in crawling
2. Scrapes each URL to get the content
3. Uses Ollama (local) or Gemini (cloud) to generate titles and descriptions
4. Creates llms.txt (list of pages with descriptions) and llms-full.txt (full content)

Features:
- Enhanced model selection inspired by Cline VS Code extension
- Automatic model detection for both Ollama and Gemini
- Support for various Gemini models (gemini-pro, gemini-1.5-pro, etc.)
- Deep crawling capabilities with smart content extraction
- Browser-based content extraction with modern web support
- Model switching during runtime with visual feedback
- Configuration presets for different use cases
- OpenWebUI-style model management with status indicators
- Export formats: text, JSON, YAML
- Parallel crawling support
- Smart caching with URL hashing
- RAM usage warnings for large models
"""

import os
import sys
import json
import yaml
import time
import argparse
import logging
import re
import asyncio
import hashlib
import psutil
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import requests
from dotenv import load_dotenv, set_key
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()

# Configuration constants
MAX_GEN_OUTPUT_TOKENS = int(os.getenv('MAX_GEN_OUTPUT_TOKENS', '1024'))
CACHE_DESCRIPTIONS = os.getenv('CACHE_DESCRIPTIONS', 'true').lower() == 'true'
DEFAULT_PARALLEL_WORKERS = int(os.getenv('DEFAULT_PARALLEL_WORKERS', '3'))

# Local model optimization constants
LOCAL_MODEL_BATCH_SIZE = 3  # Smaller batches for local models
LOCAL_MODEL_TIMEOUT = 180   # Longer timeout for local inference
LOCAL_MODEL_RETRY_DELAY = 2 # Longer delay between retries for local models

# Configure logging with enhanced formatting
# The 'encoding' parameter for FileHandler and reconfiguring stdout fixes Unicode errors on Windows.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llmstxt_generator.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Attempt to set stdout encoding to UTF-8, which is necessary for emojis in Windows terminals.
if sys.platform == "win32":
    try:
        # This is the most reliable way in Python 3.7+
        sys.stdout.reconfigure(encoding='utf-8')
    except (TypeError, AttributeError):
        # This can fail in non-interactive shells or older Python versions.
        # The script will still run, but emojis might not display correctly in the console.
        logger.info("Could not reconfigure stdout for UTF-8. Emojis may not render correctly.")

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for AI models with enhanced metadata"""
    provider: str
    model_id: str
    display_name: str
    description: str
    max_tokens: int = 8192
    temperature: float = 0.7
    supports_vision: bool = False
    cost_per_1k_tokens: float = 0.0
    status: str = "unknown"  # online, offline, unknown, available
    tags: List[str] = field(default_factory=list)  # Fixed: ensure tags is always a list
    pulls: str = ""
    size_info: str = ""
    is_available_remote: bool = False
    estimated_ram_gb: float = 0.0  # Estimated RAM requirement

class ModelManager:
    """Manages available AI models similar to Cline's model selection with enhanced features"""
    
    def __init__(self):
        self.available_models = {}
        self.current_model = None
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.model_cache_file = os.path.join(os.getenv('OUTPUT_DIR', './output'), 'model_cache.json')
        self.ollama_library_cache = {}
        self.description_cache = {}
        self._load_description_cache()
        
    def _load_description_cache(self):
        """Load cached descriptions to avoid regenerating unchanged summaries"""
        cache_file = os.path.join(os.getenv('OUTPUT_DIR', './output'), 'description_cache.json')
        if CACHE_DESCRIPTIONS and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.description_cache = json.load(f)
                logger.info(f"Loaded {len(self.description_cache)} cached descriptions")
            except Exception as e:
                logger.warning(f"Could not load description cache: {e}")
                
    def _save_description_cache(self):
        """Save description cache"""
        if not CACHE_DESCRIPTIONS:
            return
        cache_file = os.path.join(os.getenv('OUTPUT_DIR', './output'), 'description_cache.json')
        try:
            os.makedirs(os.getenv('OUTPUT_DIR', './output'), exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.description_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save description cache: {e}")
    
    def _get_content_hash(self, title: str, content: str) -> str:
        """Generate hash for content to detect changes"""
        content_str = f"{title}|||{content[:1000]}"  # Use first 1000 chars for hash
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def check_cached_description(self, url: str, title: str, content: str) -> Optional[str]:
        """Check if we have a cached description for this content"""
        if not CACHE_DESCRIPTIONS:
            return None
        content_hash = self._get_content_hash(title, content)
        cache_key = f"{url}:{content_hash}"
        return self.description_cache.get(cache_key)
    
    def cache_description(self, url: str, title: str, content: str, description: str):
        """Cache a generated description"""
        if not CACHE_DESCRIPTIONS:
            return
        content_hash = self._get_content_hash(title, content)
        cache_key = f"{url}:{content_hash}"
        self.description_cache[cache_key] = description
        self._save_description_cache()
        
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def warm_up_model(self, model_id: str) -> bool:
        """Warm up a local model by sending a simple test prompt"""
        if not self.check_ollama_status():
            logger.warning("⚠️  Ollama is not running. Please start Ollama with 'ollama serve'")
            return False
            
        try:
            logger.info(f"🔥 Warming up model {model_id}...")
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model_id,
                    "prompt": "Hello, respond with just 'Ready'",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Model {model_id} is warmed up and ready")
                return True
            else:
                logger.warning(f"⚠️  Model {model_id} responded with status {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error(f"❌ Model {model_id} warm-up timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to warm up model {model_id}: {e}")
            return False
    
    def check_model_loaded(self, model_id: str) -> bool:
        """Check if a specific model is loaded in Ollama"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/ps", timeout=5)
            if response.status_code == 200:
                running_models = response.json().get('models', [])
                for model in running_models:
                    if model.get('name') == model_id:
                        logger.info(f"✅ Model {model_id} is already loaded")
                        return True
                logger.info(f"ℹ️  Model {model_id} is not loaded yet")
                return False
            return False
        except Exception as e:
            logger.debug(f"Could not check loaded models: {e}")
            return False
    
    def _get_model_family_info(self, model_name: str) -> Dict[str, Any]:
        """Extract model family information"""
        base_name = model_name.split(':')[0].lower()
        
        model_families = {
            'deepseek-r1': {
                'display': 'DeepSeek R1',
                'category': 'reasoning',
                'emoji': '🧠',
                'description': 'Advanced reasoning model',
                'estimated_ram': 8.0
            },
            'deepseek': {
                'display': 'DeepSeek',
                'category': 'general',
                'emoji': '🤖',
                'description': 'Efficient general model',
                'estimated_ram': 4.0
            },
            'gemma': {
                'display': 'Gemma',
                'category': 'general',
                'emoji': '💎',
                'description': 'Google open model',
                'estimated_ram': 3.0
            },
            'qwen': {
                'display': 'Qwen',
                'category': 'general',
                'emoji': '🚀',
                'description': 'Alibaba multilingual model',
                'estimated_ram': 4.0
            },
            'llama': {
                'display': 'Llama',
                'category': 'general',
                'emoji': '🦙',
                'description': 'Meta open model',
                'estimated_ram': 6.0
            },
            'codestral': {
                'display': 'Codestral',
                'category': 'code',
                'emoji': '💻',
                'description': 'Mistral coding model',
                'estimated_ram': 5.0
            },
            'codegemma': {
                'display': 'CodeGemma',
                'category': 'code',
                'emoji': '💻',
                'description': 'Google coding model',
                'estimated_ram': 4.0
            },
            'phi': {
                'display': 'Phi',
                'category': 'general',
                'emoji': '🔬',
                'description': 'Microsoft efficient model',
                'estimated_ram': 2.0
            },
            'mistral': {
                'display': 'Mistral',
                'category': 'general',
                'emoji': '🌪️',
                'description': 'Mistral AI model',
                'estimated_ram': 4.0
            },
            'granite': {
                'display': 'Granite',
                'category': 'general',
                'emoji': '🪨',
                'description': 'IBM enterprise model',
                'estimated_ram': 5.0
            },
            'solar': {
                'display': 'Solar',
                'category': 'general',
                'emoji': '☀️',
                'description': 'Upstage efficient model',
                'estimated_ram': 4.0
            }
        }
        
        for family_key, family_info in model_families.items():
            if family_key in base_name:
                return family_info
                
        return {
            'display': base_name.title(),
            'category': 'general',
            'emoji': '🤖',
            'description': f'{base_name.title()} model',
            'estimated_ram': 4.0
        }
    
    def _build_display_name(self, model_name: str, family_info: Dict[str, Any]) -> str:
        """Build display name for the model"""
        parts = model_name.split(':')
        tag = parts[1] if len(parts) > 1 else 'latest'
        
        size_indicator = self._extract_size_indicator(tag)
        emoji = family_info['emoji']
        display_base = family_info['display']
        
        if size_indicator:
            return f"{emoji} {display_base} {size_indicator}"
        else:
            return f"{emoji} {display_base} ({tag})"
    
    def _format_model_name_enhanced(self, model_name: str) -> tuple[str, dict]:
        """Enhanced model name formatting with better recognition"""
        family_info = self._get_model_family_info(model_name)
        display_name = self._build_display_name(model_name, family_info)
        
        return display_name, family_info
    
    def _detect_reasoning_model(self, model_name: str) -> bool:
        """Detect if this is a reasoning model"""
        reasoning_patterns = [
            'r1', 'reasoning', 'think', 'cot', 'chain-of-thought'
        ]
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in reasoning_patterns)
    
    def _extract_model_size(self, model_name: str) -> str:
        """Extract model size information"""
        # Size patterns for different formats
        size_patterns = [
            r'(\d+(?:\.\d+)?)[Bb]',  # 7B, 13B, 1.5B, etc.
            r'(\d+(?:\.\d+)?)[Mm]',  # 7M, 13M, etc.
            r'(\d+(?:\.\d+)?)[Kk]',  # 7K, 13K, etc.
            r'(\d+(?:\.\d+)?)-?[Bb]illion',  # 7-billion, 13billion
            r'(\d+(?:\.\d+)?)-?[Mm]illion',   # 7-million, 13million
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, model_name, re.IGNORECASE)
            if match:
                size_num = match.group(1)
                size_unit = model_name[match.start():match.end()]
                
                # Normalize the unit
                if 'b' in size_unit.lower() or 'billion' in size_unit.lower():
                    return f"{size_num}B"
                elif 'm' in size_unit.lower() or 'million' in size_unit.lower():
                    return f"{size_num}M"
                elif 'k' in size_unit.lower():
                    return f"{size_num}K"
        
        return ""
    
    def _extract_size_indicator(self, tag: str) -> str:
        """Extract size from tag (e.g., '7b', '13b-chat', 'instruct')"""
        # Handle common size patterns in tags
        size_patterns = [
            r'(\d+(?:\.\d+)?)[Bb]',     # 7b, 13b, 1.5b
            r'(\d+(?:\.\d+)?)[Mm]',     # 7m, 13m  
            r'(\d+(?:\.\d+)?)[Kk]',     # 7k, 13k
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, tag)
            if match:
                return match.group(0).upper()
        
        # Look for special indicators
        if 'tiny' in tag.lower():
            return 'Tiny'
        elif 'small' in tag.lower():
            return 'Small'
        elif 'medium' in tag.lower():
            return 'Medium'
        elif 'large' in tag.lower():
            return 'Large'
        elif 'xl' in tag.lower():
            return 'XL'
        elif 'instruct' in tag.lower():
            return 'Instruct'
        elif 'chat' in tag.lower():
            return 'Chat'
        elif 'code' in tag.lower():
            return 'Code'
            
        return ""
    
    def _estimate_ram_requirements(self, model_name: str, family_info: Dict[str, Any]) -> float:
        """Estimate RAM requirements for model"""
        base_ram = family_info.get('estimated_ram', 4.0)
        
        # Adjust based on model size
        size_str = self._extract_model_size(model_name)
        if size_str:
            if 'B' in size_str:  # Billion parameters
                size_num = float(re.findall(r'\d+(?:\.\d+)?', size_str)[0])
                # Rough estimate: 1B params ≈ 2GB RAM for inference
                return max(size_num * 2, base_ram)
            elif 'M' in size_str:  # Million parameters
                size_num = float(re.findall(r'\d+(?:\.\d+)?', size_str)[0])
                return max(size_num * 0.002, base_ram)  # 1M params ≈ 2MB
        
        return base_ram
    
    def _check_ram_availability(self, required_ram_gb: float) -> Tuple[bool, str]:
        """Check if system has enough RAM for the model"""
        try:
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            
            if required_ram_gb > available_ram_gb:
                return False, f"⚠️  Model requires ~{required_ram_gb:.1f}GB RAM, but only {available_ram_gb:.1f}GB available (Total: {total_ram_gb:.1f}GB)"
            elif required_ram_gb > available_ram_gb * 0.8:  # Warn if using >80% of available RAM
                return True, f"⚠️  Model requires ~{required_ram_gb:.1f}GB RAM, using {(required_ram_gb/available_ram_gb)*100:.0f}% of available RAM"
            else:
                return True, f"✅ Model requires ~{required_ram_gb:.1f}GB RAM, {available_ram_gb:.1f}GB available"
        except Exception as e:
            return True, f"❓ Could not check RAM: {e}"
    
    def _get_status_indicator(self, status: str) -> str:
        """Get status indicator emoji"""
        indicators = {
            'online': '🟢',
            'available': '🟢', 
            'offline': '🔴',
            'unknown': '🟡'
        }
        return indicators.get(status.lower(), '🟡')
    
    def list_models(self) -> Dict[str, ModelConfig]:
        """List available models from all providers"""
        models = {}
        
        # Always try to get Ollama models (local models)
        ollama_models = self._list_ollama_models()
        models.update(ollama_models)
        
        # Also get Gemini models if API key is available
        if self.gemini_api_key:
            gemini_models = self._list_gemini_models()
            models.update(gemini_models)
        
        return models
    
    def _list_ollama_models(self) -> Dict[str, ModelConfig]:
        """List available Ollama models"""
        models = {}
        
        if not self.check_ollama_status():
            logger.debug("Ollama is not running - no local models available")
            return models
        
        try:
            # Get list of installed models
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            ollama_data = response.json()
            model_list = ollama_data.get('models', [])
            
            for model_info in model_list:
                model_name = model_info.get('name', '')
                if not model_name:
                    continue
                
                # Enhanced model processing
                display_name, family_info = self._format_model_name_enhanced(model_name)
                estimated_ram = self._estimate_ram_requirements(model_name, family_info)
                
                # Check if model is currently loaded
                is_loaded = self.check_model_loaded(model_name)
                status = 'online' if is_loaded else 'available'
                
                model_config = ModelConfig(
                    provider='ollama',
                    model_id=model_name,
                    display_name=display_name,
                    description=family_info.get('description', ''),
                    max_tokens=8192,
                    temperature=0.7,
                    status=status,
                    tags=model_info.get('details', {}).get('families', []),
                    pulls=model_info.get('details', {}).get('quantization_level', ''),
                    size_info=f"{model_info.get('size', 0) / (1024**3):.1f}GB",
                    estimated_ram_gb=estimated_ram
                )
                
                models[model_name] = model_config
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Ollama models: {e}")
        except Exception as e:
            logger.error(f"Error processing Ollama models: {e}")
            
        return models
    
    def _list_gemini_models(self) -> Dict[str, ModelConfig]:
        """List available Gemini models"""
        models = {}
        
        if not self.gemini_api_key:
            logger.warning("⚠️  No Gemini API key provided")
            return models
        
        # Predefined Gemini models (since we can't easily list them via API)
        gemini_models = [
            {
                'id': 'gemini-2.5-flash-exp',
                'name': '🌟 Gemini 2.5 Flash (Experimental)',
                'description': 'Latest cutting-edge model with enhanced capabilities',
                'max_tokens': 8192,
                'cost_per_1k': 0.075
            },
            {
                'id': 'gemini-2.0-flash-exp',
                'name': '⚡ Gemini 2.0 Flash (Experimental)',
                'description': 'Latest experimental model with multimodal capabilities',
                'max_tokens': 8192,
                'cost_per_1k': 0.075
            },
            {
                'id': 'gemini-exp-1206',
                'name': '🧪 Gemini Experimental 1206',
                'description': 'Cutting-edge experimental model',
                'max_tokens': 8192,
                'cost_per_1k': 0.075
            },
            {
                'id': 'gemini-1.5-flash',
                'name': '💫 Gemini 1.5 Flash',
                'description': 'Fast and efficient model for most tasks',
                'max_tokens': 8192,
                'cost_per_1k': 0.075
            },
            {
                'id': 'gemini-1.5-flash-8b',
                'name': '💨 Gemini 1.5 Flash 8B',
                'description': 'Lightweight and fast model for high-volume tasks',
                'max_tokens': 8192,
                'cost_per_1k': 0.0375
            },
            {
                'id': 'gemini-1.5-pro',
                'name': '🚀 Gemini 1.5 Pro', 
                'description': 'Most capable model for complex reasoning',
                'max_tokens': 32768,
                'cost_per_1k': 3.5
            },
            {
                'id': 'gemini-1.0-pro',
                'name': '⚡ Gemini 1.0 Pro',
                'description': 'Reliable model for general use',
                'max_tokens': 32768,
                'cost_per_1k': 0.5
            }
        ]
        
        for model_info in gemini_models:
            model_config = ModelConfig(
                provider='gemini',
                model_id=model_info['id'],
                display_name=model_info['name'],
                description=model_info['description'],
                max_tokens=model_info['max_tokens'],
                temperature=0.7,
                cost_per_1k_tokens=model_info['cost_per_1k'],
                status='available',
                is_available_remote=True,
                estimated_ram_gb=0.0  # Cloud model
            )
            models[model_info['id']] = model_config
            
        return models

class AIClient:
    """Enhanced AI client with retry logic and configurable limits"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.gemini_client = None
        # Add session for connection pooling with local models
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        # Performance tracking for local models
        self.local_model_stats = {
            'total_requests': 0,
            'total_tokens_generated': 0,
            'total_time_seconds': 0,
            'timeouts': 0,
            'errors': 0
        }
        
    def _get_gemini_client(self):
        """Get or create Gemini client"""
        if self.gemini_client is None:
            try:
                import google.generativeai as genai
                api_key = self.model_manager.gemini_api_key
                if not api_key:
                    raise ValueError("Gemini API key not found")
                genai.configure(api_key=api_key)
                self.gemini_client = genai
                logger.debug("Gemini client initialized successfully")
            except ImportError:
                raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Gemini client: {e}")
        return self.gemini_client
    
    def _get_adaptive_timeout(self, model_config: ModelConfig) -> int:
        """Get adaptive timeout based on model type and estimated RAM requirements"""
        if model_config.provider == 'ollama':
            # Base timeout for local models
            base_timeout = LOCAL_MODEL_TIMEOUT
            
            # Increase timeout for larger models
            if model_config.estimated_ram_gb > 8:
                base_timeout = int(base_timeout * 1.5)  # 270 seconds for large models
            elif model_config.estimated_ram_gb > 12:
                base_timeout = int(base_timeout * 2)    # 360 seconds for very large models
                
            return base_timeout
        else:
            return 60  # Default for cloud models
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_content(self, prompt: str, model_config: Optional[ModelConfig] = None) -> Optional[str]:
        """Generate content using the configured model with retry logic"""
        if model_config is None:
            model_config = self.model_manager.current_model
            
        if not model_config:
            raise ValueError("No model configuration available")
        
        logger.debug(f"Generating content with {model_config.provider}:{model_config.model_id}")
        
        if model_config.provider == 'ollama':
            return self._generate_ollama(prompt, model_config)
        elif model_config.provider == 'gemini':
            return self._generate_gemini(prompt, model_config)
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_ollama(self, prompt: str, model_config: ModelConfig) -> Optional[str]:
        """Generate content using Ollama with retry logic and connection pooling"""
        start_time = time.time()
        
        try:
            # Use adaptive timeout based on model size
            timeout = self._get_adaptive_timeout(model_config)
            
            # Track request
            self.local_model_stats['total_requests'] += 1
            
            # Use session for connection pooling
            response = self.session.post(
                f"{self.model_manager.ollama_base_url}/api/generate",
                json={
                    "model": model_config.model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": model_config.temperature,
                        "num_predict": min(model_config.max_tokens, MAX_GEN_OUTPUT_TOKENS),
                        # Optimize for local inference
                        "num_ctx": 4096,  # Context window
                        "top_k": 40,      # Top-k sampling
                        "top_p": 0.9,     # Top-p sampling
                        "repeat_penalty": 1.1  # Avoid repetition
                    }
                },
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            # Track performance
            generation_time = time.time() - start_time
            self.local_model_stats['total_time_seconds'] += generation_time
            self.local_model_stats['total_tokens_generated'] += len(generated_text.split())
            
            logger.debug(f"🚀 Generated {len(generated_text)} chars in {generation_time:.2f}s")
            return generated_text
            
        except requests.exceptions.Timeout:
            self.local_model_stats['timeouts'] += 1
            logger.warning(f"Ollama request timed out after {timeout}s for model {model_config.model_id}")
            raise
        except requests.exceptions.ConnectionError:
            self.local_model_stats['errors'] += 1
            logger.error(f"Connection error to Ollama at {self.model_manager.ollama_base_url}")
            raise
        except requests.exceptions.RequestException as e:
            self.local_model_stats['errors'] += 1
            logger.error(f"Ollama request error: {e}")
            raise
        except Exception as e:
            self.local_model_stats['errors'] += 1
            logger.error(f"Ollama generation error: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_gemini(self, prompt: str, model_config: ModelConfig) -> Optional[str]:
        """Generate content using Gemini with retry logic"""
        try:
            genai = self._get_gemini_client()
            
            generation_config = {
                "temperature": model_config.temperature,
                "max_output_tokens": min(model_config.max_tokens, MAX_GEN_OUTPUT_TOKENS),
                "top_p": 0.95,
            }
            
            model = genai.GenerativeModel(
                model_config.model_id,
                generation_config=generation_config
            )
            
            response = model.generate_content(prompt)
            
            if response.candidates and len(response.candidates) > 0:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                logger.warning("No content generated by Gemini")
                return None
                
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def print_local_model_performance(self):
        """Print performance statistics for local model usage"""
        stats = self.local_model_stats
        if stats['total_requests'] == 0:
            return
            
        avg_time = stats['total_time_seconds'] / stats['total_requests']
        avg_tokens = stats['total_tokens_generated'] / stats['total_requests']
        tokens_per_second = stats['total_tokens_generated'] / stats['total_time_seconds'] if stats['total_time_seconds'] > 0 else 0
        success_rate = ((stats['total_requests'] - stats['errors']) / stats['total_requests']) * 100
        
        print("\n" + "="*60)
        print("🏁 LOCAL MODEL PERFORMANCE REPORT")
        print("="*60)
        print(f"📊 Request Statistics:")
        print(f"   • Total requests: {stats['total_requests']}")
        print(f"   • Successful requests: {stats['total_requests'] - stats['errors']}")
        print(f"   • Success rate: {success_rate:.1f}%")
        print(f"   • Timeouts: {stats['timeouts']}")
        print(f"   • Errors: {stats['errors']}")
        
        print(f"\n⚡ Performance Metrics:")
        print(f"   • Average response time: {avg_time:.2f}s")
        print(f"   • Average tokens per response: {avg_tokens:.0f}")
        print(f"   • Inference speed: {tokens_per_second:.1f} tokens/second")
        print(f"   • Total processing time: {stats['total_time_seconds']:.1f}s")
        print(f"   • Total tokens generated: {stats['total_tokens_generated']:,}")
        
        # Performance rating
        if tokens_per_second > 50:
            rating = "🚀 Excellent"
        elif tokens_per_second > 20:
            rating = "✅ Good"
        elif tokens_per_second > 10:
            rating = "⚠️  Fair"
        else:
            rating = "🐌 Slow"
            
        print(f"\n🎯 Performance Rating: {rating}")
        print("="*60)

class WebCrawler:
    """Enhanced web crawler using Crawl4AI v0.6.0+ features"""
    
    def __init__(self):
        self.session_data = {}
        self.session_id = f"llmstxt_session_{int(time.time())}"
        
    async def discover_all_links_first(self, base_url: str, max_pages: int = 50, safety_limit: int = None, comprehensive: bool = False) -> List[Dict[str, Any]]:
        """
        Enhanced deep crawling strategy: Multi-level recursive link discovery and prioritization.
        
        Strategy:
        1. Crawl main page and discover all links
        2. Score and prioritize discovered links
        3. In comprehensive mode: Crawl ALL discovered links recursively (for full text mode)
        4. In regular mode: Crawl only high-priority pages up to max_pages limit
        5. Continue until target reached or no new quality links found
        
        This ensures appropriate coverage based on the crawling mode.
        """
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
            from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
            import urllib.parse
            
            base_domain = urllib.parse.urlparse(base_url).netloc
            all_discovered_links = set()
            crawled_pages = []
            crawled_urls = set()  # Track already crawled URLs to avoid duplicates
            pages_to_explore = []  # Queue of pages to explore for more links
            
            logger.info(f"🚀 Enhanced Deep Crawling Strategy for: {base_url}")
            logger.info(f"🎯 Target: {max_pages} pages with multi-level link discovery")
            logger.info(f"🔍 Phase 1: Discovering all links on main page: {base_url}")
            
            # Phase 1: Crawl main page and discover all internal links
            main_page_config = CrawlerRunConfig(
                scraping_strategy=LXMLWebScrapingStrategy(),
                word_count_threshold=10,  # Lower threshold for link discovery
                page_timeout=30000,
                cache_mode=CacheMode.BYPASS,
                verbose=True
            )
            
            async with AsyncWebCrawler(verbose=True) as crawler:
                # Get the main page first
                main_result = await crawler.arun(base_url, config=main_page_config)
                
                if main_result and main_result.success:
                    # Extract main page content
                    main_content = ""
                    if hasattr(main_result, 'markdown') and main_result.markdown:
                        main_content = main_result.markdown.fit_markdown or main_result.markdown.raw_markdown
                    elif hasattr(main_result, 'cleaned_html') and main_result.cleaned_html:
                        main_content = main_result.cleaned_html
                    
                    main_word_count = len(main_content.split()) if main_content else 0
                    
                    # Add main page to results
                    crawled_pages.append({
                        'url': main_result.url,
                        'title': self._extract_title_v6(main_result),
                        'content': main_content,
                        'word_count': main_word_count,
                        'score': 10.0,  # Give main page highest score
                        'depth': 0,
                        'session_id': self.session_id,
                        'metadata': main_result.metadata or {},
                        'crawl_timestamp': datetime.now().isoformat(),
                        'discovery_phase': 'main_page'
                    })
                    
                    logger.info(f"✅ Main page crawled: {main_word_count} words")
                    crawled_urls.add(main_result.url)
                    
                    # Add main page to exploration queue for recursive link discovery
                    pages_to_explore.append(main_result)
                    
                    # Extract all internal links from main page using enhanced discovery
                    discovered_main_links = self._extract_all_links(main_result, base_url, base_domain)
                    all_discovered_links.update(discovered_main_links)
                    
                    logger.info(f"🔗 Discovered {len(all_discovered_links)} unique internal links from main page")
                    
                    # Phase 2: Multi-level crawling (mode-dependent)
                    if comprehensive:
                        logger.info(f"🚀 Phase 2: COMPREHENSIVE multi-level crawling - targeting ALL links")
                        max_depth = 5  # Increased depth for comprehensive coverage
                        actual_safety_limit = safety_limit or (max_pages * 10)  # Use provided or calculated safety limit
                    else:
                        logger.info(f"🚀 Phase 2: Selective multi-level crawling - targeting best links")
                        max_depth = 3  # Regular depth for selective crawling
                        actual_safety_limit = max_pages  # Limit to requested page count
                    
                    current_depth = 1
                    
                    while current_depth <= max_depth and len(crawled_pages) < actual_safety_limit:
                        logger.info(f"🌊 Crawling depth {current_depth} - Current pages: {len(crawled_pages)} | Safety limit: {actual_safety_limit}")
                        
                        # Get ALL uncrawled links for this depth level
                        uncrawled_links = [url for url in all_discovered_links if url not in crawled_urls]
                        
                        if not uncrawled_links:
                            logger.info(f"🔚 No more uncrawled links found at depth {current_depth}")
                            break
                        
                        # Score and filter links based on crawling mode
                        scored_links = []
                        for link in uncrawled_links:
                            score = self._score_url_importance(link, base_url)
                            if comprehensive:
                                # Comprehensive mode: Include ALL links (even low-scored ones)
                                if score > 0.0:
                                    scored_links.append((link, score))
                            else:
                                # Regular mode: Only include high-quality links
                                if score > 1.0:  # Higher threshold for regular mode
                                    scored_links.append((link, score))
                        
                        # Sort by score and determine how many to crawl
                        scored_links.sort(key=lambda x: x[1], reverse=True)
                        
                        if comprehensive:
                            links_to_crawl = scored_links  # Crawl ALL discovered links in comprehensive mode
                        else:
                            # Regular mode: limit to remaining page budget
                            remaining_budget = actual_safety_limit - len(crawled_pages)
                            links_to_crawl = scored_links[:remaining_budget]
                        
                        if not links_to_crawl:
                            logger.info(f"🔚 No more links to crawl at depth {current_depth}")
                            break
                        
                        if comprehensive:
                            logger.info(f"📊 Depth {current_depth}: Crawling ALL {len(links_to_crawl)} discovered links (comprehensive)")
                        else:
                            logger.info(f"📊 Depth {current_depth}: Crawling {len(links_to_crawl)} highest-priority links (selective)")
                        
                        # Crawl ALL links at this depth level
                        new_pages_found = 0
                        for link, score in links_to_crawl:
                            if len(crawled_pages) >= actual_safety_limit:
                                logger.warning(f"🛑 Reached safety limit of {actual_safety_limit} pages. Stopping crawl.")
                                break
                            
                            try:
                                logger.info(f"🔄 [{len(crawled_pages)+1}] Depth {current_depth}: {link} (Score: {score:.2f})")
                                
                                link_result = await crawler.arun(link, config=main_page_config)
                                crawled_urls.add(link)  # Mark as crawled regardless of success
                                
                                if link_result and link_result.success:
                                    link_content = ""
                                    if hasattr(link_result, 'markdown') and link_result.markdown:
                                        link_content = link_result.markdown.fit_markdown or link_result.markdown.raw_markdown
                                    elif hasattr(link_result, 'cleaned_html') and link_result.cleaned_html:
                                        link_content = link_result.cleaned_html
                                    
                                    link_word_count = len(link_content.split()) if link_content else 0
                                    
                                    if link_word_count >= 50:  # Quality threshold
                                        crawled_pages.append({
                                            'url': link_result.url,
                                            'title': self._extract_title_v6(link_result),
                                            'content': link_content,
                                            'word_count': link_word_count,
                                            'score': score,
                                            'depth': current_depth,
                                            'session_id': self.session_id,
                                            'metadata': link_result.metadata or {},
                                            'crawl_timestamp': datetime.now().isoformat(),
                                            'discovery_phase': f'depth_{current_depth}'
                                        })
                                        
                                        new_pages_found += 1
                                        logger.info(f"✅ [{len(crawled_pages)}] Added: {link_word_count} words | Score: {score:.2f} | Depth: {current_depth}")
                                        
                                        # Discover more links from this page for next depth level
                                        if current_depth < max_depth:
                                            new_links = self._extract_all_links(link_result, base_url, base_domain)
                                            before_count = len(all_discovered_links)
                                            all_discovered_links.update(new_links)
                                            after_count = len(all_discovered_links)
                                            
                                            if after_count > before_count:
                                                if comprehensive:
                                                    logger.debug(f"🔗 Found {after_count - before_count} new links from {link} (comprehensive mode)")
                                                else:
                                                    logger.debug(f"🔗 Found {after_count - before_count} new links from {link} (selective mode)")
                                    else:
                                        logger.debug(f"⏭️ Skipping low-content page ({link_word_count} words): {link}")
                                else:
                                    logger.warning(f"❌ Failed to crawl {link}: {link_result.error_message if link_result else 'No result'}")
                            
                            except Exception as e:
                                logger.warning(f"❌ Error crawling {link}: {e}")
                                crawled_urls.add(link)  # Mark as crawled to avoid retry
                                continue
                        
                        logger.info(f"📈 Depth {current_depth} completed: {new_pages_found} pages added, {len(all_discovered_links)} total links discovered")
                        current_depth += 1
                    
                    # Summary of crawling based on mode
                    total_links_discovered = len(all_discovered_links)
                    total_attempted = len(crawled_urls)
                    
                    if comprehensive:
                        logger.info(f"🎯 COMPREHENSIVE crawl completed:")
                        logger.info(f"   • Total pages successfully crawled: {len(crawled_pages)}")
                        logger.info(f"   • Total unique links discovered: {total_links_discovered}")
                        logger.info(f"   • Total crawl attempts: {total_attempted}")
                        logger.info(f"   • Maximum depth reached: {current_depth - 1}")
                        logger.info(f"   • Success rate: {len(crawled_pages)}/{total_attempted} = {100*len(crawled_pages)/max(total_attempted, 1):.1f}%")
                        logger.info(f"   • Coverage: {100*total_attempted/max(total_links_discovered, 1):.1f}% of discovered links attempted")
                    else:
                        logger.info(f"🎯 Selective crawl completed:")
                        logger.info(f"   • Total pages successfully crawled: {len(crawled_pages)}")
                        logger.info(f"   • Total unique links discovered: {total_links_discovered}")
                        logger.info(f"   • High-quality links crawled: {total_attempted}")
                        logger.info(f"   • Maximum depth reached: {current_depth - 1}")
                        logger.info(f"   • Success rate: {len(crawled_pages)}/{total_attempted} = {100*len(crawled_pages)/max(total_attempted, 1):.1f}%")
                    

                
                else:
                    logger.error(f"❌ Failed to crawl main page: {main_result.error_message if main_result else 'No result'}")
                    return []
            

            return crawled_pages
        
        except ImportError as e:
            logger.error(f"❌ Crawl4AI import error: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Systematic crawling failed: {e}", exc_info=True)
            return []

    def _score_url_importance(self, url: str, base_url: str) -> float:
        """Score URL importance based on structure and content indicators"""
        score = 1.0
        url_lower = url.lower()
        
        # Skip non-content files completely
        file_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.css', '.js', '.pdf', '.zip', '.xml', '.txt']
        if any(url_lower.endswith(ext) for ext in file_extensions):
            return 0.0  # Don't crawl asset files
        
        # High value pages for alternates.ai specifically
        high_value_keywords = [
            'agents', 'tools', 'ai', 'automation', 'solutions', 'features',
            'pricing', 'about', 'docs', 'documentation', 'guide', 'tutorial', 
            'api', 'reference', 'blog', 'article', 'news', 'support', 'help', 
            'faq', 'getting-started', 'overview', 'mcp', 'model', 'claude'
        ]
        
        for keyword in high_value_keywords:
            if keyword in url_lower:
                score += 3.0
                break
        
        # Medium value pages
        medium_value_keywords = [
            'product', 'service', 'resources', 'download', 'install', 
            'setup', 'config', 'examples', 'changelog', 'contact', 'team'
        ]
        
        for keyword in medium_value_keywords:
            if keyword in url_lower:
                score += 1.5
                break
        
        # Heavy penalty for utility/asset pages
        low_value_keywords = [
            'login', 'signup', 'register', 'cart', 'checkout', 'account',
            'profile', 'admin', 'search', 'tag', 'category', 'author',
            'icon', 'favicon', 'apple-touch', 'manifest', 'robots'
        ]
        
        for keyword in low_value_keywords:
            if keyword in url_lower:
                score -= 5.0  # Heavy penalty
                break
        
        # URL structure scoring
        path_segments = url.replace(base_url, '').strip('/').split('/')
        
        # Root level pages are important
        if len(path_segments) <= 1 or (len(path_segments) == 2 and path_segments[1] == ''):
            score += 2.0
        # Second level pages are also good
        elif len(path_segments) <= 2:
            score += 1.5
        # Deeper pages get less priority
        elif len(path_segments) > 4:
            score -= 1.0
        
        # Boost pages that look like actual content
        if any(segment for segment in path_segments if len(segment) > 3 and segment.isalpha()):
            score += 1.0
        
        # Heavy penalty for pagination and query parameters
        if '?page=' in url or '&page=' in url or '/page/' in url:
            score -= 3.0
        
        if '?' in url and len(url.split('?')[1]) > 10:  # Any query strings
            score -= 2.0
        
        return max(0.1, score)  # Minimum score of 0.1

    async def crawl_website(self, base_url: str, max_pages: int = 50) -> List[Dict[str, Any]]:
        """
        Crawl website using Crawl4AI's BestFirstCrawlingStrategy following official best practices.
        
        Implements recommendations from: https://docs.crawl4ai.com/core/deep-crawling/
        """
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
            from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
            from crawl4ai.extraction_strategy import LLMExtractionStrategy
            from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
            from crawl4ai.deep_crawling.filters import (
                FilterChain,
                DomainFilter,
                URLPatternFilter,
                ContentTypeFilter,
                SEOFilter
            )
            from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
            import urllib.parse

            base_domain = urllib.parse.urlparse(base_url).netloc
            pages = []

            # 1. Create an enhanced relevance scorer with domain-specific keywords
            # Extract base name from URL for more targeted scoring
            import urllib.parse
            parsed_url = urllib.parse.urlparse(base_url)
            domain_name = parsed_url.netloc.replace('www.', '').split('.')[0]
            
            # Enhanced keyword list based on common page types and domain
            enhanced_keywords = [
                "documentation", "guide", "tutorial", "api", "reference", 
                "blog", "article", "news", "feature", "product", "service",
                "about", "contact", "help", "support", "changelog", "release",
                "tools", "resources", "pricing", "plans", "download", "install",
                domain_name  # Add domain name as a relevant keyword
            ]
            
            keyword_scorer = KeywordRelevanceScorer(
                keywords=enhanced_keywords,
                weight=1.0  # Increase weight for better scoring
            )

                        # 2. Create sophisticated filter chain following best practices
            # Note: Using try-catch to handle API differences between Crawl4AI versions
            filter_chain = None
            try:
                # Try with the correct API parameters for URLPatternFilter
                filter_chain = FilterChain([
                    # Domain boundaries (essential for focused crawling)
                    DomainFilter(allowed_domains=[base_domain]),
                    
                    # Content type filtering - use allowed_types as recommended
                    ContentTypeFilter(
                        allowed_types=["text/html", "application/xhtml+xml"]
                    )
                ])
            except Exception as filter_error:
                logger.warning(f"⚠️ Advanced filtering not available: {filter_error}")
                logger.info("🔧 Using basic domain filtering only")
                try:
                    filter_chain = FilterChain([
                        DomainFilter(allowed_domains=[base_domain])
                    ])
                except:
                    logger.warning("⚠️ FilterChain not available, proceeding without filters")
                    filter_chain = None

            # 3. Configure BestFirstCrawlingStrategy with supported parameters only
            # Adjust depth based on max_pages for better coverage
            crawl_depth = 4 if max_pages > 100 else 3 if max_pages > 50 else 2
            
            deep_crawl_strategy = BestFirstCrawlingStrategy(
                max_depth=crawl_depth,  # Dynamic depth based on target page count
                max_pages=max_pages,
                include_external=False,  # Stay within domain
                url_scorer=keyword_scorer,
                filter_chain=filter_chain
                # Note: ignore_robots_txt and respect_nofollow are not supported parameters
            )

            # 4. Set up crawler configuration with only supported parameters
            crawler_config = CrawlerRunConfig(
                deep_crawl_strategy=deep_crawl_strategy,
                scraping_strategy=LXMLWebScrapingStrategy(),
                word_count_threshold=50,  # Higher threshold for quality content
                page_timeout=30000,  # 30 seconds timeout
                verbose=True,
                cache_mode=CacheMode.ENABLED  # Use cache for efficiency
                # Note: Removed unsupported parameters: js_timeout, wait_for_images, remove_overlay_elements, stream, extraction_strategy
            )

            # 5. Execute the crawl with proper analytics
            async with AsyncWebCrawler(verbose=True) as crawler:
                logger.info(f"🚀 Starting BestFirst deep crawl on {base_url}")
                logger.info(f"📊 Configuration: max_depth={crawl_depth}, max_pages={max_pages}, streaming=True")
                logger.info(f"🔧 Filters active: {'Yes' if filter_chain else 'Basic only'}")
                
                # Track analytics as recommended in docs
                depth_counts = {}
                total_score = 0
                processed_count = 0
                
                async for result in await crawler.arun(base_url, config=crawler_config):
                    if result.success:
                        # Extract content using best available source
                        content = ""
                        if hasattr(result, 'markdown') and result.markdown:
                            content = result.markdown.fit_markdown or result.markdown.raw_markdown
                        elif hasattr(result, 'cleaned_html') and result.cleaned_html:
                            content = result.cleaned_html
                        
                        word_count = len(content.split()) if content else 0
                        
                        if word_count > crawler_config.word_count_threshold:
                            score = result.metadata.get('score', 0)
                            depth = result.metadata.get('depth', 0)
                            
                            pages.append({
                                'url': result.url,
                                'title': self._extract_title_v6(result),
                                'content': content,
                                'word_count': word_count,
                                'score': score,
                                'depth': depth,
                                'session_id': self.session_id,
                                'metadata': result.metadata,
                                'crawl_timestamp': datetime.now().isoformat()
                            })
                            
                            # Analytics tracking (best practice from docs)
                            depth_counts[depth] = depth_counts.get(depth, 0) + 1
                            total_score += score
                            processed_count += 1
                            
                            logger.info(f"✅ [{len(pages)}/{max_pages}] Depth:{depth} | Score:{score:.2f} | {word_count} words | {result.url}")
                        else:
                            logger.debug(f"⏭️ Skipping low-content page ({word_count} words): {result.url}")
                    else:
                        logger.warning(f"❌ Failed to crawl {result.url}: {result.error_message}")
                
                # Provide analytics summary as recommended in best practices
                if processed_count > 0:
                    avg_score = total_score / processed_count
                    logger.info(f"📈 Crawl Analytics:")
                    logger.info(f"   Total pages: {len(pages)}")
                    logger.info(f"   Average score: {avg_score:.2f}")
                    logger.info(f"   Pages by depth: {dict(sorted(depth_counts.items()))}")
                
                logger.info(f"🎯 BestFirst deep crawl completed. Discovered {len(pages)} high-quality pages.")
                return pages

        except ImportError as e:
            logger.error(f"❌ Crawl4AI import error: {e}")
            logger.error("💡 Please install with: pip install 'crawl4ai[all]>=0.6.0'")
            return []
        except Exception as e:
            logger.error(f"❌ Deep crawling failed: {e}", exc_info=True)
            return []

    def _normalize_url(self, url: str) -> str:
        """Normalize URL to avoid duplicates by removing fragments and common tracking params."""
        import urllib.parse
        
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.rstrip('/') or '/'
        
        query_params = urllib.parse.parse_qs(parsed.query)
        allowed_params = {}
        tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'gclid', 'fbclid', 'gclsrc', '_ga', '_gl', 'mc_cid', 'mc_eid',
            'ref', 'referrer', 'source'
        }
        
        for key, value in query_params.items():
            if key.lower() not in tracking_params:
                allowed_params[key] = value
        
        new_query = urllib.parse.urlencode(allowed_params, doseq=True)
        
        return urllib.parse.urlunparse((
            parsed.scheme, parsed.netloc.lower(), path,
            parsed.params, new_query, ''
        ))
    
    def _extract_title_v6(self, result) -> str:
        """Enhanced title extraction with better fallbacks."""
        # Try metadata first
        if hasattr(result, 'metadata') and result.metadata and result.metadata.get('title'):
            title = result.metadata['title'].strip()
            if title and title.lower() not in ['untitled', '']:
                return self._clean_title_text(title)

        # Fallback to markdown H1
        if result.markdown and result.markdown.raw_markdown:
            match = re.search(r'^#\s+(.+)', result.markdown.raw_markdown.strip())
            if match:
                title = match.group(1).strip()
                if title:
                    return self._clean_title_text(title)
        
        # Fallback to URL
        if result.url:
            try:
                path_part = result.url.rstrip('/').split('/')[-1]
                if path_part:
                    title = path_part.replace('-', ' ').replace('_', ' ').title()
                    if len(title) > 3:
                        return self._clean_title_text(title)
            except:
                pass

        return "Untitled Page"

    def _clean_title_text(self, title: str) -> str:
        """Clean and normalize a page title."""
        if not title:
            return "Untitled Page"
        
        # Remove common site name suffixes/prefixes
        title = re.sub(r'\s*[-|–—]\s*[^-|–—]*$', '', title)
        title = re.sub(r'^\s*[^-|–—]*\s*[-|–—]\s*', '', title)
        
        # Remove HTML entities
        title = re.sub(r'&[a-zA-Z0-9#]+;', ' ', title)
        
        # Capitalize if all lower/upper
        if title.islower() or title.isupper():
            title = title.title()
            
        return ' '.join(title.split()).strip()

    def _extract_all_links(self, result, base_url: str, base_domain: str) -> set:
        """Enhanced link extraction from crawl result using multiple techniques"""
        discovered_links = set()
        
        try:
            # Method 1: Extract from Crawl4AI's links property
            if hasattr(result, 'links') and result.links:
                internal_links = result.links.get('internal', [])
                for link in internal_links:
                    if isinstance(link, dict):
                        link_url = link.get('href') or link.get('url')
                    else:
                        link_url = str(link)
                    
                    if link_url and base_domain in link_url:
                        normalized_url = self._normalize_url(link_url)
                        if normalized_url != base_url:
                            discovered_links.add(normalized_url)
            
            # Method 2: Enhanced HTML link extraction
            if hasattr(result, 'html') and result.html:
                import re
                
                # Multiple patterns to catch different link formats
                link_patterns = [
                    r'href=["\']([^"\']*)["\']',
                    r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>',
                    r'window\.location\s*=\s*["\']([^"\']+)["\']',
                    r'location\.href\s*=\s*["\']([^"\']+)["\']',
                    r'<link[^>]+href=["\']([^"\']+)["\'][^>]*>'
                ]
                
                found_links = set()
                for pattern in link_patterns:
                    matches = re.findall(pattern, result.html, re.IGNORECASE)
                    found_links.update(matches)
                
                # Look for data attributes and JavaScript navigation
                data_patterns = [
                    r'data-(?:href|link|url|navigate)=["\']([^"\']*)["\']',
                    r'data-page=["\']([^"\']*)["\']',
                    r'onclick=["\'][^"\']*location[^"\']*=[^"\']*["\']([^"\']+)["\']'
                ]
                
                for pattern in data_patterns:
                    matches = re.findall(pattern, result.html, re.IGNORECASE)
                    found_links.update(matches)
                
                # Process discovered links
                for link in found_links:
                    # Skip empty links, fragments, and external protocols
                    if not link or link.startswith('#') or link.startswith('mailto:') or link.startswith('tel:') or link.startswith('javascript:'):
                        continue
                    
                    # Convert relative links to absolute
                    if link.startswith('/'):
                        full_url = f"{base_url.rstrip('/')}{link}"
                    elif link.startswith('http') and base_domain in link:
                        full_url = link
                    elif not link.startswith('http'):  # Relative links
                        full_url = f"{base_url.rstrip('/')}/{link.lstrip('/')}"
                    else:
                        continue
                    
                    normalized_url = self._normalize_url(full_url)
                    if normalized_url != base_url:  # Don't add the same page
                        discovered_links.add(normalized_url)
            
            # Method 3: Extract from markdown content (for navigation menus)
            if hasattr(result, 'markdown') and result.markdown:
                markdown_content = result.markdown.raw_markdown or result.markdown.fit_markdown or ""
                if markdown_content:
                    import re
                    # Markdown link pattern: [text](url)
                    markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown_content)
                    for text, link in markdown_links:
                        if link and not link.startswith('#'):
                            if link.startswith('/'):
                                full_url = f"{base_url.rstrip('/')}{link}"
                            elif not link.startswith('http'):
                                full_url = f"{base_url.rstrip('/')}/{link.lstrip('/')}"
                            else:
                                full_url = link
                            
                            if base_domain in full_url:
                                normalized_url = self._normalize_url(full_url)
                                if normalized_url != base_url:
                                    discovered_links.add(normalized_url)
        
        except Exception as e:
            logger.debug(f"Error extracting links: {e}")
        
        return discovered_links

class LLMsTxtGenerator:
    """Main generator class with enhanced export and parallel processing capabilities"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.ai_client = AIClient(self.model_manager)
        self.crawler = WebCrawler()
        self.output_dir = os.getenv('OUTPUT_DIR', './output')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def interactive_model_selection(self):
        """Interactive model selection with user-friendly interface"""
        try:
            models = self.model_manager.list_models()
            if not models:
                print("❌ No AI models available. Please check your API keys or Ollama installation.")
                return False
            
            # Group models by provider
            gemini_models = []
            ollama_models = []
            
            for model_id, config in models.items():
                if config.provider == 'gemini':
                    gemini_models.append((model_id, config))
                elif config.provider == 'ollama':
                    ollama_models.append((model_id, config))
            
            # Display available models
            print("\n📋 Available AI Models:")
            print("=" * 50)
            
            model_options = []
            option_num = 1
            
            if gemini_models:
                print(f"\n🌟 Gemini Models (Cloud):")
                for model_id, config in gemini_models:
                    status_icon = self.model_manager._get_status_indicator(config.status)
                    print(f"  {option_num}. {status_icon} {config.display_name}")
                    print(f"     {config.description}")
                    model_options.append((model_id, config))
                    option_num += 1
            
            if ollama_models:
                print(f"\n🤖 Ollama Models (Local):")
                for model_id, config in ollama_models:
                    status_icon = self.model_manager._get_status_indicator(config.status)
                    ram_info = f" ({config.estimated_ram_gb:.1f}GB RAM)" if config.estimated_ram_gb > 0 else ""
                    print(f"  {option_num}. {status_icon} {config.display_name}{ram_info}")
                    print(f"     {config.description}")
                    model_options.append((model_id, config))
                    option_num += 1
            
            # Get user selection
            while True:
                try:
                    choice = input(f"\nSelect a model (1-{len(model_options)}): ").strip()
                    if not choice:
                        print("❌ Please enter a number.")
                        continue
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(model_options):
                        selected_model_id, selected_config = model_options[choice_num - 1]
                        
                        # Set the selected model
                        self.model_manager.current_model = selected_config
                        
                        print(f"\n✅ Selected: {selected_config.display_name}")
                        
                        # Show additional info for local models
                        if selected_config.provider == 'ollama':
                            ram_ok, ram_message = self.model_manager._check_ram_availability(selected_config.estimated_ram_gb)
                            print(f"🧠 {ram_message}")
                            if not ram_ok:
                                proceed = input("⚠️ Continue anyway? (y/N): ").strip().lower()
                                if proceed != 'y':
                                    print("Model selection cancelled.")
                                    return False
                        
                        return True
                    else:
                        print(f"❌ Please enter a number between 1 and {len(model_options)}.")
                        
                except ValueError:
                    print("❌ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n❌ Model selection cancelled.")
                    return False
                    
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            return False
    
    def remove_page_separators(self, text: str) -> str:
        """Remove page separators from text for clean output"""
        # Remove old style page separators
        text = re.sub(r'<\|crawl4ai-page-\d+-lllmstxt\|>\n', '', text)
        # Remove HTML comments
        text = re.sub(r'<!-- .* -->\n', '', text)
        # Remove horizontal rules between pages
        text = re.sub(r'\n---\n\n', '\n\n', text)
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    
    def limit_pages(self, full_text: str, max_pages: int) -> str:
        """Limit the number of pages in full text output"""
        pages = full_text.split('<|crawl4ai-page-')
        if len(pages) <= 1:
            return full_text
        
        # First element is the header
        result = pages[0]
        
        # Add up to max_pages
        for i in range(1, min(len(pages), max_pages + 1)):
            result += '<|crawl4ai-page-' + pages[i]
        
        return result
    
    def extract_domain_from_url(self, url: str) -> str:
        """Extract clean domain name from URL for filename generation"""
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace("www.", "")
        # Clean domain for safe filename use
        domain = re.sub(r'[^\w\-_.]', '_', domain)
        return domain
    
    def _detect_hallucination(self, description: str, title: str, content: str) -> bool:
        """Detect if the AI generated hallucinated content"""
        description_lower = description.lower()
        title_lower = title.lower()
        content_lower = content[:500].lower()  # Check first 500 chars
        
        # Red flags for hallucination
        hallucination_indicators = [
            'game character', 'tasks a-j', 'proof by contradiction', 'tree of thought',
            'let\'s say we have', 'consider each of these tasks', 'shortest path',
            'deploy tool a', 'tool b', 'tool c', 'character can only work',
            'proof by exhaustion', 'direct proof and inductive logic'
        ]
        
        # Check if description contains hallucination indicators
        for indicator in hallucination_indicators:
            if indicator in description_lower:
                logger.warning(f"🚨 Detected hallucination: '{indicator}' in description for {title}")
                return True
        
        # Check if description is completely unrelated to title/content
        title_words = set(title_lower.split())
        content_words = set(content_lower.split())
        description_words = set(description_lower.split())
        
        # Remove common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those'}
        title_words -= common_words
        content_words -= common_words
        description_words -= common_words
        
        # If description has no overlap with title or content, it's likely hallucinated
        if title_words and content_words:
            title_overlap = len(description_words & title_words) / len(title_words) if title_words else 0
            content_overlap = len(description_words & content_words) / len(content_words) if content_words else 0
            
            if title_overlap < 0.1 and content_overlap < 0.05:  # Very low overlap
                logger.warning(f"🚨 Detected unrelated content in description for {title}")
                return True
        
        return False

    def _extract_key_sentences(self, title: str, content: str) -> str:
        """Enhanced extraction of key sentences from actual page content"""
        try:
            # Clean and normalize content first
            content = self._clean_content_text(content)
            
            if not content or len(content.strip()) < 50:
                return self._clean_title(title) if title else "Website content"
            
            # Extract meaningful sentences
            sentences = []
            
            # Method 1: Extract from first paragraphs
            paragraphs = content.split('\n\n')
            good_paragraphs = []
            
            for para in paragraphs[:10]:  # Check first 10 paragraphs
                para = para.strip()
                if len(para) > 50 and len(para) < 500:  # Reasonable paragraph length
                    # Skip if it's mostly navigation/UI text
                    if not self._is_navigation_text(para):
                        good_paragraphs.append(para)
                        if len(good_paragraphs) >= 3:
                            break
            
            # Extract sentences from good paragraphs
            for para in good_paragraphs:
                para_sentences = re.split(r'[.!?]+', para)
                for sentence in para_sentences[:3]:  # Max 3 sentences per paragraph
                    sentence = sentence.strip()
                    if self._is_good_sentence(sentence):
                        sentences.append(sentence)
                        if len(sentences) >= 5:  # Max 5 sentences total
                            break
                if len(sentences) >= 5:
                    break
            
            # Method 2: If no good sentences, try bullet points or lists
            if not sentences:
                list_items = re.findall(r'(?:^|\n)[-*•]\s*([^\n]{20,150})', content, re.MULTILINE)
                for item in list_items[:3]:
                    item = item.strip()
                    if self._is_good_sentence(item):
                        sentences.append(item)
            
            # Method 3: If still no sentences, extract key phrases
            if not sentences:
                # Look for lines that might be descriptions
                lines = content.split('\n')
                for line in lines[:20]:
                    line = line.strip()
                    if 20 <= len(line) <= 150 and not self._is_navigation_text(line):
                        if ':' not in line or line.count(':') == 1:  # Avoid lists
                            sentences.append(line)
                            if len(sentences) >= 3:
                                break
            
            # Format result
            if sentences:
                # Clean up sentences
                cleaned_sentences = []
                for sentence in sentences[:3]:  # Use max 3 sentences
                    cleaned = self._clean_sentence(sentence)
                    if cleaned and len(cleaned) > 10:
                        cleaned_sentences.append(cleaned)
                
                if cleaned_sentences:
                    result = '. '.join(cleaned_sentences)
                    if not result.endswith('.'):
                        result += '.'
                    return result
            
            # Final fallback: use title with content type detection
            return self._create_content_description(title, content)
                
        except Exception as e:
            logger.debug(f"Error extracting sentences: {e}")
            return self._clean_title(title) if title else "Website content"
    
    def _clean_content_text(self, content: str) -> str:
        """Clean raw content text for better processing"""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common UI/navigation elements
        ui_patterns = [
            r'\b(?:click here|read more|learn more|sign up|log in|subscribe)\b',
            r'\b(?:menu|navigation|nav|header|footer|sidebar)\b',
            r'\b(?:skip to|go to|back to|return to)\b',
            r'\b(?:search|filter|sort by|view all)\b',
            r'\bcookie(?:s)?\s+(?:policy|notice|consent)\b',
            r'\bprivacy\s+policy\b',
            r'\bterms\s+(?:of\s+)?(?:service|use)\b',
            r'\bcopyright\s+\d{4}\b',
            r'\bfollow\s+us\s+on\b',
        ]
        
        for pattern in ui_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _is_navigation_text(self, text: str) -> bool:
        """Check if text is likely navigation/UI content"""
        text_lower = text.lower()
        
        nav_indicators = [
            'click here', 'read more', 'learn more', 'sign up', 'log in',
            'subscribe', 'newsletter', 'follow us', 'social media',
            'cookie', 'privacy policy', 'terms of service', 'copyright',
            'all rights reserved', 'menu', 'navigation', 'back to top',
            'skip to content', 'search', 'filter', 'sort by'
        ]
        
        # Check if text is mostly navigation
        nav_count = sum(1 for indicator in nav_indicators if indicator in text_lower)
        word_count = len(text.split())
        
        return nav_count > 0 and (nav_count / max(word_count, 1)) > 0.3
    
    def _is_good_sentence(self, sentence: str) -> bool:
        """Check if sentence is good for description"""
        if not sentence or len(sentence.strip()) < 15:
            return False
        
        sentence = sentence.strip()
        
        # Skip if too short or too long
        if len(sentence) < 15 or len(sentence) > 200:
            return False
        
        # Skip if it's likely navigation/UI text
        if self._is_navigation_text(sentence):
            return False
        
        # Skip if it's mostly punctuation or numbers
        word_chars = sum(1 for c in sentence if c.isalnum() or c.isspace())
        if word_chars / len(sentence) < 0.7:
            return False
        
        # Skip if it doesn't contain meaningful words
        meaningful_words = ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will', 'would', 'could', 'should']
        if not any(word in sentence.lower() for word in meaningful_words) and len(sentence.split()) < 5:
            return False
        
        return True
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean individual sentence"""
        if not sentence:
            return ""
        
        sentence = sentence.strip()
        
        # Remove extra whitespace
        sentence = ' '.join(sentence.split())
        
        # Remove trailing punctuation repetition
        sentence = re.sub(r'[.!?]{2,}$', '.', sentence)
        
        # Ensure proper capitalization
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize page titles"""
        if not title:
            return "Page Content"
        
        # Remove HTML entities and tags
        title = re.sub(r'&[a-zA-Z0-9#]+;', ' ', title)  # HTML entities
        title = re.sub(r'<[^>]+>', '', title)  # HTML tags
        
        # Clean up common title patterns
        title = re.sub(r'\s*[-|–—]\s*[^-|–—]*$', '', title)  # Remove site suffix
        title = re.sub(r'^\s*[^-|–—]*\s*[-|–—]\s*', '', title)  # Remove site prefix
        
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        # Capitalize properly
        if title.islower() or title.isupper():
            title = title.title()
        
        return title.strip() if title.strip() else "Page Content"

    def _create_content_description(self, title: str, content: str) -> str:
        """Create description based on content type analysis"""
        content_lower = content.lower()
        clean_title = self._clean_title(title)
        
        # Analyze content type
        if 'api' in content_lower and ('documentation' in content_lower or 'docs' in content_lower):
            return f"API documentation for {clean_title}"
        elif 'tutorial' in content_lower or 'guide' in content_lower:
            return f"Tutorial and guide for {clean_title}"
        elif 'pricing' in content_lower or 'plans' in content_lower:
            return f"Pricing information for {clean_title}"
        elif 'features' in content_lower or 'capabilities' in content_lower:
            return f"Features and capabilities of {clean_title}"
        elif 'download' in content_lower or 'install' in content_lower:
            return f"Download and installation for {clean_title}"
        elif 'support' in content_lower or 'help' in content_lower:
            return f"Support and help for {clean_title}"
        elif 'about' in content_lower or 'company' in content_lower:
            return f"About {clean_title}"
        elif 'blog' in content_lower or 'news' in content_lower:
            return f"Blog and news from {clean_title}"
        elif 'contact' in content_lower:
            return f"Contact information for {clean_title}"
        else:
            return f"Information about {clean_title}"



    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def generate_description_with_fallbacks(self, title: str, content: str, url: str = "") -> str:
        """Use LLM only to clean extracted text, not to generate descriptions"""
        # Check cache first
        cached_desc = self.model_manager.check_cached_description(url, title, content)
        if cached_desc:
            logger.debug(f"Using cached description for {url}")
            return cached_desc
        
        # Extract actual text from the page
        extracted_text = self._extract_key_sentences(title, content)
        
        # If extracted text is clean enough, use it directly
        if len(extracted_text.split()) <= 25 and not self._detect_hallucination(extracted_text, title, content):
            logger.info(f"✅ Using direct extraction for {title[:50]}...")
            self.model_manager.cache_description(url, title, content, extracted_text)
            return extracted_text
        
        # Use LLM only to clean the extracted text
        clean_prompt = f"""Clean this text. Remove extra words, fix grammar, make it 15-20 words. Use only the information given and make it descriptive 

{extracted_text}

Cleaned text:"""
        
        max_attempts = 2  # Fewer attempts since we're just cleaning
        for attempt in range(max_attempts):
            try:
                logger.debug(f"Cleaning attempt {attempt + 1}/{max_attempts} for {title[:50]}...")
                
                # Get current model and set shorter timeout for cleaning
                current_model = self.model_manager.current_model
                timeout_seconds = 30 if current_model and current_model.provider == 'ollama' else 20
                
                # Generate with timeout
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                
                def clean_with_timeout():
                    return self.ai_client.generate_content(clean_prompt)
                
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(clean_with_timeout)
                        result = future.result(timeout=timeout_seconds)
                    
                    if result and result.strip():
                        result = result.strip()
                        
                        # Clean up response
                        prefixes_to_remove = ['cleaned text:', 'clean text:', 'result:']
                        for prefix in prefixes_to_remove:
                            if result.lower().startswith(prefix):
                                result = result[len(prefix):].strip()
                        
                        # Remove unwanted formatting
                        result = re.sub(r'[*"\'`]', '', result)  # Remove quotes and asterisks
                        result = result.strip()
                        
                        # Validate result
                        word_count = len(result.split())
                        if 8 <= word_count <= 30:  # Reasonable length
                            # Check it's still related to original content
                            if not self._detect_hallucination(result, title, extracted_text):
                                logger.info(f"✅ Cleaned text for {title[:50]}... ({word_count} words)")
                                self.model_manager.cache_description(url, title, content, result)
                                return result
                            else:
                                logger.warning(f"🚨 Cleaned text unrelated in attempt {attempt + 1}")
                                continue
                        else:
                            logger.warning(f"⚠️  Cleaned text wrong length ({word_count} words), retrying...")
                            continue
                    else:
                        logger.warning(f"⚠️  Empty response in cleaning attempt {attempt + 1}")
                        continue
                        
                except FutureTimeoutError:
                    logger.warning(f"⏰ Cleaning timeout after {timeout_seconds}s in attempt {attempt + 1}")
                    continue
                except Exception as e:
                    logger.warning(f"❌ Cleaning error in attempt {attempt + 1}: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"❌ Failed cleaning attempt {attempt + 1} for {title}: {e}")
                continue
        
        # All cleaning attempts failed, use extracted text directly
        logger.info(f"🔄 Cleaning failed, using extracted text for {title[:50]}...")
        return self._create_smart_fallback(title, content, url)

    def _create_smart_fallback(self, title: str, content: str, url: str) -> str:
        """Create intelligent fallback descriptions based on actual content analysis"""
        try:
            content_lower = content.lower()
            title_lower = title.lower()
            clean_title = title.split('|')[0].split('-')[0].strip()
            
            # Extract key phrases from content
            key_phrases = []
            
            # Look for common website patterns
            if 'api' in content_lower and ('documentation' in content_lower or 'docs' in content_lower):
                return f"API documentation and developer resources for {clean_title}"
            elif 'pricing' in content_lower or 'plans' in content_lower or 'subscription' in content_lower:
                return f"Pricing plans and subscription options for {clean_title}"
            elif 'download' in content_lower or 'install' in content_lower:
                return f"Download and installation information for {clean_title}"
            elif 'tutorial' in content_lower or 'guide' in content_lower or 'how to' in content_lower:
                return f"Tutorial and guide for using {clean_title}"
            elif 'features' in content_lower or 'capabilities' in content_lower:
                return f"Features and capabilities overview of {clean_title}"
            elif 'support' in content_lower or 'help' in content_lower or 'faq' in content_lower:
                return f"Support and help documentation for {clean_title}"
            elif 'blog' in content_lower or 'news' in content_lower or 'article' in content_lower:
                return f"Blog posts and articles about {clean_title}"
            elif 'about' in content_lower or 'company' in content_lower:
                return f"About page and company information for {clean_title}"
            elif 'contact' in content_lower or 'email' in content_lower:
                return f"Contact information and details for {clean_title}"
            elif 'terms' in content_lower or 'privacy' in content_lower or 'policy' in content_lower:
                return f"Terms of service and privacy policy for {clean_title}"
            else:
                # Generic but safe fallback
                domain = url.split('/')[2] if '/' in url else clean_title
                return f"Information and content from {domain}"
                
        except Exception as e:
            logger.error(f"Error creating smart fallback: {e}")
            return f"Content from {url}" if url else "Website content"

    def process_urls_in_batches(self, pages: List[Dict], batch_size: int = 10, 
                               parallel_workers: int = 5) -> List[Dict]:
        """Process URLs in batches with production-level error handling"""
        # Adapt batch processing for local models
        current_model = self.model_manager.current_model
        if current_model and current_model.provider == 'ollama':
            # Use smaller batches and fewer workers for local models
            batch_size = min(batch_size, LOCAL_MODEL_BATCH_SIZE)
            parallel_workers = min(parallel_workers, 2)  # Conservative for local models
            logger.info(f"🔧 Optimizing for local model: batch_size={batch_size}, workers={parallel_workers}")
        
        logger.info(f"📊 Processing {len(pages)} pages in batches of {batch_size}")
        
        all_results = []
        failed_pages = []
        
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(pages) + batch_size - 1) // batch_size
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} pages)")
            
            batch_results = []
            batch_failures = []
            
            # Process batch with ThreadPoolExecutor and proper error handling
            with ThreadPoolExecutor(max_workers=min(parallel_workers, len(batch))) as executor:
                future_to_page = {
                    executor.submit(
                        self.generate_description_with_fallbacks, 
                        page['title'], 
                        page['content'], 
                        page.get('url', '')
                    ): page
                    for page in batch
                }
                
                for future in as_completed(future_to_page):
                    page = future_to_page[future]
                    try:
                        description = future.result(timeout=120)  # 2 minute max per page
                        batch_results.append({
                            'url': page['url'],
                            'title': page['title'],
                            'description': description,
                            'word_count': page.get('word_count', 0),
                            'session_id': page.get('session_id', ''),
                            'timestamp': datetime.now().isoformat(),
                            'index': len(all_results) + len(batch_results)
                        })
                        logger.debug(f"✅ Generated description for: {page['title'][:50]}...")
                    except Exception as e:
                        logger.error(f"❌ Failed to generate description for {page['title']}: {e}")
                        # Add entry with emergency fallback
                        emergency_fallback = self._create_smart_fallback(page['title'], page['content'], page.get('url', ''))
                        batch_failures.append({
                            'url': page['url'],
                            'title': page['title'],
                            'description': emergency_fallback,
                            'word_count': page.get('word_count', 0),
                            'session_id': page.get('session_id', ''),
                            'timestamp': datetime.now().isoformat(),
                            'index': len(all_results) + len(batch_results) + len(batch_failures),
                            'error': str(e)
                        })
            
            # Add batch results to total
            all_results.extend(batch_results)
            all_results.extend(batch_failures)
            failed_pages.extend(batch_failures)
            
            # Progress update
            success_rate = len(batch_results) / len(batch) * 100
            logger.info(f"📈 Batch {batch_num} completed: {len(batch_results)}/{len(batch)} successful ({success_rate:.1f}%)")
            
            # Add delay between batches for local models to avoid overloading
            if current_model and current_model.provider == 'ollama' and batch_num < total_batches:
                delay_seconds = LOCAL_MODEL_RETRY_DELAY
                logger.info(f"⏱️  Waiting {delay_seconds}s before next batch (local model optimization)")
                time.sleep(delay_seconds)
        
        # Summary
        total_success = len(all_results) - len(failed_pages)
        overall_success_rate = total_success / len(pages) * 100 if pages else 0
        
        if failed_pages:
            logger.warning(f"⚠️  {len(failed_pages)} pages used fallback descriptions")
        
        logger.info(f"🎯 Processing complete: {total_success}/{len(pages)} successful ({overall_success_rate:.1f}%)")
        
        return all_results
    
    async def generate_llmstxt(self, base_url: str, max_pages: int = 50, 
                             export_format: str = 'text', 
                             include_full_text: bool = True,
                             parallel_workers: int = None,
                             batch_size: int = 10,
                             max_full_text_pages: int = None,
                             full_text_only: bool = False,
                             crawl_strategy: str = 'systematic',
                             safety_limit: int = None):
        """Generate LLMs.txt from a website with enhanced parallel processing and export options"""
        start_time = time.time()
        
        # Ensure we have a model selected
        if not self.model_manager.current_model:
            logger.error("❌ No model selected. Use interactive mode or specify a model.")
            return
        
        current_model = self.model_manager.current_model
        logger.info(f"🤖 Using model: {current_model.display_name} ({current_model.provider})")
        
        # Warm up local models for better performance
        if current_model.provider == 'ollama':
            logger.info("🔥 Warming up local model for optimal performance...")
            if not self.model_manager.warm_up_model(current_model.model_id):
                logger.warning("⚠️  Model warm-up failed, but continuing anyway...")
            
            # Check RAM availability
            ram_ok, ram_message = self.model_manager._check_ram_availability(current_model.estimated_ram_gb)
            logger.info(f"🧠 {ram_message}")
            if not ram_ok:
                logger.warning("⚠️  Low RAM detected. Consider using a smaller model or increasing system RAM.")
        
        # Adaptive worker configuration based on model type
        if parallel_workers is None:
            if current_model.provider == 'ollama':
                parallel_workers = min(DEFAULT_PARALLEL_WORKERS, 2)  # Conservative for local models
                logger.info("🔧 Using conservative parallelism for local model")
            else:
                parallel_workers = DEFAULT_PARALLEL_WORKERS
        
        # Determine crawling strategy based on output type
        if full_text_only or include_full_text:
            # For full text generation, increase page limit to get comprehensive content
            actual_max_pages = max_pages * 3 if full_text_only else max_pages
            logger.info(f"🌐 Starting comprehensive website crawl for full content: {base_url}")
            logger.info(f"📊 Enhanced crawling - targeting up to {actual_max_pages} pages for complete coverage")
        else:
            actual_max_pages = max_pages
        logger.info(f"🌐 Starting website crawl: {base_url}")
        
        logger.info(f"📊 Configuration: max_pages={actual_max_pages}, format={export_format}, parallel_workers={parallel_workers}")
        
        # Choose crawling strategy based on parameter and mode
        if crawl_strategy == 'systematic':
            if full_text_only:
                logger.info("🎯 Using COMPREHENSIVE deep crawling strategy (Full Text Mode)")
                # Calculate safety limit for comprehensive crawling in full text mode
                calculated_safety_limit = safety_limit or (actual_max_pages * 10)
                logger.info(f"🛡️ Safety limit set to {calculated_safety_limit} pages for comprehensive crawling")
                pages = await self.crawler.discover_all_links_first(base_url, actual_max_pages, calculated_safety_limit, comprehensive=True)
            else:
                logger.info("🎯 Using systematic discovery-first crawling strategy")
                # Use comprehensive crawling for better content coverage in normal mode too
                logger.info(f"📊 Regular mode: Comprehensive crawling targeting {actual_max_pages} high-quality pages")
                calculated_safety_limit = safety_limit or (actual_max_pages * 5)  # Less aggressive than full-text mode
                logger.info(f"🛡️ Safety limit set to {calculated_safety_limit} pages for comprehensive crawling")
                pages = await self.crawler.discover_all_links_first(base_url, actual_max_pages, calculated_safety_limit, comprehensive=True)
        else:
            logger.info("🌊 Using BestFirst deep crawling strategy")
            pages = await self.crawler.crawl_website(base_url, actual_max_pages)
        
        if not pages:
            logger.error("❌ No pages could be crawled")
            return
        
        logger.info(f"✅ Successfully crawled {len(pages)} pages")
        
        # Always process pages through AI to generate descriptions for better content analysis
        logger.info("🧠 Generating descriptions using AI model...")
        llms_entries = self.process_urls_in_batches(
            pages, 
            batch_size=batch_size, 
            parallel_workers=parallel_workers
        )
        
        if not llms_entries:
            logger.error("❌ No descriptions could be generated")
            return
        
        # Create metadata
        domain = self.extract_domain_from_url(base_url)
        metadata = {
            'source_url': base_url,
            'domain': domain,
            'total_pages': len(pages),
            'generated_at': datetime.now().isoformat(),
            'model_used': f"{current_model.provider}:{current_model.model_id}",
            'model_display_name': current_model.display_name,
            'export_format': export_format,
            'processing_time_seconds': round(time.time() - start_time, 2),
            'estimated_model_ram_gb': current_model.estimated_ram_gb,
            'batch_size': batch_size,
            'parallel_workers': parallel_workers or 1,
            'max_full_text_pages': max_full_text_pages,
            'full_text_only': full_text_only,
            'comprehensive_crawl': full_text_only or include_full_text,
            'actual_max_pages': actual_max_pages
        }
        
        # Generate output files
        await self._write_output_files(base_url, llms_entries, pages, metadata, export_format, include_full_text, full_text_only)
        
        # Print summary
        self._print_generation_summary(metadata, llms_entries)
        
        logger.info(f"🎉 Generation completed in {metadata['processing_time_seconds']}s")
    
    def _print_generation_summary(self, metadata: Dict, entries: List[Dict]):
        """Print a comprehensive summary of the generation process"""
        print("\n" + "="*60)
        print("🎉 GENERATION SUMMARY")
        print("="*60)
        print(f"📊 Statistics:")
        print(f"   • Total pages found: {metadata['total_pages']}")
        print(f"   • Pages successfully processed: {len(entries)}")
        print(f"   • Success rate: {100 * (len(entries) / metadata['total_pages']):.2f}%")
        print(f"   • Processing time: {metadata['processing_time_seconds']}s")
        print(f"   • Average time per page: {metadata['processing_time_seconds'] / max(len(entries), 1):.2f}s")
        
        # Show comprehensive crawling info
        if metadata.get('comprehensive_crawl'):
            print(f"   • Comprehensive crawl: ✅ Enhanced page discovery")
            print(f"   • Target pages: {metadata.get('actual_max_pages', 'Unknown')}")
            if metadata.get('full_text_only'):
                print(f"   • Mode: Full content only (no AI descriptions)")
            else:
                print(f"   • Mode: Full content + AI descriptions")
        
        print(f"\n🤖 Model & Configuration:")
        print(f"   • Model used: {metadata['model_used']}")
        print(f"   • Batch size: {metadata.get('batch_size', 'Unknown')}")
        print(f"   • Parallel workers: {metadata.get('parallel_workers', 'Unknown')}")
        print(f"   • Export format: {metadata['export_format']}")
        
        # Calculate word count statistics
        word_counts = [entry.get('word_count', 0) for entry in entries if 'word_count' in entry]
        if word_counts:
            print(f"\n📝 Content Statistics:")
            print(f"   • Total words processed: {sum(word_counts):,}")
            print(f"   • Average words per page: {sum(word_counts) // len(word_counts):,}")
            print(f"   • Longest page: {max(word_counts):,} words")
            print(f"   • Shortest page: {min(word_counts):,} words")
        
        # Show output files generated
        print(f"\n📄 Generated Files:")
        domain = metadata.get('domain', 'unknown')
        if not metadata.get('full_text_only'):
            print(f"   • {domain}-llms.txt (structured index with AI descriptions)")
        if metadata.get('comprehensive_crawl') or metadata.get('full_text_only'):
            print(f"   • {domain}-llms-full.txt (complete content)")
        if metadata.get('export_format') == 'json':
            print(f"   • {domain}-llms.json (structured data)")
        elif metadata.get('export_format') == 'yaml':
            print(f"   • {domain}-llms.yaml (structured data)")
        
        print("="*60)
    
    async def _crawl_parallel(self, base_url: str, max_pages: int, workers: int) -> List[Dict[str, Any]]:
        """Parallel crawling implementation"""
        # For now, use the existing crawler but log that parallel is requested
        # Future implementation could spawn multiple crawler instances
        logger.info(f"Parallel crawling requested with {workers} workers (currently using sequential)")
        return await self.crawler.crawl_website(base_url, max_pages)
    
    async def _write_output_files(self, base_url: str, llms_entries: List[Dict], 
                                pages: List[Dict], metadata: Dict, 
                                export_format: str, include_full_text: bool, full_text_only: bool = False):
        """Write output files in various formats"""
        domain = self.extract_domain_from_url(base_url)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format.lower() == 'json':
            await self._write_json_output(domain, timestamp, llms_entries, pages, metadata, include_full_text, full_text_only)
        elif export_format.lower() == 'yaml':
            await self._write_yaml_output(domain, timestamp, llms_entries, pages, metadata, include_full_text, full_text_only)
        else:
            await self._write_text_output(base_url, timestamp, llms_entries, pages, metadata, include_full_text, full_text_only)
    
    async def _write_text_output(self, base_url: str, timestamp: str, 
                               llms_entries: List[Dict], pages: List[Dict], 
                               metadata: Dict, include_full_text: bool, full_text_only: bool = False):
        """Write traditional text format output with better filename handling"""
        domain = self.extract_domain_from_url(base_url)
        
        # Generate llms.txt (unless full_text_only is True)
        if not full_text_only:
            llms_filename = os.path.join(self.output_dir, f'{domain}-llms.txt')
            with open(llms_filename, 'w', encoding='utf-8') as f:
                # H1 header with site name (required by spec)
                site_name = self._extract_site_name(base_url, pages)
                f.write(f"# {site_name}\n\n")
                
                # Blockquote with site summary (recommended by spec)
                site_summary = self._generate_site_summary(pages)
                f.write(f"> {site_summary}\n\n")
                
                # Optional details section
                f.write(f"Generated from {metadata['total_pages']} pages on {metadata['generated_at'].split('T')[0]} using automated crawling.\n\n")
                
                # Categorize entries by content type
                categorized_entries = self._categorize_entries(llms_entries, pages)
                
                # Write H2 sections with categorized links
                for category, entries in categorized_entries.items():
                    if entries:  # Only write section if it has entries
                        f.write(f"## {category}\n\n")
                        
                        for entry in entries:
                            # Find the corresponding page content for this entry
                            page_content = ""
                            for page in pages:
                                if page.get('url') == entry['url']:
                                    # Get raw content and clean it up
                                    raw_content = page.get('content', '')
                                    # Clean HTML/markdown and extract meaningful text
                                    import re
                                    # Remove HTML tags, links, and markdown
                                    cleaned = re.sub(r'<[^>]+>', '', raw_content)  # Remove HTML
                                    cleaned = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', cleaned)  # Remove images
                                    cleaned = re.sub(r'\[[^\]]*\]\([^)]*\)', '', cleaned)  # Remove links
                                    cleaned = re.sub(r'[#*_`]', '', cleaned)  # Remove markdown
                                    # Take meaningful content, skip navigation/header stuff
                                    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
                                    meaningful_content = []
                                    for line in lines:
                                        if len(line) > 30 and not any(skip in line.lower() for skip in ['logo', 'navigation', 'menu', 'skip to']):
                                            meaningful_content.append(line)
                                        if len(' '.join(meaningful_content)) > 150:
                                            break
                                    page_content = ' '.join(meaningful_content)[:200] + ("..." if len(' '.join(meaningful_content)) > 200 else "")
                                    break
                            
                            # Use raw content instead of AI description
                            content_to_show = page_content if page_content else entry.get('description', 'No content available')
                            f.write(f"- [{entry['title']}]({entry['url']}): {content_to_show}\n")
                        
                        f.write("\n")  # Add spacing between sections
            
            logger.info(f"📄 Created llms.txt following official spec: {llms_filename}")
        
        # Generate llms-full.txt (if include_full_text is True OR full_text_only is True)
        if include_full_text or full_text_only:
            full_filename = os.path.join(self.output_dir, f'{domain}-llms-full.txt')
            with open(full_filename, 'w', encoding='utf-8') as f:
                f.write(f"# {base_url} llms-full.txt\n")
                f.write(f"# Generated on {metadata['generated_at']} using {metadata['model_used']}\n")
                f.write(f"# Total pages: {metadata['total_pages']}\n")
                f.write(f"# Processing time: {metadata['processing_time_seconds']}s\n")
                
                # Add limit info if applicable
                if metadata.get('max_full_text_pages') and len(pages) > metadata.get('max_full_text_pages'):
                    f.write(f"# Page limit applied: {metadata.get('max_full_text_pages')} pages\n")
                f.write("\n")
                
                for i, page in enumerate(pages, 1):
                    f.write(f"## {page['title']}\n")
                    f.write(f"**URL:** {page['url']}\n\n")
                    f.write(page['content'])
                    f.write('\n\n---\n\n')
            
            logger.info(f"📄 Created llms-full.txt: {full_filename}")
            
            # Optionally create a clean version without page separators (only if not full_text_only)
            if not full_text_only:
                clean_filename = os.path.join(self.output_dir, f'{domain}-llms-full-clean.txt')
                with open(clean_filename, 'w', encoding='utf-8') as f:
                    with open(full_filename, 'r', encoding='utf-8') as source:
                        clean_content = self.remove_page_separators(source.read())
                        f.write(clean_content)
                
                logger.info(f"🧹 Created clean version: {clean_filename}")
    
    async def _write_json_output(self, domain: str, timestamp: str, 
                               llms_entries: List[Dict], pages: List[Dict], 
                               metadata: Dict, include_full_text: bool, full_text_only: bool = False):
        """Write JSON format output with better filename handling"""
        domain = self.extract_domain_from_url(metadata['source_url'])
        
        output_data = {
            'metadata': metadata,
            'llms_entries': llms_entries
        }
        
        if include_full_text:
            output_data['full_content'] = pages
        
        json_filename = os.path.join(self.output_dir, f'{domain}-llms.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Created JSON output: {json_filename}")
    
    async def _write_yaml_output(self, domain: str, timestamp: str, 
                               llms_entries: List[Dict], pages: List[Dict], 
                               metadata: Dict, include_full_text: bool, full_text_only: bool = False):
        """Write YAML format output with better filename handling"""
        domain = self.extract_domain_from_url(metadata['source_url'])
        
        output_data = {
            'metadata': metadata,
            'llms_entries': llms_entries
        }
        
        if include_full_text:
            output_data['full_content'] = pages
        
        yaml_filename = os.path.join(self.output_dir, f'{domain}-llms.yaml')
        with open(yaml_filename, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info(f"📄 Created YAML output: {yaml_filename}")

    def _save_api_key_to_env(self, api_key: str):
        """Safely save API key to .env file with atomic write operation"""
        env_file = '.env'
        
        try:
            # Create a temporary file for atomic write
            with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                           dir=os.path.dirname(os.path.abspath(env_file)), 
                                           prefix='.env_tmp_') as tmp_file:
                tmp_path = tmp_file.name
                
                # Read existing .env content
                existing_lines = []
                if os.path.exists(env_file):
                    with open(env_file, 'r', encoding='utf-8') as f:
                        existing_lines = f.readlines()
                
                # Write content to temporary file
                gemini_key_found = False
                for line in existing_lines:
                    if line.strip().startswith('GEMINI_API_KEY='):
                        tmp_file.write(f'GEMINI_API_KEY={api_key}\n')
                        gemini_key_found = True
                    else:
                        tmp_file.write(line)
                
                # Add key if not found
                if not gemini_key_found:
                    tmp_file.write(f'GEMINI_API_KEY={api_key}\n')
            
            # Atomic move: replace original file with temporary file
            if os.name == 'nt':  # Windows
                # On Windows, we need to remove the target file first
                if os.path.exists(env_file):
                    os.remove(env_file)
            shutil.move(tmp_path, env_file)
            
            logger.info(f"✅ API key saved to {env_file}")
            
        except Exception as e:
            # Clean up temporary file on error
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            logger.error(f"Failed to save API key to .env: {e}")
            raise

    def generate_description(self, title: str, content: str, url: str = "") -> str:
        """Compatibility wrapper for the enhanced description generation"""
        return self.generate_description_with_fallbacks(title, content, url)

    def get_crawl_recommendations(self, max_pages: int) -> Dict[str, Any]:
        """
        Provide crawling configuration recommendations based on Crawl4AI best practices.
        
        Based on: https://docs.crawl4ai.com/core/deep-crawling/
        """
        recommendations = {
            "strategy": "BestFirstCrawlingStrategy",
            "reasoning": "Recommended for intelligent exploration prioritization",
            "max_depth": 3 if max_pages > 20 else 2,
            "depth_reasoning": "Best practice: ≤3 to avoid exponential growth",
            "streaming": True,
            "streaming_reasoning": "Process results immediately as they're discovered",
            "filters": [
                "DomainFilter (stay within domain)",
                "URLPatternFilter (target valuable content)",
                "ContentTypeFilter (HTML only)",
                "SEOFilter (quality assessment)"
            ],
            "scorer": "KeywordRelevanceScorer",
            "scorer_reasoning": "Prioritizes most relevant pages first",
            "extraction_strategy": "LLMExtractionStrategy (if API key available)",
            "word_count_threshold": 50,
            "threshold_reasoning": "Higher threshold for quality content",
            "tips": [
                "Set realistic limits - be cautious with max_depth > 3",
                "Use max_pages to set hard limits and control execution time",
                "Experiment with keyword weights for optimal prioritization",
                "Monitor analytics (depth distribution, average scores)",
                "Be a good web citizen - respect robots.txt"
            ]
        }
        
        # Adjust recommendations based on crawl size
        if max_pages <= 10:
            recommendations["max_depth"] = 2
            recommendations["focus"] = "Quality over quantity - focus on most relevant pages"
        elif max_pages <= 50:
            recommendations["max_depth"] = 3
            recommendations["focus"] = "Balanced crawl - good coverage with quality control"
        else:
            recommendations["max_depth"] = 3
            recommendations["focus"] = "Large scale crawl - monitor performance and set timeouts"
            recommendations["additional_tips"] = [
                "Consider using score_threshold for BFS/DFS strategies",
                "Monitor memory usage for large crawls",
                "Use progress tracking and periodic analytics"
            ]
        
        return recommendations

    def print_crawl_recommendations(self, max_pages: int):
        """Print formatted crawling recommendations"""
        recs = self.get_crawl_recommendations(max_pages)
        
        print(f"\n🎯 Crawl Configuration Recommendations (max_pages={max_pages})")
        print("=" * 60)
        print(f"📋 Strategy: {recs['strategy']}")
        print(f"   {recs['reasoning']}")
        print(f"📊 Max Depth: {recs['max_depth']}")
        print(f"   {recs['depth_reasoning']}")
        print(f"🔄 Streaming: {recs['streaming']}")
        print(f"   {recs['streaming_reasoning']}")
        print(f"🎯 Focus: {recs['focus']}")
        
        print(f"\n🛡️ Recommended Filters:")
        for filter_item in recs['filters']:
            print(f"   • {filter_item}")
        
        print(f"\n💡 Best Practice Tips:")
        for tip in recs['tips']:
            print(f"   • {tip}")
            
        if 'additional_tips' in recs:
            print(f"\n⚠️ Large Scale Crawl Tips:")
            for tip in recs['additional_tips']:
                print(f"   • {tip}")
        
        print(f"\n📚 Reference: https://docs.crawl4ai.com/core/deep-crawling/")
        print("=" * 60)

    def _extract_site_name(self, base_url: str, pages: List[Dict]) -> str:
        """Extract a clean site name following llms.txt specification"""
        # Try to get from the main page title first
        if pages:
            main_page = next((page for page in pages if page.get('url') == base_url or page.get('url') == base_url.rstrip('/')), None)
            if main_page and main_page.get('title'):
                title = main_page['title']
                # Clean common title patterns
                title = title.split('|')[0].split('-')[0].strip()
                if title and len(title) > 3:
                    return title
        
        # Fallback to domain name
        import urllib.parse
        parsed = urllib.parse.urlparse(base_url)
        domain = parsed.netloc.replace('www.', '')
        return domain.split('.')[0].title()
    
    def _generate_site_summary(self, pages: List[Dict]) -> str:
        """Generate a concise site summary for the blockquote section"""
        if not pages:
            return "A website with various content and resources."
        
        # Look for common patterns in page content to determine site type
        all_content = ' '.join([page.get('content', '')[:500].lower() for page in pages[:5]])  # First 5 pages
        
        # Detect site type based on content patterns
        if any(keyword in all_content for keyword in ['api', 'documentation', 'docs', 'developer', 'reference']):
            return "Documentation and developer resources for software tools and APIs."
        elif any(keyword in all_content for keyword in ['pricing', 'plans', 'subscription', 'buy', 'purchase']):
            return "Software and service marketplace with pricing information and tool comparisons."
        elif any(keyword in all_content for keyword in ['blog', 'article', 'news', 'post']):
            return "Blog and articles covering various topics and insights."
        elif any(keyword in all_content for keyword in ['tutorial', 'guide', 'how to', 'learn']):
            return "Educational content with tutorials and learning resources."
        elif any(keyword in all_content for keyword in ['product', 'service', 'solution', 'tool']):
            return "Product and service information with detailed descriptions and features."
        else:
            return "A comprehensive website with information, resources, and various content sections."
    
    def _categorize_entries(self, llms_entries: List[Dict], pages: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize entries into logical sections following llms.txt best practices"""
        categories = {
            "Documentation": [],
            "Products & Services": [],
            "Resources": [],
            "API & Technical": [],
            "Optional": []
        }
        
        for entry in llms_entries:
            url_lower = entry['url'].lower()
            title_lower = entry['title'].lower()
            
            # Find corresponding page content for better categorization
            page_content = ""
            for page in pages:
                if page.get('url') == entry['url']:
                    page_content = page.get('content', '')[:300].lower()
                    break
            
            # Categorize based on URL patterns and content
            if any(keyword in url_lower for keyword in ['/docs', '/documentation', '/api', '/reference']):
                categories["API & Technical"].append(entry)
            elif any(keyword in url_lower for keyword in ['/guide', '/tutorial', '/help', '/support']):
                categories["Documentation"].append(entry)
            elif any(keyword in page_content for keyword in ['api', 'endpoint', 'developer', 'code', 'technical']):
                categories["API & Technical"].append(entry)
            elif any(keyword in page_content for keyword in ['pricing', 'plans', 'product', 'service', 'tool']):
                categories["Products & Services"].append(entry)
            elif any(keyword in url_lower for keyword in ['/blog', '/news', '/article', '/resources']):
                categories["Resources"].append(entry)
            elif any(keyword in url_lower for keyword in ['?page=', '/page/', '/compare', '/vs']):
                categories["Optional"].append(entry)  # Secondary pages
            else:
                # Default to main sections based on content hints
                if 'pricing' in page_content or 'product' in page_content:
                    categories["Products & Services"].append(entry)
                elif 'documentation' in page_content or 'guide' in page_content:
                    categories["Documentation"].append(entry)
                else:
                    categories["Resources"].append(entry)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

def create_sample_env_file():
    """Create a sample .env file for users to reference"""
    sample_content = """# LLMs.txt Generator Configuration
# Copy values and update as needed

# =============================================================================
# AI MODEL CONFIGURATION
# =============================================================================

# Gemini API Configuration
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Ollama Configuration (for local models)
# Default: http://localhost:11434
OLLAMA_BASE_URL=http://localhost:11434

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Output directory for generated files
# Default: ./output
OUTPUT_DIR=./output

# Maximum tokens for AI generation
# Default: 1024
MAX_GEN_OUTPUT_TOKENS=1024

# Enable/disable description caching to avoid regenerating unchanged summaries
# Default: true
CACHE_DESCRIPTIONS=true

# Default number of parallel workers for processing
# Default: 3 (reduced automatically for local models)
DEFAULT_PARALLEL_WORKERS=3

# =============================================================================
# SECURITY NOTES
# =============================================================================

# - Never commit your .env file with real API keys to version control
# - Add .env to your .gitignore file
# - Keep your API keys secure and rotate them regularly
"""

    try:
        with open('env.example.llmstxt', 'w', encoding='utf-8') as f:
            f.write(sample_content)
        return True
    except Exception as e:
        logger.warning(f"Could not create sample .env file: {e}")
        return False

def main():
    """Enhanced main function with comprehensive argument parsing"""
    
    # Create sample .env file if it doesn't exist
    if not os.path.exists('env.example.llmstxt'):
        create_sample_env_file()
    
    parser = argparse.ArgumentParser(
        description='Generate llms.txt and llms-full.txt files using Crawl4AI and AI models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (will prompt for website and options)
  python generate-llmstxt-crawl4ai.py

  # Direct usage
  python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com

  # Export as JSON with parallel crawling
  python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com --format json --parallel-crawl 5

  # YAML export without full text
  python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com --format yaml --no-full-text

  # Custom batch processing
  python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com --batch-size 15 --max-full-pages 30

  # List available models
  python generate-llmstxt-crawl4ai.py --list-models --llm-provider ollama
  
  # Use Gemini 2.0 Flash model
  python generate-llmstxt-crawl4ai.py https://docs.crawl4ai.com --llm-provider gemini --gemini-model gemini-2.0-flash-exp
        """
    )
    
    # Website URL (positional argument)
    parser.add_argument('url', nargs='?', help='Website URL to crawl')
    
    # Crawling options
    parser.add_argument('--max-pages', type=int, default=50, 
                        help='Maximum pages to crawl (default: 50)')
    parser.add_argument('--max-full-pages', type=int, default=None,
                        help='Maximum pages to include in full text output')
    parser.add_argument('--parallel-crawl', type=int, default=None,
                        help='Number of parallel crawlers to use')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for AI processing (default: 10)')
    parser.add_argument('--crawl-strategy', choices=['systematic', 'bestfirst'], default='systematic',
                        help='Crawling strategy: systematic (discover all links first) or bestfirst (deep crawling)')
    parser.add_argument('--safety-limit', type=int, default=None,
                        help='Maximum total pages to crawl as safety limit (default: 3x max-pages)')
    parser.add_argument('--force-scoring', action='store_true',
                        help='Force enable scoring for better prioritization')
    
    # Output options
    parser.add_argument('--format', choices=['text', 'json', 'yaml'], default='text',
                        help='Output format (default: text)')
    parser.add_argument('--no-full-text', action='store_true',
                        help='Skip generating full text content')
    parser.add_argument('--full-text-only', action='store_true',
                        help='Generate only full text without descriptions')
    parser.add_argument('--output-dir', default='./output',
                        help='Output directory (default: ./output)')
    
    # Model options
    parser.add_argument('--llm-provider', choices=['ollama', 'gemini'], 
                        help='AI model provider')
    parser.add_argument('--ollama-model', help='Specific Ollama model to use')
    parser.add_argument('--gemini-model', help='Specific Gemini model to use')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    
    # Advanced options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--cache-descriptions', action='store_true', default=True,
                        help='Enable description caching')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set output directory
    os.environ['OUTPUT_DIR'] = args.output_dir
    
    try:
        generator = LLMsTxtGenerator()
        
        # Handle model listing
        if args.list_models:
            models = generator.model_manager.list_models()
            if args.llm_provider:
                models = {k: v for k, v in models.items() if v.provider == args.llm_provider}
            
            print(f"\n🤖 Available AI Models:")
            print("=" * 60)
            for model_id, config in models.items():
                status_icon = generator.model_manager._get_status_indicator(config.status)
                print(f"{status_icon} {config.display_name}")
                print(f"   Provider: {config.provider}")
                print(f"   Description: {config.description}")
                if config.estimated_ram_gb > 0:
                    print(f"   Estimated RAM: {config.estimated_ram_gb:.1f} GB")
                print()
            return
        
        # Interactive mode if no URL provided
        if not args.url:
            print("🚀 LLMs.txt Generator - Interactive Mode")
            print("=" * 50)
            
            # Get website URL
            while True:
                url = input("\n🌐 Enter website URL (e.g., https://docs.crawl4ai.com): ").strip()
                if url:
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    break
                print("❌ Please enter a valid URL")
            
            # Model selection first
            print(f"\n🤖 AI Model Selection:")
            print("Choose an AI model to generate page descriptions...")
            if not generator.interactive_model_selection():
                print("❌ No model selected. Exiting.")
                return
            
            # Get output options
            print(f"\n📁 Output Options:")
            print("1. Text only (llms.txt)")
            print("2. Text with full content (llms.txt + llms-full.txt)")
            print("3. Full content only (llms-full.txt)")
            print("4. JSON format")
            print("5. YAML format")
            
            output_choice = input("Choose output format (1-5, default: 2): ").strip()
            
            if output_choice == "1":
                include_full_text = False
                export_format = 'text'
                full_text_only = False
                # Ask for page count for text-only mode
                print(f"\n📊 Crawling Options:")
                max_pages = input(f"Max pages to crawl (default: 50): ").strip()
                max_pages = int(max_pages) if max_pages.isdigit() else 50
            elif output_choice == "3":
                include_full_text = True
                export_format = 'text'
                full_text_only = True
                max_pages = 100  # Auto-set higher limit for comprehensive crawling
                print(f"\n🎯 Full content mode: Auto-setting to {max_pages} pages for comprehensive coverage")
            elif output_choice == "4":
                include_full_text = True
                export_format = 'json'
                full_text_only = False
                # Ask for page count for JSON mode
                print(f"\n📊 Crawling Options:")
                max_pages = input(f"Max pages to crawl (default: 50): ").strip()
                max_pages = int(max_pages) if max_pages.isdigit() else 50
            elif output_choice == "5":
                include_full_text = True
                export_format = 'yaml'
                full_text_only = False
                # Ask for page count for YAML mode
                print(f"\n📊 Crawling Options:")
                max_pages = input(f"Max pages to crawl (default: 50): ").strip()
                max_pages = int(max_pages) if max_pages.isdigit() else 50
            else:  # Default: option 2
                include_full_text = True
                export_format = 'text'
                full_text_only = False
                # Ask for page count for default mode
                print(f"\n📊 Crawling Options:")
                max_pages = input(f"Max pages to crawl (default: 50): ").strip()
                max_pages = int(max_pages) if max_pages.isdigit() else 50
            
            args.url = url
            args.max_pages = max_pages
            args.format = export_format
            args.no_full_text = not include_full_text
            args.full_text_only = full_text_only
            
            # Skip model selection since we already did it
            model_already_selected = True
        
        # Handle model selection based on command line arguments
        model_already_selected = locals().get('model_already_selected', False)
        
        if not model_already_selected:
            if args.llm_provider or args.ollama_model or args.gemini_model:
                models = generator.model_manager.list_models()
                selected_model = None
                
                # Try to find exact model match first
                if args.ollama_model:
                    for model_id, config in models.items():
                        if config.provider == 'ollama' and config.model_id == args.ollama_model:
                            selected_model = config
                            break
                elif args.gemini_model:
                    for model_id, config in models.items():
                        if config.provider == 'gemini' and config.model_id == args.gemini_model:
                            selected_model = config
                            break
                elif args.llm_provider:
                    # Select best available model from provider
                    provider_models = [config for config in models.values() if config.provider == args.llm_provider]
                    if provider_models:
                        # Sort by status (online first) and then by display name
                        provider_models.sort(key=lambda x: (x.status != 'online', x.display_name))
                        selected_model = provider_models[0]
                
                if selected_model:
                    generator.model_manager.current_model = selected_model
                    print(f"🤖 Selected model: {selected_model.display_name} ({selected_model.provider})")
                else:
                    print(f"❌ Could not find specified model. Available models:")
                    for model_id, config in models.items():
                        if not args.llm_provider or config.provider == args.llm_provider:
                            print(f"   • {config.model_id} ({config.provider})")
                    return
            else:
                # Interactive model selection if no model specified
                if not generator.interactive_model_selection():
                    print("❌ No model selected. Exiting.")
                    return
        
        # Set up parameters
        include_full_text = not args.no_full_text
        
        # Run the generator
        print(f"\n🚀 Starting generation for {args.url}")
        print(f"📊 Max pages: {args.max_pages}")
        print(f"📁 Format: {args.format}")
        print(f"📄 Include full text: {include_full_text}")
        if args.full_text_only:
            print(f"🎯 Mode: Full text only (no descriptions)")
        
        asyncio.run(generator.generate_llmstxt(
            base_url=args.url,
            max_pages=args.max_pages,
            export_format=args.format,
            include_full_text=include_full_text,
            parallel_workers=args.parallel_crawl,
            batch_size=args.batch_size,
            max_full_text_pages=args.max_full_pages,
            full_text_only=args.full_text_only,
            crawl_strategy=args.crawl_strategy,
            safety_limit=args.safety_limit
        ))
        
        print("\n🎉 Generation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Generation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
