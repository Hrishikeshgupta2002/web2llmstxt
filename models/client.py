#!/usr/bin/env python3
"""
AI Client and Model Manager for LLMs.txt Generator.

This module contains the ModelManager class for handling available AI models
and the AIClient class for generating content using various AI providers.
"""

import os
import json
import time
import hashlib
import logging
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from models.config_types import ModelConfig

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages available AI models similar to Cline's model selection with enhanced features"""
    
    def __init__(self):
        self.available_models = {}
        self.current_model = None
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.model_cache_file = os.path.join(os.getenv('OUTPUT_DIR', './output'), 'model_cache.json')
        self.description_cache = {}
        self._load_description_cache()
        
    def _load_description_cache(self):
        """Load cached descriptions to avoid regenerating unchanged summaries"""
        cache_file = os.path.join(os.getenv('OUTPUT_DIR', './output'), 'description_cache.json')
        cache_enabled = os.getenv('CACHE_DESCRIPTIONS', 'true').lower() == 'true'
        
        if cache_enabled and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.description_cache = json.load(f)
                logger.info(f"Loaded {len(self.description_cache)} cached descriptions")
            except Exception as e:
                logger.warning(f"Could not load description cache: {e}")
                
    def _save_description_cache(self):
        """Save description cache"""
        cache_enabled = os.getenv('CACHE_DESCRIPTIONS', 'true').lower() == 'true'
        if not cache_enabled:
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
        cache_enabled = os.getenv('CACHE_DESCRIPTIONS', 'true').lower() == 'true'
        if not cache_enabled:
            return None
            
        content_hash = self._get_content_hash(title, content)
        cache_key = f"{url}:{content_hash}"
        return self.description_cache.get(cache_key)
    
    def cache_description(self, url: str, title: str, content: str, description: str):
        """Cache a generated description"""
        cache_enabled = os.getenv('CACHE_DESCRIPTIONS', 'true').lower() == 'true'
        if not cache_enabled:
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
            logger.error(f"❌ Model {model_id} warm-up failed: {e}")
            return False
    
    def _get_status_indicator(self, status: str) -> str:
        """Get status indicator emoji"""
        indicators = {
            "online": "🟢",
            "offline": "🔴", 
            "unknown": "🟡",
            "available": "✅"
        }
        return indicators.get(status, "❓")
    
    def _check_ram_availability(self, required_ram_gb: float) -> Tuple[bool, str]:
        """Check if system has enough RAM for model"""
        try:
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            is_available = available_ram_gb >= required_ram_gb
            status = f"{available_ram_gb:.1f}GB available"
            return is_available, status
        except:
            return True, "RAM check unavailable"
    
    def list_models(self) -> Dict[str, ModelConfig]:
        """List all available models from both Ollama and Gemini"""
        models = {}
        
        # Add Ollama models
        ollama_models = self._list_ollama_models()
        models.update(ollama_models)
        
        # Add Gemini models
        gemini_models = self._list_gemini_models()
        models.update(gemini_models)
        
        self.available_models = models
        return models
    
    def _list_ollama_models(self) -> Dict[str, ModelConfig]:
        """List available Ollama models"""
        models = {}
        
        if not self.check_ollama_status():
            return models
        
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                for model_info in data.get('models', []):
                    model_name = model_info['name']
                    
                    # Create enhanced model config
                    config = ModelConfig(
                        provider="ollama",
                        model_id=model_name,
                        display_name=self._build_display_name(model_name),
                        description=f"Local Ollama model - {model_name}",
                        max_tokens=8192,
                        temperature=0.7,
                        status="available",
                        estimated_ram_gb=self._estimate_ram_requirements(model_name)
                    )
                    
                    models[f"ollama_{model_name}"] = config
                    
        except Exception as e:
            logger.debug(f"Error listing Ollama models: {e}")
        
        return models
    
    def _build_display_name(self, model_name: str) -> str:
        """Build enhanced display name for model"""
        # Remove common suffixes and format nicely
        clean_name = model_name.replace(':latest', '').replace('_', ' ')
        
        # Capitalize first letter of each word
        parts = clean_name.split('-')
        formatted_parts = []
        for part in parts:
            if part.replace('.', '').replace('b', '').isdigit():  # Size indicators
                formatted_parts.append(part.upper())
            else:
                formatted_parts.append(part.capitalize())
        
        return ' '.join(formatted_parts)
    
    def _estimate_ram_requirements(self, model_name: str) -> float:
        """Estimate RAM requirements for model"""
        model_lower = model_name.lower()
        
        # Simple size estimation based on model name
        if '70b' in model_lower or '72b' in model_lower:
            return 40.0
        elif '13b' in model_lower or '14b' in model_lower:
            return 8.0
        elif '7b' in model_lower or '8b' in model_lower:
            return 4.0
        elif '3b' in model_lower or '1b' in model_lower:
            return 2.0
        else:
            return 4.0  # Default estimate
    
    def _list_gemini_models(self) -> Dict[str, ModelConfig]:
        """List available Gemini models (always shown for setup guidance)"""
        models = {}
        
        # Predefined Gemini models with their capabilities
        gemini_models = [
            {
                'id': 'gemini-1.5-flash',
                'name': 'Gemini 1.5 Flash',
                'description': 'Fast and efficient model optimized for speed',
                'max_tokens': 1048576,
                'supports_vision': True
            },
            {
                'id': 'gemini-1.5-pro',
                'name': 'Gemini 1.5 Pro',
                'description': 'Advanced model with enhanced reasoning capabilities',
                'max_tokens': 2097152,
                'supports_vision': True
            },
            {
                'id': 'gemini-pro',
                'name': 'Gemini Pro',
                'description': 'Production-ready model for complex tasks',
                'max_tokens': 32768,
                'supports_vision': False
            }
        ]
        
        for model_info in gemini_models:
            config = ModelConfig(
                provider="gemini",
                model_id=model_info['id'],
                display_name=model_info['name'],
                description=model_info['description'],
                max_tokens=model_info['max_tokens'],
                temperature=0.7,
                supports_vision=model_info['supports_vision'],
                status="available" if self.gemini_api_key else "offline"
            )
            
            models[f"gemini_{model_info['id']}"] = config
        
        return models


class AIClient:
    """AI client for generating content using various providers"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.current_model = None
        self._gemini_client = None
        
    def set_model(self, model_config: ModelConfig):
        """Set the current model for generation"""
        self.current_model = model_config
        self.model_manager.current_model = model_config
        
        # Initialize provider-specific clients
        if model_config.provider == "gemini":
            self._init_gemini_client()
    
    def _init_gemini_client(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.model_manager.gemini_api_key)
            self._gemini_client = genai
            logger.debug("Gemini client initialized")
        except ImportError:
            logger.error("google-generativeai not installed. Please install with: pip install google-generativeai")
            self._gemini_client = None
    
    def _get_adaptive_timeout(self, model_config: ModelConfig) -> int:
        """Get adaptive timeout based on model type"""
        if model_config.provider == "ollama":
            return int(os.getenv('LOCAL_MODEL_TIMEOUT', '180'))
        else:
            return 60  # Cloud models are typically faster
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_content(self, prompt: str, model_config: Optional[ModelConfig] = None) -> Optional[str]:
        """Generate content using the current or specified model"""
        if model_config is None:
            model_config = self.current_model
            
        if not model_config:
            logger.error("No model configured for generation")
            return None
        
        try:
            if model_config.provider == "ollama":
                return self._generate_ollama(prompt, model_config)
            elif model_config.provider == "gemini":
                return self._generate_gemini(prompt, model_config)
            else:
                logger.error(f"Unsupported provider: {model_config.provider}")
                return None
                
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_ollama(self, prompt: str, model_config: ModelConfig) -> Optional[str]:
        """Generate content using Ollama"""
        if not self.model_manager.check_ollama_status():
            logger.error("Ollama is not running")
            return None
        
        try:
            timeout = self._get_adaptive_timeout(model_config)
            
            response = requests.post(
                f"{self.model_manager.ollama_base_url}/api/generate",
                json={
                    "model": model_config.model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": model_config.temperature,
                        "num_predict": int(os.getenv('MAX_GEN_OUTPUT_TOKENS', '1024'))
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama request failed with status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_gemini(self, prompt: str, model_config: ModelConfig) -> Optional[str]:
        """Generate content using Gemini"""
        if not self._gemini_client:
            self._init_gemini_client()
            
        if not self._gemini_client:
            logger.error("Gemini client not available")
            return None
        
        try:
            model = self._gemini_client.GenerativeModel(model_config.model_id)
            
            generation_config = self._gemini_client.types.GenerationConfig(
                temperature=model_config.temperature,
                max_output_tokens=int(os.getenv('MAX_GEN_OUTPUT_TOKENS', '1024'))
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return None