"""
Manages AI model configurations, availability, and metadata.

This module includes the ModelManager class which is responsible for:
- Loading and saving description caches.
- Checking the status of Ollama and specific models.
- Listing available models from different providers (Ollama, Gemini).
- Formatting model display names and estimating resource requirements.
"""
import os
import json
import hashlib
import psutil
import re
import requests
from typing import Dict, List, Optional, Tuple, Any

from llmsgen.models.config_types import ModelConfig
from llmsgen.config import logger, CACHE_DESCRIPTIONS, OUTPUT_DIR

class ModelManager:
    """Manages available AI models similar to Cline's model selection with enhanced features"""

    def __init__(self):
        self.available_models: Dict[str, ModelConfig] = {}
        self.current_model: Optional[ModelConfig] = None
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.model_cache_file = os.path.join(OUTPUT_DIR, 'model_cache.json')
        self.ollama_library_cache: Dict = {} # Consider defining a type for this
        self.description_cache: Dict[str, str] = {}
        self._load_description_cache()

    def _load_description_cache(self):
        """Load cached descriptions to avoid regenerating unchanged summaries"""
        cache_file = os.path.join(OUTPUT_DIR, 'description_cache.json')
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
        cache_file = os.path.join(OUTPUT_DIR, 'description_cache.json')
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
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
        except requests.exceptions.RequestException:
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
                # result = response.json() # Not used
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
                'display': 'DeepSeek R1', 'category': 'reasoning', 'emoji': '🧠',
                'description': 'Advanced reasoning model', 'estimated_ram': 8.0
            },
            'deepseek': {
                'display': 'DeepSeek', 'category': 'general', 'emoji': '🤖',
                'description': 'Efficient general model', 'estimated_ram': 4.0
            },
            'gemma': {
                'display': 'Gemma', 'category': 'general', 'emoji': '💎',
                'description': 'Google open model', 'estimated_ram': 3.0
            },
            'qwen': {
                'display': 'Qwen', 'category': 'general', 'emoji': '🚀',
                'description': 'Alibaba multilingual model', 'estimated_ram': 4.0
            },
            'llama': {
                'display': 'Llama', 'category': 'general', 'emoji': '🦙',
                'description': 'Meta open model', 'estimated_ram': 6.0
            },
            'codestral': {
                'display': 'Codestral', 'category': 'code', 'emoji': '💻',
                'description': 'Mistral coding model', 'estimated_ram': 5.0
            },
            'codegemma': {
                'display': 'CodeGemma', 'category': 'code', 'emoji': '💻',
                'description': 'Google coding model', 'estimated_ram': 4.0
            },
            'phi': {
                'display': 'Phi', 'category': 'general', 'emoji': '🔬',
                'description': 'Microsoft efficient model', 'estimated_ram': 2.0
            },
            'mistral': {
                'display': 'Mistral', 'category': 'general', 'emoji': '🌪️',
                'description': 'Mistral AI model', 'estimated_ram': 4.0
            },
            'granite': {
                'display': 'Granite', 'category': 'general', 'emoji': '🪨',
                'description': 'IBM enterprise model', 'estimated_ram': 5.0
            },
            'solar': {
                'display': 'Solar', 'category': 'general', 'emoji': '☀️',
                'description': 'Upstage efficient model', 'estimated_ram': 4.0
            }
        }

        for family_key, family_info in model_families.items():
            if family_key in base_name:
                return family_info

        return {
            'display': base_name.title(), 'category': 'general', 'emoji': '🤖',
            'description': f'{base_name.title()} model', 'estimated_ram': 4.0
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
        size_patterns = [
            r'(\d+(?:\.\d+)?)[Bb]', r'(\d+(?:\.\d+)?)[Mm]', r'(\d+(?:\.\d+)?)[Kk]',
            r'(\d+(?:\.\d+)?)-?[Bb]illion', r'(\d+(?:\.\d+)?)-?[Mm]illion',
        ]

        for pattern in size_patterns:
            match = re.search(pattern, model_name, re.IGNORECASE)
            if match:
                size_num = match.group(1)
                size_unit = model_name[match.start():match.end()]

                if 'b' in size_unit.lower() or 'billion' in size_unit.lower(): return f"{size_num}B"
                elif 'm' in size_unit.lower() or 'million' in size_unit.lower(): return f"{size_num}M"
                elif 'k' in size_unit.lower(): return f"{size_num}K"
        return ""

    def _extract_size_indicator(self, tag: str) -> str:
        """Extract size from tag (e.g., '7b', '13b-chat', 'instruct')"""
        size_patterns = [r'(\d+(?:\.\d+)?)[Bb]', r'(\d+(?:\.\d+)?)[Mm]', r'(\d+(?:\.\d+)?)[Kk]']
        for pattern in size_patterns:
            match = re.search(pattern, tag)
            if match: return match.group(0).upper()

        if 'tiny' in tag.lower(): return 'Tiny'
        if 'small' in tag.lower(): return 'Small'
        if 'medium' in tag.lower(): return 'Medium'
        if 'large' in tag.lower(): return 'Large'
        if 'xl' in tag.lower(): return 'XL'
        if 'instruct' in tag.lower(): return 'Instruct'
        if 'chat' in tag.lower(): return 'Chat'
        if 'code' in tag.lower(): return 'Code'
        return ""

    def _estimate_ram_requirements(self, model_name: str, family_info: Dict[str, Any]) -> float:
        """Estimate RAM requirements for model"""
        base_ram = family_info.get('estimated_ram', 4.0)
        size_str = self._extract_model_size(model_name)
        if size_str:
            size_num_match = re.findall(r'\d+(?:\.\d+)?', size_str)
            if not size_num_match: return base_ram # Should not happen if size_str is valid
            size_num = float(size_num_match[0])
            if 'B' in size_str: return max(size_num * 2, base_ram)
            if 'M' in size_str: return max(size_num * 0.002, base_ram)
        return base_ram

    def _check_ram_availability(self, required_ram_gb: float) -> Tuple[bool, str]:
        """Check if system has enough RAM for the model"""
        try:
            mem = psutil.virtual_memory()
            available_ram_gb = mem.available / (1024**3)
            total_ram_gb = mem.total / (1024**3)

            if required_ram_gb > available_ram_gb:
                return False, f"⚠️  Model requires ~{required_ram_gb:.1f}GB RAM, but only {available_ram_gb:.1f}GB available (Total: {total_ram_gb:.1f}GB)"
            if required_ram_gb > available_ram_gb * 0.8:
                return True, f"⚠️  Model requires ~{required_ram_gb:.1f}GB RAM, using {(required_ram_gb/available_ram_gb)*100:.0f}% of available RAM"
            return True, f"✅ Model requires ~{required_ram_gb:.1f}GB RAM, {available_ram_gb:.1f}GB available"
        except Exception as e:
            return True, f"❓ Could not check RAM: {e}"

    def _get_status_indicator(self, status: str) -> str:
        """Get status indicator emoji"""
        return {'online': '🟢', 'available': '🟢', 'offline': '🔴', 'unknown': '🟡'}.get(status.lower(), '🟡')

    def list_models(self) -> Dict[str, ModelConfig]:
        """List available models from all providers"""
        models = {}
        models.update(self._list_ollama_models())
        if self.gemini_api_key:
            models.update(self._list_gemini_models())
        return models

    def _list_ollama_models(self) -> Dict[str, ModelConfig]:
        """List available Ollama models"""
        models: Dict[str, ModelConfig] = {}
        if not self.check_ollama_status():
            logger.debug("Ollama is not running - no local models available")
            return models

        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            response.raise_for_status()
            ollama_data = response.json()

            for model_info in ollama_data.get('models', []):
                model_name = model_info.get('name', '')
                if not model_name: continue

                display_name, family_info = self._format_model_name_enhanced(model_name)
                estimated_ram = self._estimate_ram_requirements(model_name, family_info)
                status = 'online' if self.check_model_loaded(model_name) else 'available'

                models[model_name] = ModelConfig(
                    provider='ollama', model_id=model_name, display_name=display_name,
                    description=family_info.get('description', ''), status=status,
                    tags=model_info.get('details', {}).get('families', []),
                    pulls=model_info.get('details', {}).get('quantization_level', ''),
                    size_info=f"{model_info.get('size', 0) / (1024**3):.1f}GB",
                    estimated_ram_gb=estimated_ram
                )
        except requests.exceptions.RequestException as e: logger.error(f"Failed to fetch Ollama models: {e}")
        except Exception as e: logger.error(f"Error processing Ollama models: {e}")
        return models

    def _list_gemini_models(self) -> Dict[str, ModelConfig]:
        """List available Gemini models"""
        models: Dict[str, ModelConfig] = {}
        if not self.gemini_api_key:
            logger.warning("⚠️  No Gemini API key provided")
            return models

        gemini_model_data = [
            {'id': 'gemini-2.5-flash-exp', 'name': '🌟 Gemini 2.5 Flash (Experimental)', 'desc': 'Latest cutting-edge model', 'max_t': 8192, 'cost': 0.075},
            {'id': 'gemini-2.0-flash-exp', 'name': '⚡ Gemini 2.0 Flash (Experimental)', 'desc': 'Latest experimental model', 'max_t': 8192, 'cost': 0.075},
            {'id': 'gemini-exp-1206', 'name': '🧪 Gemini Experimental 1206', 'desc': 'Cutting-edge experimental', 'max_t': 8192, 'cost': 0.075},
            {'id': 'gemini-1.5-flash', 'name': '💫 Gemini 1.5 Flash', 'desc': 'Fast and efficient', 'max_t': 8192, 'cost': 0.075},
            {'id': 'gemini-1.5-flash-8b', 'name': '💨 Gemini 1.5 Flash 8B', 'desc': 'Lightweight and fast', 'max_t': 8192, 'cost': 0.0375},
            {'id': 'gemini-1.5-pro', 'name': '🚀 Gemini 1.5 Pro', 'desc': 'Most capable model', 'max_t': 32768, 'cost': 3.5},
            {'id': 'gemini-1.0-pro', 'name': '⚡ Gemini 1.0 Pro', 'desc': 'Reliable general use', 'max_t': 32768, 'cost': 0.5}
        ]

        for m_info in gemini_model_data:
            models[m_info['id']] = ModelConfig(
                provider='gemini', model_id=m_info['id'], display_name=m_info['name'],
                description=m_info['desc'], max_tokens=m_info['max_t'],
                cost_per_1k_tokens=m_info['cost'], status='available', is_available_remote=True
            )
        return models
