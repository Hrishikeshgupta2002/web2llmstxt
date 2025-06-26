import time
import requests
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from llmsgen.models.manager import ModelManager
from llmsgen.models.config_types import ModelConfig
from llmsgen.config import logger, MAX_GEN_OUTPUT_TOKENS, LOCAL_MODEL_TIMEOUT

# Attempt to import google.generativeai and handle if not found
try:
    import google.generativeai as genai
except ImportError:
    genai = None # Placeholder if not installed

class AIClient:
    """Enhanced AI client with retry logic and configurable limits"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.gemini_client = None
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.local_model_stats = {
            'total_requests': 0, 'total_tokens_generated': 0,
            'total_time_seconds': 0, 'timeouts': 0, 'errors': 0
        }

    def _get_gemini_client(self):
        """Get or create Gemini client"""
        if self.gemini_client is None:
            if genai is None:
                 raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
            try:
                api_key = self.model_manager.gemini_api_key
                if not api_key:
                    raise ValueError("Gemini API key not found")
                genai.configure(api_key=api_key)
                self.gemini_client = genai
                logger.debug("Gemini client initialized successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Gemini client: {e}")
        return self.gemini_client

    def _get_adaptive_timeout(self, model_config: ModelConfig) -> int:
        """Get adaptive timeout based on model type and estimated RAM requirements"""
        if model_config.provider == 'ollama':
            base_timeout = LOCAL_MODEL_TIMEOUT
            if model_config.estimated_ram_gb > 12: return int(base_timeout * 2)
            if model_config.estimated_ram_gb > 8: return int(base_timeout * 1.5)
            return base_timeout
        return 60  # Default for cloud models

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_content(self, prompt: str, model_config: Optional[ModelConfig] = None) -> Optional[str]:
        """Generate content using the configured model with retry logic"""
        current_model_config = model_config if model_config is not None else self.model_manager.current_model

        if not current_model_config:
            raise ValueError("No model configuration available")

        logger.debug(f"Generating content with {current_model_config.provider}:{current_model_config.model_id}")

        if current_model_config.provider == 'ollama':
            return self._generate_ollama(prompt, current_model_config)
        if current_model_config.provider == 'gemini':
            return self._generate_gemini(prompt, current_model_config)
        raise ValueError(f"Unsupported provider: {current_model_config.provider}")

    def _generate_ollama(self, prompt: str, model_config: ModelConfig) -> Optional[str]:
        """Generate content using Ollama with retry logic and connection pooling"""
        start_time = time.time()
        timeout = self._get_adaptive_timeout(model_config)
        self.local_model_stats['total_requests'] += 1

        try:
            response = self.session.post(
                f"{self.model_manager.ollama_base_url}/api/generate",
                json={
                    "model": model_config.model_id, "prompt": prompt, "stream": False,
                    "options": {
                        "temperature": model_config.temperature,
                        "num_predict": min(model_config.max_tokens, MAX_GEN_OUTPUT_TOKENS),
                        "num_ctx": 4096, "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.1
                    }
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result.get('response', '').strip()

            duration = time.time() - start_time
            self.local_model_stats['total_time_seconds'] += duration
            self.local_model_stats['total_tokens_generated'] += len(generated_text.split())
            logger.debug(f"🚀 Generated {len(generated_text)} chars in {duration:.2f}s")
            return generated_text

        except requests.exceptions.Timeout:
            self.local_model_stats['timeouts'] += 1
            logger.warning(f"Ollama request timed out after {timeout}s for model {model_config.model_id}")
            raise
        except requests.exceptions.RequestException as e: # Catches ConnectionError too
            self.local_model_stats['errors'] += 1
            logger.error(f"Ollama request error for {model_config.model_id}: {e}")
            raise
        except Exception as e: # Catch any other unexpected errors
            self.local_model_stats['errors'] += 1
            logger.error(f"Unexpected Ollama generation error for {model_config.model_id}: {e}")
            raise

    def _generate_gemini(self, prompt: str, model_config: ModelConfig) -> Optional[str]:
        """Generate content using Gemini with retry logic"""
        try:
            client = self._get_gemini_client()
            if client is None: # Should be caught by _get_gemini_client, but as a safeguard
                raise RuntimeError("Gemini client not available/installed.")

            generation_config = {
                "temperature": model_config.temperature,
                "max_output_tokens": min(model_config.max_tokens, MAX_GEN_OUTPUT_TOKENS),
                "top_p": 0.95,
            }

            model = client.GenerativeModel(
                model_config.model_id,
                generation_config=generation_config
            )
            response = model.generate_content(prompt)

            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()

            logger.warning(f"No content generated by Gemini for model {model_config.model_id}")
            return None

        except Exception as e:
            logger.error(f"Gemini generation error for {model_config.model_id}: {e}")
            raise # Re-raise to be caught by tenacity

    def print_local_model_performance(self):
        """Print performance statistics for local model usage"""
        stats = self.local_model_stats
        if stats['total_requests'] == 0: return

        avg_time = stats['total_time_seconds'] / stats['total_requests']
        avg_tokens = stats['total_tokens_generated'] / stats['total_requests']
        tps = stats['total_tokens_generated'] / stats['total_time_seconds'] if stats['total_time_seconds'] > 0 else 0
        success_rate = ((stats['total_requests'] - stats['errors']) / stats['total_requests']) * 100

        report = [
            "\n" + "="*60, "🏁 LOCAL MODEL PERFORMANCE REPORT", "="*60,
            "📊 Request Statistics:",
            f"   • Total requests: {stats['total_requests']}",
            f"   • Successful requests: {stats['total_requests'] - stats['errors']}",
            f"   • Success rate: {success_rate:.1f}%",
            f"   • Timeouts: {stats['timeouts']}",
            f"   • Errors: {stats['errors']}",
            "\n⚡ Performance Metrics:",
            f"   • Average response time: {avg_time:.2f}s",
            f"   • Average tokens per response: {avg_tokens:.0f}",
            f"   • Inference speed: {tps:.1f} tokens/second",
            f"   • Total processing time: {stats['total_time_seconds']:.1f}s",
            f"   • Total tokens generated: {stats['total_tokens_generated']:,}",
            f"\n🎯 Performance Rating: {'🚀 Excellent' if tps > 50 else '✅ Good' if tps > 20 else '⚠️ Fair' if tps > 10 else '🐌 Slow'}",
            "="*60
        ]
        print('\n'.join(report))
