#!/usr/bin/env python3
"""
LLMs.txt Generator Module

Handles the generation of llms.txt and llms-full.txt files with enhanced features.
"""

import os
import json
import yaml
import time
import logging
import re
import hashlib
import psutil
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import tempfile
from tenacity import retry, stop_after_attempt, wait_exponential

from models.client import AIClient, ModelManager
from models.config_types import ModelConfig
from utils.text_utils import clean_text, extract_key_sentences
from utils.file_utils import ensure_output_dir, write_safe_file
from crawler.web_crawler import WebCrawler

logger = logging.getLogger(__name__)

class LLMsTxtGenerator:
    """Generates llms.txt files with AI-powered descriptions and content extraction"""
    
    def __init__(self):
        self.model_manager = None
        self.ai_client = None
        self.crawler = WebCrawler()
        self.domain = ""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.getenv('OUTPUT_DIR', './output')
        
        # Create output directory
        ensure_output_dir(self.output_dir)
        
    def set_clients(self, model_manager, ai_client: AIClient):
        """Set the model manager and AI client"""
        self.model_manager = model_manager
        self.ai_client = ai_client
        
    def interactive_model_selection(self):
        """Interactive model selection with enhanced features"""
        # Initialize model manager if not set
        if not self.model_manager:
            self.model_manager = ModelManager()
            
        print("\n" + "="*80)
        print("🤖 AI Model Selection")
        print("="*80)
        
        # Check if Ollama is running
        if not self.model_manager.check_ollama_status():
            print("⚠️  Ollama is not running. Only Gemini models will be available.")
            print("   To use local models, start Ollama with 'ollama serve'\n")
        
        # List available models
        models = self.model_manager.list_models()
        if not models:
            print("❌ No models available. Please check your setup.")
            return None
            
        # Display models with enhanced information
        print("Available models:\n")
        for i, (model_id, config) in enumerate(models.items(), 1):
            status_icon = self.model_manager._get_status_indicator(config.status)
            
            # Build model info line
            model_info = f"{i:2d}. {status_icon} {config.display_name}"
            
            # Add memory requirements for local models
            if config.provider == "ollama" and config.estimated_ram_gb > 0:
                ram_available, ram_status = self.model_manager._check_ram_availability(config.estimated_ram_gb)
                ram_icon = "✅" if ram_available else "⚠️"
                model_info += f" {ram_icon} (~{config.estimated_ram_gb:.1f}GB RAM)"
            
            # Add setup indicator for Gemini models without API key
            elif config.provider == "gemini" and not self.model_manager.gemini_api_key:
                model_info += " 🔑 (Setup Required)"
                
            print(model_info)
            
            # Add description and tags
            if config.description:
                print(f"     {config.description}")
            if config.tags:
                tags_str = " ".join(f"#{tag}" for tag in config.tags[:3])
                print(f"     {tags_str}")
            print()
        
        # Get user selection
        while True:
            try:
                choice = input(f"Select a model (1-{len(models)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                    
                model_index = int(choice) - 1
                if 0 <= model_index < len(models):
                    selected_model_id = list(models.keys())[model_index]
                    selected_config = models[selected_model_id]
                    
                    # Check if Gemini model is selected but no API key configured
                    if selected_config.provider == "gemini" and not self.model_manager.gemini_api_key:
                        return self._handle_gemini_setup(selected_config)
                    
                    # Warm up local models
                    if selected_config.provider == "ollama":
                        if not self.model_manager.warm_up_model(selected_config.model_id):
                            print(f"❌ Failed to warm up {selected_config.display_name}")
                            continue
                    
                    print(f"✅ Selected: {selected_config.display_name}")
                    return selected_config
                else:
                    print("❌ Invalid selection. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("❌ Invalid input. Please enter a number or 'q'.")
    
    def _handle_gemini_setup(self, selected_config):
        """Handle Gemini API key setup when model is selected without key"""
        print(f"\n🔑 Gemini API Key Required")
        print("=" * 50)
        print(f"You selected: {selected_config.display_name}")
        print("❌ No Gemini API key configured.")
        print("\n📋 To use Gemini models, you need a Google AI API key:")
        print("   1. Visit: https://aistudio.google.com/app/apikey")
        print("   2. Create a new API key")
        print("   3. Copy the key")
        print("\n🔧 Setup Options:")
        print("   [1] Enter API key now (saves to .env file)")
        print("   [2] Setup manually later")
        print("   [3] Choose a different model")
        
        while True:
            setup_choice = input("\nChoose setup option (1-3): ").strip()
            
            if setup_choice == "1":
                return self._setup_gemini_key_interactive(selected_config)
            elif setup_choice == "2":
                self._show_manual_setup_instructions()
                return None
            elif setup_choice == "3":
                return self.interactive_model_selection()  # Restart model selection
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    def _setup_gemini_key_interactive(self, selected_config):
        """Interactive Gemini API key setup"""
        print(f"\n🔑 Enter Gemini API Key")
        print("=" * 40)
        print("📝 Paste your Google AI API key below:")
        print("   (The key will be saved to your .env file)")
        
        while True:
            api_key = input("API Key: ").strip()
            
            if not api_key:
                print("❌ Please enter a valid API key.")
                continue
            
            if len(api_key) < 20:  # Basic validation
                print("❌ API key seems too short. Please check and try again.")
                continue
            
            # Save to .env file
            try:
                from utils.file_utils import save_api_key_to_env
                save_api_key_to_env(api_key, "GEMINI_API_KEY")
                
                # Update the model manager
                self.model_manager.gemini_api_key = api_key
                os.environ['GEMINI_API_KEY'] = api_key
                
                print("✅ API key saved successfully!")
                print(f"🎯 Selected model: {selected_config.display_name}")
                
                # Update model status
                selected_config.status = "available"
                return selected_config
                
            except Exception as e:
                print(f"❌ Failed to save API key: {e}")
                print("🔧 You can manually add it to your .env file:")
                print(f"   GEMINI_API_KEY={api_key}")
                return None
    
    def _show_manual_setup_instructions(self):
        """Show manual setup instructions"""
        print(f"\n📋 Manual Setup Instructions")
        print("=" * 40)
        print("1. Create or edit a .env file in your project root")
        print("2. Add this line:")
        print("   GEMINI_API_KEY=your_actual_api_key_here")
        print("3. Restart the script")
        print("\n💡 Or run: python scripts/generate_llms.py --setup-env")
        print("   This creates a sample .env file for you to edit.")
    
    def interactive_crawling_mode_selection(self):
        """Interactive crawling mode selection"""
        print("\n" + "="*80)
        print("🕷️ Crawling Mode Selection")
        print("="*80)
        print("Choose how you want to crawl the website:\n")
        
        print("1. 📄 Normal LLMs.txt (Recommended)")
        print("   • Crawls the main page + direct links from homepage")
        print("   • Fast and efficient for most websites")
        print("   • Good for getting an overview of the site")
        print("   • Typically finds 10-50 pages\n")
        
        print("2. 🌊 Full Deep Crawl LLMs.txt (Comprehensive)")
        print("   • Discovers ALL links from main page recursively")
        print("   • Crawls the entire domain until all pages are found")
        print("   • Takes longer but captures everything")
        print("   • Can find hundreds or thousands of pages")
        print("   • ⚠️  Use with caution on large sites\n")
        
        print("3. 🗺️ Sitemap-Based Crawl (Most Efficient)")
        print("   • Uses website's sitemap.xml for complete URL discovery")
        print("   • Fastest way to find ALL pages on a website")
        print("   • Most reliable and comprehensive method")
        print("   • Can discover thousands of pages instantly")
        print("   • ✅ Recommended for large sites\n")
        
        while True:
            try:
                mode_choice = input("Select crawling mode (1-3): ").strip()
                
                if mode_choice == "1":
                    print("✅ Selected: Normal LLMs.txt (main page + direct links)")
                    return "normal", False, None
                elif mode_choice == "2":
                    # Ask for confirmation for deep crawl
                    print("\n⚠️  Deep Crawl Confirmation")
                    print("Deep crawling will attempt to find and crawl ALL pages on the domain.")
                    print("This can take a very long time on large websites.")
                    
                    confirm = input("Are you sure you want to proceed? (y/N): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        print("✅ Selected: Full Deep Crawl (entire domain)")
                        return "comprehensive", True, None
                    else:
                        print("↩️  Returning to mode selection...")
                        continue
                elif mode_choice == "3":
                    return self._handle_sitemap_selection()
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled.")
                return "normal", False, None
    
    def _handle_sitemap_selection(self):
        """Handle sitemap URL input and validation"""
        print("\n🗺️ Sitemap-Based Crawling Setup")
        print("=" * 50)
        print("Options:")
        print("  [1] Auto-discover sitemap (tries common locations)")
        print("  [2] Provide specific sitemap URL")
        print("  [3] Back to crawling mode selection")
        
        while True:
            sitemap_choice = input("\nChoose sitemap option (1-3): ").strip()
            
            if sitemap_choice == "1":
                print("✅ Selected: Auto-discover sitemap")
                print("   Will try: /sitemap.xml, /sitemap_index.xml, /robots.txt")
                return "sitemap", False, "auto"
                
            elif sitemap_choice == "2":
                print("\n📝 Enter Sitemap URL")
                print("Examples:")
                print("  • https://example.com/sitemap.xml")
                print("  • https://example.com/sitemap_index.xml")
                print("  • https://example.com/sitemaps/main.xml")
                
                while True:
                    sitemap_url = input("\nSitemap URL: ").strip()
                    
                    if not sitemap_url:
                        print("❌ Please enter a valid sitemap URL.")
                        continue
                    
                    if not sitemap_url.startswith(('http://', 'https://')):
                        sitemap_url = 'https://' + sitemap_url
                    
                    if sitemap_url.endswith('.xml') or 'sitemap' in sitemap_url.lower():
                        print(f"✅ Selected: Custom sitemap ({sitemap_url})")
                        return "sitemap", False, sitemap_url
                    else:
                        print("⚠️  URL doesn't look like a sitemap. Continue anyway? (y/N)")
                        confirm = input().strip().lower()
                        if confirm in ['y', 'yes']:
                            print(f"✅ Selected: Custom URL ({sitemap_url})")
                            return "sitemap", False, sitemap_url
                        else:
                            continue
                            
            elif sitemap_choice == "3":
                return self.interactive_crawling_mode_selection()
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    def remove_page_separators(self, text: str) -> str:
        """Remove page separator markers from text"""
        separators = [
            r'\n--- Page \d+ ---\n',
            r'\n=== Page \d+ ===\n',
            r'\n\*\*\* Page \d+ \*\*\*\n',
            r'\n--- End of Page \d+ ---\n'
        ]
        
        for separator in separators:
            text = re.sub(separator, '\n\n', text, flags=re.IGNORECASE)
        
        return text
    
    def limit_pages(self, full_text: str, max_pages: int) -> str:
        """Limit the number of pages in full text output"""
        if max_pages <= 0:
            return full_text
            
        # Split by page separators and limit
        pages = re.split(r'\n--- Page \d+ ---\n', full_text)
        if len(pages) > max_pages:
            limited_pages = pages[:max_pages]
            full_text = '\n--- Page {} ---\n'.format(1).join(limited_pages)
            full_text += f"\n\n[Content truncated - showing first {max_pages} pages of {len(pages)} total]"
        
        return full_text
    
    def extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL for file naming"""
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    
    def _detect_hallucination(self, description: str, title: str, content: str) -> bool:
        """Detect potential hallucinations in AI-generated descriptions"""
        if not description or len(description.strip()) < 10:
            return True
            
        # Check for generic/templated responses
        generic_phrases = [
            "this page contains",
            "this website provides",
            "this article discusses",
            "the main content includes",
            "key topics covered",
            "important information about"
        ]
        
        description_lower = description.lower()
        if any(phrase in description_lower for phrase in generic_phrases):
            # Check if description contains specific content from the page
            title_words = set(title.lower().split())
            content_words = set(content.lower()[:500].split())  # First 500 chars
            description_words = set(description_lower.split())
            
            # If description has minimal overlap with actual content, it's likely hallucinated
            title_overlap = len(title_words & description_words) / max(len(title_words), 1)
            content_overlap = len(content_words & description_words) / max(len(content_words), 1)
            
            if title_overlap < 0.2 and content_overlap < 0.1:
                return True
        
        # Check for repetitive patterns
        words = description.split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # If any word appears more than 3 times in a short description, it's suspicious
            max_freq = max(word_freq.values())
            if max_freq > 3 and len(words) < 50:
                return True
        
        return False
    
    def _extract_key_sentences(self, title: str, content: str) -> str:
        """Extract key sentences from content for description generation"""
        return extract_key_sentences(title, content)
    
    def _clean_content_text(self, content: str) -> str:
        """Clean and normalize content text"""
        return clean_text(content)
    
    def _is_navigation_text(self, text: str) -> bool:
        """Check if text is likely navigation/menu content"""
        nav_indicators = [
            'menu', 'navigation', 'nav', 'breadcrumb', 'sidebar',
            'footer', 'header', 'skip to', 'toggle', 'dropdown',
            'click here', 'read more', 'learn more', 'see all',
            'view all', 'show more', 'load more'
        ]
        
        text_lower = text.lower().strip()
        if len(text_lower) < 5:
            return True
            
        # Short texts with nav indicators are likely navigation
        if len(text_lower) < 50:
            return any(indicator in text_lower for indicator in nav_indicators)
        
        return False
    
    def _is_good_sentence(self, sentence: str) -> bool:
        """Check if a sentence is good for content description"""
        sentence = sentence.strip()
        
        # Basic length check
        if len(sentence) < 20 or len(sentence) > 300:
            return False
        
        # Check for navigation text
        if self._is_navigation_text(sentence):
            return False
        
        # Must have some actual content words
        content_words = [word for word in sentence.split() if len(word) > 3]
        if len(content_words) < 3:
            return False
        
        # Avoid sentences that are mostly links or technical jargon
        if sentence.count('http') > 0 or sentence.count('www') > 0:
            return False
        
        # Avoid sentences with too many special characters
        special_char_ratio = sum(1 for c in sentence if not c.isalnum() and c not in ' .,!?-') / len(sentence)
        if special_char_ratio > 0.2:
            return False
        
        return True
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean individual sentence"""
        # Remove extra whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        # Remove leading/trailing punctuation artifacts
        sentence = re.sub(r'^[^\w]*', '', sentence)
        sentence = re.sub(r'[^\w.!?]*$', '', sentence)
        
        # Ensure proper capitalization
        if sentence and not sentence[0].isupper():
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize page titles"""
        if not title:
            return "Untitled Page"
        
        # Remove common title suffixes
        suffixes_to_remove = [
            r'\s*\|\s*.*$',  # Everything after |
            r'\s*-\s*.*$',   # Everything after -
            r'\s*::\s*.*$',  # Everything after ::
            r'\s*»\s*.*$',   # Everything after »
            r'\s*>\s*.*$',   # Everything after >
        ]
        
        for suffix in suffixes_to_remove:
            new_title = re.sub(suffix, '', title).strip()
            if len(new_title) >= 5:  # Only use cleaned version if it's long enough
                title = new_title
                break
        
        # Clean up common artifacts
        title = re.sub(r'\s+', ' ', title).strip()
        title = title.replace('&amp;', '&')
        title = title.replace('&lt;', '<')
        title = title.replace('&gt;', '>')
        title = title.replace('&quot;', '"')
        
        return title or "Untitled Page"
    
    def _create_content_description(self, title: str, content: str) -> str:
        """Create a description based on content analysis"""
        # Clean the content
        clean_content = self._clean_content_text(content)
        
        # Extract key sentences
        key_sentences = self._extract_key_sentences(title, clean_content)
        
        if not key_sentences:
            return f"Page about {title.lower()}"
        
        # Limit to reasonable length
        if len(key_sentences) > 200:
            sentences = key_sentences.split('. ')
            key_sentences = '. '.join(sentences[:2]) + '.'
        
        return key_sentences
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def generate_description_with_fallbacks(self, title: str, content: str, url: str = "") -> str:
        """Generate description with multiple fallback strategies"""
        if not self.ai_client:
            return self._create_content_description(title, content)
        
        # Check cache first
        cached_desc = self.model_manager.check_cached_description(url, title, content)
        if cached_desc:
            logger.debug(f"Using cached description for {url}")
            return cached_desc
        
        # Clean inputs
        clean_title = self._clean_title(title)
        clean_content = self._clean_content_text(content)
        
        # Create AI prompt with better context
        key_content = self._extract_key_sentences(clean_title, clean_content)
        
        prompt = f"""Create a concise, informative description for this webpage. The description should be 1-2 sentences and capture the main purpose/content.

Title: {clean_title}

Key Content: {key_content[:800]}

Instructions:
- Be specific and factual
- Avoid generic phrases like "this page contains" or "this website provides"
- Focus on what makes this page unique or valuable
- Keep it under 150 characters if possible
- Don't hallucinate information not present in the content

Description:"""

        try:
            # Generate with AI
            description = self.ai_client.generate_content(prompt)
            
            if description:
                description = description.strip()
                
                # Clean up the response
                if description.startswith('"') and description.endswith('"'):
                    description = description[1:-1]
                
                # Detect hallucination
                if self._detect_hallucination(description, clean_title, clean_content):
                    logger.warning(f"Detected potential hallucination for {url}, using fallback")
                    description = self._create_smart_fallback(clean_title, clean_content, url)
                else:
                    # Cache successful generation
                    self.model_manager.cache_description(url, title, content, description)
                
                return description
            
        except Exception as e:
            logger.warning(f"AI generation failed for {url}: {e}")
        
        # Fallback to content-based description
        return self._create_smart_fallback(clean_title, clean_content, url)
    
    def _create_smart_fallback(self, title: str, content: str, url: str) -> str:
        """Create intelligent fallback description"""
        # Extract domain for context
        try:
            domain = self.extract_domain_from_url(url)
        except:
            domain = "website"
        
        # Analyze content type
        content_lower = content.lower()
        
        # Check for specific content types
        if any(word in content_lower for word in ['tutorial', 'guide', 'how to', 'step by step']):
            return f"Tutorial or guide on {title.lower()}"
        elif any(word in content_lower for word in ['blog', 'article', 'post', 'news']):
            return f"Article about {title.lower()}"
        elif any(word in content_lower for word in ['product', 'service', 'pricing', 'buy', 'purchase']):
            return f"Product or service page for {title.lower()}"
        elif any(word in content_lower for word in ['about', 'company', 'team', 'history']):
            return f"Information about {domain}"
        elif any(word in content_lower for word in ['contact', 'email', 'phone', 'address']):
            return f"Contact information for {domain}"
        else:
            # Generic but informative fallback
            key_sentences = self._extract_key_sentences(title, content)
            if key_sentences and len(key_sentences) > 20:
                return key_sentences[:120] + "..." if len(key_sentences) > 120 else key_sentences
            else:
                return f"Page about {title.lower()} on {domain}"
    
    def process_urls_in_batches(self, pages: List[Dict], batch_size: int = 10, 
                               parallel_workers: int = 5) -> List[Dict]:
        """Process URLs in batches with parallel execution"""
        if not pages:
            return []
        
        logger.info(f"🔄 Processing {len(pages)} pages in batches of {batch_size} with {parallel_workers} workers")
        
        processed_pages = []
        total_batches = (len(pages) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(pages))
            batch = pages[start_idx:end_idx]
            
            logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch)} pages)")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = []
                for page in batch:
                    future = executor.submit(self._process_single_page, page)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=180)  # 3 minute timeout per page
                        if result:
                            processed_pages.append(result)
                    except Exception as e:
                        logger.error(f"Error processing page: {e}")
            
            # Brief pause between batches to be respectful
            if batch_num < total_batches - 1:
                time.sleep(1)
        
        logger.info(f"✅ Completed processing {len(processed_pages)}/{len(pages)} pages")
        return processed_pages
    
    def _process_single_page(self, page: Dict) -> Optional[Dict]:
        """Process a single page and generate description"""
        try:
            url = page.get('url', '')
            title = page.get('title', 'Untitled')
            content = page.get('content', '')
            
            # Skip pages with no meaningful content
            if not content or len(content.strip()) < 50:
                logger.debug(f"Skipping page with minimal content: {url}")
                return None
            
            # Generate description
            description = self.generate_description_with_fallbacks(title, content, url)
            
            # Create result
            result = {
                'url': url,
                'title': self._clean_title(title),
                'description': description,
                'content': content,
                'word_count': len(content.split()),
                'processed_at': datetime.now().isoformat()
            }
            
            logger.debug(f"✅ Processed: {title[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error processing page {page.get('url', 'unknown')}: {e}")
            return None
    
    async def generate_llmstxt(self, base_url: str, max_pages: int = 50, 
                             export_format: str = 'text', 
                             include_full_text: bool = True,
                             parallel_workers: int = None,
                             batch_size: int = 10,
                             max_full_text_pages: int = None,
                             full_text_only: bool = False,
                             crawl_strategy: str = 'systematic',
                             safety_limit: int = None,
                             comprehensive_crawl: bool = False,
                             sitemap_url: str = None):
        """Main method to generate llms.txt files"""
        start_time = time.time()
        self.domain = self.extract_domain_from_url(base_url)
        
        logger.info(f"🚀 Starting llms.txt generation for {base_url}")
        logger.info(f"📊 Parameters: max_pages={max_pages}, format={export_format}, workers={parallel_workers}")
        
        # Crawl website based on mode
        if crawl_strategy == "sitemap":
            logger.info(f"🗺️ Crawling website using sitemap strategy...")
            pages = await self.crawler.crawl_from_sitemap(base_url, sitemap_url, max_pages)
        else:
            crawl_mode = "comprehensive" if comprehensive_crawl else "normal"
            logger.info(f"🕷️ Crawling website with {crawl_mode} {crawl_strategy} strategy...")
            pages = await self.crawler.crawl_website(base_url, max_pages, comprehensive=comprehensive_crawl)
        
        if not pages:
            logger.error("❌ No pages found during crawling")
            return
        
        logger.info(f"📄 Found {len(pages)} pages to process")
        
        # Process pages to generate descriptions
        if not full_text_only:
            processed_pages = self.process_urls_in_batches(
                pages, 
                batch_size=batch_size, 
                parallel_workers=parallel_workers or 5
            )
        else:
            processed_pages = []
        
        # Create metadata
        metadata = {
            'domain': self.domain,
            'base_url': base_url,
            'generation_time': datetime.now().isoformat(),
            'total_pages_found': len(pages),
            'pages_processed': len(processed_pages),
            'generation_duration': time.time() - start_time,
            'parameters': {
                'max_pages': max_pages,
                'export_format': export_format,
                'include_full_text': include_full_text,
                'parallel_workers': parallel_workers,
                'batch_size': batch_size,
                'crawl_strategy': crawl_strategy,
                'comprehensive_crawl': comprehensive_crawl,
                'sitemap_url': sitemap_url if crawl_strategy == 'sitemap' else None
            }
        }
        
        # Write output files
        await self._write_output_files(
            base_url, processed_pages, pages, metadata, 
            export_format, include_full_text, full_text_only
        )
        
        # Print summary
        self._print_generation_summary(metadata, processed_pages)
        
        logger.info(f"✅ Generation completed in {time.time() - start_time:.2f} seconds")
    
    def _print_generation_summary(self, metadata: Dict, entries: List[Dict]):
        """Print a comprehensive generation summary"""
        print("\n" + "="*80)
        print("📊 GENERATION SUMMARY")
        print("="*80)
        print(f"🌐 Domain: {metadata['domain']}")
        print(f"📄 Pages Found: {metadata['total_pages_found']}")
        print(f"✅ Pages Processed: {metadata['pages_processed']}")
        print(f"⏱️  Duration: {metadata['generation_duration']:.2f} seconds")
        
        if entries:
            avg_desc_length = sum(len(entry.get('description', '')) for entry in entries) / len(entries)
            print(f"📝 Average Description Length: {avg_desc_length:.1f} characters")
        
        print("\n📁 Output files created in ./output/ directory")
        print("="*80)
    
    async def _write_output_files(self, base_url: str, llms_entries: List[Dict], 
                                pages: List[Dict], metadata: Dict, 
                                export_format: str, include_full_text: bool, full_text_only: bool = False):
        """Write output files in specified format"""
        ensure_output_dir()
        
        if export_format == 'text':
            await self._write_text_output(base_url, self.timestamp, llms_entries, pages, metadata, include_full_text, full_text_only)
        elif export_format == 'json':
            await self._write_json_output(self.domain, self.timestamp, llms_entries, pages, metadata, include_full_text, full_text_only)
        elif export_format == 'yaml':
            await self._write_yaml_output(self.domain, self.timestamp, llms_entries, pages, metadata, include_full_text, full_text_only)
        else:
            logger.warning(f"Unknown export format: {export_format}, defaulting to text")
            await self._write_text_output(base_url, self.timestamp, llms_entries, pages, metadata, include_full_text, full_text_only)
    
    async def _write_text_output(self, base_url: str, timestamp: str, 
                               llms_entries: List[Dict], pages: List[Dict], 
                               metadata: Dict, include_full_text: bool, full_text_only: bool = False):
        """Write text format output files"""
        domain = self.extract_domain_from_url(base_url)
        
        # Create llms.txt (unless full_text_only is True)
        if not full_text_only and llms_entries:
            llms_filename = os.path.join('output', f"{domain}-llms.txt")
            
            content = f"# {domain.upper()} - LLMs.txt\n"
            content += f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"# Total pages: {len(llms_entries)}\n\n"
            
            for entry in llms_entries:
                content += f"## {entry['title']}\n"
                content += f"URL: {entry['url']}\n"
                content += f"Description: {entry['description']}\n\n"
            
            write_safe_file(llms_filename, content)
            logger.info(f"📝 Created {llms_filename}")
        
        # Create llms-full.txt
        if include_full_text and pages:
            full_filename = os.path.join('output', f"{domain}-llms-full.txt")
            
            content = f"# {domain.upper()} - Full Content\n"
            content += f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"# Total pages: {len(pages)}\n\n"
            content += "="*80 + "\n\n"
            
            for i, page in enumerate(pages, 1):
                content += f"--- Page {i} ---\n"
                content += f"Title: {page.get('title', 'Untitled')}\n"
                content += f"URL: {page.get('url', '')}\n"
                content += f"Content:\n{page.get('content', '')}\n\n"
                content += "="*80 + "\n\n"
            
            write_safe_file(full_filename, content)
            logger.info(f"📄 Created {full_filename}")
    
    async def _write_json_output(self, domain: str, timestamp: str, 
                               llms_entries: List[Dict], pages: List[Dict], 
                               metadata: Dict, include_full_text: bool, full_text_only: bool = False):
        """Write JSON format output"""
        output_data = {
            'metadata': metadata,
            'llms_entries': llms_entries if not full_text_only else [],
            'full_content': pages if include_full_text else []
        }
        
        filename = os.path.join('output', f"{domain}-llms-{timestamp}.json")
        write_safe_file(filename, json.dumps(output_data, indent=2, ensure_ascii=False))
        logger.info(f"📄 Created {filename}")
    
    async def _write_yaml_output(self, domain: str, timestamp: str, 
                               llms_entries: List[Dict], pages: List[Dict], 
                               metadata: Dict, include_full_text: bool, full_text_only: bool = False):
        """Write YAML format output"""
        output_data = {
            'metadata': metadata,
            'llms_entries': llms_entries if not full_text_only else [],
            'full_content': pages if include_full_text else []
        }
        
        filename = os.path.join('output', f"{domain}-llms-{timestamp}.yaml")
        write_safe_file(filename, yaml.dump(output_data, default_flow_style=False, allow_unicode=True))
        logger.info(f"📄 Created {filename}")
    
    def generate_description(self, title: str, content: str, url: str = "") -> str:
        """Public method to generate description for a single page"""
        return self.generate_description_with_fallbacks(title, content, url)