#!/usr/bin/env python3
"""
Enhanced web crawler using Crawl4AI v0.6.0+ features for the LLMs.txt Generator package.

This module contains the WebCrawler class extracted and adapted from the original
generate-llmstxt-crawl4ai.py file to work within the llmsgen package structure.
"""

import os
import json
import time
import logging
import re
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class WebCrawler:
    """Enhanced web crawler using Crawl4AI v0.6.0+ features"""
    
    def __init__(self):
        self.session_data = {}
        self.session_id = f"llmstxt_session_{int(time.time())}"
        
        # Set up local Playwright browsers path automatically
        self._setup_local_playwright()
    
    def _setup_local_playwright(self):
        """Set up local Playwright browsers path automatically"""
        try:
            # Import configuration
            from config import USE_LOCAL_PLAYWRIGHT, LOCAL_PLAYWRIGHT_BROWSERS
            
            if not USE_LOCAL_PLAYWRIGHT:
                logger.debug("Local Playwright disabled in configuration")
                return
                
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels to get to the project root, then into llmsgen/playwright
            project_root = os.path.dirname(os.path.dirname(current_dir))
            
            if LOCAL_PLAYWRIGHT_BROWSERS == 'auto':
                # Auto-detect local browsers path
                local_browsers_path = os.path.join(project_root, 'llmsgen', 'playwright', '.local-browsers')
            else:
                # Use specified path
                local_browsers_path = LOCAL_PLAYWRIGHT_BROWSERS
            
            # Only set if the local browsers directory exists
            if os.path.exists(local_browsers_path):
                os.environ['PLAYWRIGHT_BROWSERS_PATH'] = local_browsers_path
                logger.info(f"🎭 Using local Playwright browsers: {local_browsers_path}")
            else:
                logger.debug(f"Local Playwright browsers not found at: {local_browsers_path}")
                logger.debug("Will use system Playwright installation")
                
        except Exception as e:
            logger.debug(f"Could not set up local Playwright: {e}")
            logger.debug("Will use system Playwright installation")
        
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
                    
                    # Extract all internal links from main page using enhanced discovery
                    discovered_main_links = self._extract_all_links(main_result, base_url, base_domain)
                    all_discovered_links.update(discovered_main_links)
                    
                    logger.info(f"🔗 Discovered {len(all_discovered_links)} unique internal links from main page")
                else:
                    logger.error(f"❌ Failed to crawl main page: {main_result.error_message if main_result else 'No result'}")
                    return []
            
            return crawled_pages
        
        except ImportError as e:
            logger.error(f"❌ Crawl4AI import error: {e}")
            logger.error("💡 Please install with: pip install 'crawl4ai[all]>=0.6.0'")
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
        
        # High value pages
        high_value_keywords = [
            'documentation', 'docs', 'guide', 'tutorial', 'api', 'reference',
            'blog', 'article', 'news', 'feature', 'product', 'service',
            'about', 'contact', 'help', 'support', 'changelog', 'release',
            'tools', 'resources', 'pricing', 'plans', 'download', 'install'
        ]
        
        for keyword in high_value_keywords:
            if keyword in url_lower:
                score += 2.0
                break
        
        # Medium value indicators
        medium_value_keywords = ['learn', 'faq', 'getting-started', 'overview', 'intro']
        for keyword in medium_value_keywords:
            if keyword in url_lower:
                score += 1.0
                break
        
        # Penalize deep paths (likely specific/less important)
        path_depth = url.count('/') - 2  # Subtract protocol and domain
        if path_depth > 3:
            score -= (path_depth - 3) * 0.5
        
        # Penalize query strings (often dynamic/less important)
        if '?' in url and len(url.split('?')[1]) > 10:
            score -= 2.0
        
        return max(0.1, score)  # Minimum score of 0.1

    async def crawl_website(self, base_url: str, max_pages: int = 50, comprehensive: bool = False) -> List[Dict[str, Any]]:
        """
        Crawl website using Crawl4AI's BestFirstCrawlingStrategy following official best practices.
        
        Args:
            base_url: Starting URL to crawl
            max_pages: Maximum number of pages to crawl
            comprehensive: If True, does deep recursive crawling of entire domain
        
        Implements recommendations from: https://docs.crawl4ai.com/core/deep-crawling/
        """
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
            from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
            from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
            from crawl4ai.deep_crawling.filters import (
                FilterChain,
                DomainFilter,
                ContentTypeFilter
            )
            from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
            import urllib.parse

            base_domain = urllib.parse.urlparse(base_url).netloc
            pages = []

            # 1. Create an enhanced relevance scorer with domain-specific keywords
            # Extract base name from URL for more targeted scoring
            parsed_url = urllib.parse.urlparse(base_url)
            domain_name = parsed_url.netloc.replace('www.', '').split('.')[0]
            
            # Enhanced keyword list based on common page types and domain
            enhanced_keywords = [
                "documentation", "guide", "tutorial", "api", "reference", 
                "blog", "article", "news", "feature", "product", "service",
                "about", "contact", "help", "support", "changelog", "release",
                "tools", "resources", "pricing", "plans", "download", "install",
                "agents", "knowledge", "hub", "page", "list", "directory",
                "profile", "details", "search", "browse", "category", "tag",
                domain_name  # Add domain name as a relevant keyword
            ]
            
            keyword_scorer = KeywordRelevanceScorer(
                keywords=enhanced_keywords,
                weight=1.0  # Increase weight for better scoring
            )

            # 2. Create sophisticated filter chain following best practices
            filter_chain = None
            try:
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

            # 3. Configure BestFirstCrawlingStrategy with comprehensive mode support
            if comprehensive:
                # Comprehensive mode: Maximum depth and pages for complete domain coverage
                crawl_depth = 10  # Very deep crawling for comprehensive mode
                effective_max_pages = max_pages * 10  # Allow many more pages to be discovered
                logger.info(f"🌊 Comprehensive crawling mode: depth={crawl_depth}, max_pages={effective_max_pages}")
            else:
                # Normal mode: More aggressive settings for better coverage
                crawl_depth = 6 if max_pages > 100 else 5 if max_pages > 50 else 4
                effective_max_pages = max_pages * 2  # Allow discovering more pages than requested
                logger.info(f"📄 Normal crawling mode: depth={crawl_depth}, max_pages={effective_max_pages}")
            
            deep_crawl_strategy = BestFirstCrawlingStrategy(
                max_depth=crawl_depth,
                max_pages=effective_max_pages,
                include_external=False,  # Stay within domain
                url_scorer=keyword_scorer,
                filter_chain=filter_chain
            )

            # 4. Set up crawler configuration with only supported parameters
            crawler_config = CrawlerRunConfig(
                deep_crawl_strategy=deep_crawl_strategy,
                scraping_strategy=LXMLWebScrapingStrategy(),
                word_count_threshold=50,  # Higher threshold for quality content
                page_timeout=30000,  # 30 seconds timeout
                verbose=True,
                cache_mode=CacheMode.ENABLED  # Use cache for efficiency
            )

            # 5. Execute the crawl with proper analytics
            async with AsyncWebCrawler(verbose=True) as crawler:
                logger.info(f"🚀 Starting BestFirst deep crawl on {base_url}")
                logger.info(f"📊 Configuration: max_depth={crawl_depth}, max_pages={max_pages}")
                logger.info(f"🔧 Filters active: {'Yes' if filter_chain else 'Basic only'}")
                
                # Track analytics as recommended in docs
                depth_counts = {}
                total_score = 0
                processed_count = 0
                
                crawl_results = await crawler.arun(base_url, config=crawler_config)
                if isinstance(crawl_results, list):
                    results_to_process = crawl_results
                else:
                    results_to_process = [crawl_results]
                
                for result in results_to_process:
                    if result and result.success:
                        # Extract content using best available source
                        content = ""
                        if hasattr(result, 'markdown') and result.markdown:
                            content = result.markdown.fit_markdown or result.markdown.raw_markdown
                        elif hasattr(result, 'cleaned_html') and result.cleaned_html:
                            content = result.cleaned_html
                        
                        word_count = len(content.split()) if content else 0
                        
                        if word_count > crawler_config.word_count_threshold:
                            score = result.metadata.get('score', 0) if result.metadata else 0
                            depth = result.metadata.get('depth', 0) if result.metadata else 0
                            
                            pages.append({
                                'url': result.url,
                                'title': self._extract_title_v6(result),
                                'content': content,
                                'word_count': word_count,
                                'score': score,
                                'depth': depth,
                                'session_id': self.session_id,
                                'metadata': result.metadata or {},
                                'crawl_timestamp': datetime.now().isoformat()
                            })
                            
                            # Analytics tracking (best practice from docs)
                            depth_counts[depth] = depth_counts.get(depth, 0) + 1
                            total_score += score
                            processed_count += 1
                            
                            logger.info(f"✅ [{len(pages)}/{effective_max_pages}] Depth:{depth} | Score:{score:.2f} | {word_count} words | {result.url}")
                            
                            # Enhanced pagination discovery for better coverage
                            pagination_urls = self._discover_pagination_links(result, base_url, base_domain)
                            if pagination_urls:
                                logger.info(f"🔍 Found {len(pagination_urls)} additional pagination URLs to explore")
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
        if hasattr(result, 'markdown') and result.markdown and result.markdown.raw_markdown:
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

    def _discover_pagination_links(self, result, base_url: str, base_domain: str) -> set:
        """
        Dynamically discover pagination patterns without hardcoding.
        
        Detects common pagination patterns:
        - Next/Previous buttons
        - Numbered page links  
        - URL parameters (?page=N)
        - Load more buttons
        - Sequential pagination
        """
        discovered_urls = set()
        
        if not hasattr(result, 'html') or not result.html:
            return discovered_urls
            
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # Pattern 1: Next/Previous button detection
            next_patterns = [
                # Common next button selectors
                'a[rel="next"]',
                'a.next', 'a.next-page', 'a.page-next',
                'a[aria-label*="next" i]', 'a[title*="next" i]',
                'a:-soup-contains("Next")', 'a:-soup-contains(">")', 'a:-soup-contains("→")',
                '.pagination a:-soup-contains("Next")',
                '.pager a:-soup-contains("Next")',
                'a[href*="page="]:-soup-contains("Next")',
            ]
            
            for pattern in next_patterns:
                try:
                    next_links = soup.select(pattern)
                    for link in next_links:
                        href = link.get('href')
                        if href and href != '#' and not href.startswith('javascript:'):
                            full_url = self._make_absolute_url(href, base_url)
                            if full_url and base_domain in full_url:
                                discovered_urls.add(full_url)
                except:
                    continue
            
            # Pattern 2: Numbered pagination detection
            pagination_selectors = [
                '.pagination a[href*="page="]',
                '.pager a[href*="page="]', 
                '.page-numbers a',
                'a[href*="page/"]:not([href$="/page/"])',
                'a[href*="p="]:not([href$="p="])',
                'nav a[href*="page"]',
                '.pagination-list a',
            ]
            
            for selector in pagination_selectors:
                try:
                    page_links = soup.select(selector)
                    for link in page_links:
                        href = link.get('href')
                        if href and href != '#':
                            full_url = self._make_absolute_url(href, base_url)
                            if full_url and base_domain in full_url:
                                discovered_urls.add(full_url)
                except:
                    continue
            
            # Pattern 3: Dynamic page number generation
            current_url = result.url
            discovered_urls.update(self._generate_sequential_pages(current_url, base_domain))
            
            # Pattern 4: Load more / Show more patterns
            load_more_patterns = [
                'a[href*="load"]', 'a[href*="more"]',
                'button[data-url]', 'a[data-page]',
                '.load-more[href]', '.show-more[href]'
            ]
            
            for pattern in load_more_patterns:
                try:
                    elements = soup.select(pattern)
                    for element in elements:
                        href = element.get('href') or element.get('data-url')
                        if href and href != '#':
                            full_url = self._make_absolute_url(href, base_url)
                            if full_url and base_domain in full_url:
                                discovered_urls.add(full_url)
                except:
                    continue
            
            # Pattern 5: Form-based pagination
            forms = soup.find_all('form')
            for form in forms:
                action = form.get('action', '')
                if 'page' in action.lower() or 'search' in action.lower():
                    try:
                        full_url = self._make_absolute_url(action, base_url)
                        if full_url and base_domain in full_url:
                            discovered_urls.add(full_url)
                    except:
                        continue
            
            if discovered_urls:
                logger.debug(f"🔍 Discovered {len(discovered_urls)} pagination URLs from {result.url}")
                
        except Exception as e:
            logger.debug(f"Error in pagination discovery: {e}")
        
        return discovered_urls
    
    def _make_absolute_url(self, href: str, base_url: str) -> str:
        """Convert relative URLs to absolute URLs"""
        try:
            import urllib.parse
            if href.startswith('http'):
                return href
            return urllib.parse.urljoin(base_url, href)
        except:
            return ""
    
    def _generate_sequential_pages(self, current_url: str, base_domain: str, max_pages: int = 20) -> set:
        """
        Generate sequential page URLs based on current URL pattern.
        
        Detects patterns like:
        - /page/2/ -> /page/3/, /page/4/
        - ?page=2 -> ?page=3, ?page=4  
        - /p2/ -> /p3/, /p4/
        """
        generated_urls = set()
        
        try:
            import re
            import urllib.parse
            
            # Pattern detection for common pagination formats
            patterns = [
                (r'/page/(\d+)/', lambda m, n: current_url.replace(f'/page/{m.group(1)}/', f'/page/{n}/')),
                (r'[?&]page=(\d+)', lambda m, n: re.sub(r'([?&])page=\d+', f'\\1page={n}', current_url)),
                (r'[?&]p=(\d+)', lambda m, n: re.sub(r'([?&])p=\d+', f'\\1p={n}', current_url)),
                (r'/p(\d+)/', lambda m, n: current_url.replace(f'/p{m.group(1)}/', f'/p{n}/')),
                (r'[?&]offset=(\d+)', lambda m, n: re.sub(r'([?&])offset=\d+', f'\\1offset={n*10}', current_url)),
            ]
            
            for pattern, url_generator in patterns:
                match = re.search(pattern, current_url)
                if match:
                    try:
                        current_page = int(match.group(1))
                        
                        # Generate next pages (limited to prevent infinite crawling)
                        for next_page in range(current_page + 1, min(current_page + max_pages, current_page + 10)):
                            try:
                                new_url = url_generator(match, next_page)
                                if new_url and base_domain in new_url and new_url != current_url:
                                    generated_urls.add(new_url)
                            except:
                                continue
                        
                        # Also try previous pages if we're not on page 1
                        if current_page > 1:
                            for prev_page in range(max(1, current_page - 5), current_page):
                                try:
                                    new_url = url_generator(match, prev_page)
                                    if new_url and base_domain in new_url and new_url != current_url:
                                        generated_urls.add(new_url)
                                except:
                                    continue
                                    
                    except ValueError:
                        continue
                    break  # Found pattern, no need to check others
            
            # If no pattern found, try adding page parameters to base URL
            if not generated_urls and '?' not in current_url:
                base_url = current_url.rstrip('/')
                for page_num in range(2, 6):  # Try pages 2-5
                    for param in ['page', 'p']:
                        test_url = f"{base_url}?{param}={page_num}"
                        if base_domain in test_url:
                            generated_urls.add(test_url)
                            
        except Exception as e:
            logger.debug(f"Error generating sequential pages: {e}")
        
        return generated_urls
    
    def _get_domain_variants(self, domain: str) -> set:
        """
        Get all valid domain variants for matching (www vs non-www, etc.)
        Production-level domain matching to handle all edge cases.
        """
        variants = {domain.lower()}
        
        # Handle www variants
        if domain.startswith('www.'):
            variants.add(domain[4:])  # Remove www
        else:
            variants.add(f'www.{domain}')  # Add www
        
        # Handle common subdomain patterns
        if '.' in domain:
            parts = domain.split('.')
            if len(parts) >= 2:
                # Main domain without any subdomain
                main_domain = '.'.join(parts[-2:])
                variants.add(main_domain)
                variants.add(f'www.{main_domain}')
        
        return variants
    
    def _is_valid_domain_url(self, url: str, valid_domains: set) -> bool:
        """
        Check if URL belongs to one of the valid domains.
        Production-level URL validation with comprehensive checks.
        """
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            url_domain = parsed.netloc.lower()
            
            # Direct domain match
            if url_domain in valid_domains:
                return True
            
            # Check if any valid domain is contained in the URL domain
            for valid_domain in valid_domains:
                if url_domain == valid_domain or url_domain.endswith(f'.{valid_domain}'):
                    return True
            
            return False
        except Exception:
            return False 

    async def crawl_from_sitemap(self, base_url: str, sitemap_url: str = None, max_pages: int = 50) -> List[Dict[str, Any]]:
        """
        Crawl website using sitemap.xml for URL discovery.
        
        Args:
            base_url: Base website URL
            sitemap_url: Direct sitemap URL or 'auto' for auto-discovery
            max_pages: Maximum number of pages to crawl
        """
        try:
            import urllib.parse
            
            # Get primary domain and all valid domain variants
            base_domain = urllib.parse.urlparse(base_url).netloc
            valid_domains = self._get_domain_variants(base_domain)
            
            # Get sitemap URLs
            if sitemap_url == "auto" or sitemap_url is None:
                sitemap_urls = await self._auto_discover_sitemaps(base_url)
                if not sitemap_urls:
                    logger.error("❌ No sitemaps found via auto-discovery")
                    return []
                logger.info(f"🗺️ Auto-discovered {len(sitemap_urls)} sitemap(s)")
            else:
                sitemap_urls = [sitemap_url]
                logger.info(f"🗺️ Using provided sitemap: {sitemap_url}")
            
            # Extract URLs from sitemaps
            all_urls = set()
            for sitemap in sitemap_urls:
                urls = await self._parse_sitemap(sitemap, valid_domains)
                all_urls.update(urls)
                logger.info(f"📄 Found {len(urls)} URLs in {sitemap}")
            
            if not all_urls:
                logger.error("❌ No URLs found in sitemaps")
                return []
            
            logger.info(f"🎯 Total unique URLs discovered: {len(all_urls)}")
            
            # Limit URLs if necessary
            if len(all_urls) > max_pages:
                logger.info(f"⚠️ Limiting to {max_pages} pages (found {len(all_urls)})")
                # Prioritize important URLs (shorter paths, main sections)
                sorted_urls = sorted(all_urls, key=lambda url: (url.count('/'), len(url)))
                all_urls = set(sorted_urls[:max_pages])
            
            # Crawl the discovered URLs
            return await self._crawl_url_list(list(all_urls), base_url, valid_domains)
            
        except Exception as e:
            logger.error(f"❌ Sitemap crawling failed: {e}", exc_info=True)
            return []
    
    async def _auto_discover_sitemaps(self, base_url: str) -> List[str]:
        """
        Auto-discover sitemap URLs from common locations.
        Production-level with comprehensive discovery and error handling.
        """
        import urllib.parse
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Get domain variants for flexible matching
        base_domain = urllib.parse.urlparse(base_url).netloc
        valid_domains = self._get_domain_variants(base_domain)
        sitemaps = []
        
        # Create robust HTTP session with retries
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Common sitemap locations (extended list)
        common_locations = [
            '/sitemap.xml',
            '/sitemap_index.xml', 
            '/sitemaps.xml',
            '/sitemap/sitemap.xml',
            '/sitemaps/sitemap.xml',
            '/xmlsitemap.xml',
            '/sitemap/index.xml',
            '/wp-sitemap.xml',  # WordPress
            '/sitemap-index.xml',
            '/robots_sitemap.xml'
        ]
        
        # Try common locations with enhanced error handling
        for location in common_locations:
            test_url = f"{base_url.rstrip('/')}{location}"
            try:
                response = session.head(test_url, timeout=15, allow_redirects=True)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if ('xml' in content_type or 
                        test_url.endswith('.xml') or 
                        'sitemap' in content_type):
                        sitemaps.append(test_url)
                        logger.debug(f"✅ Found sitemap at: {test_url}")
            except Exception as e:
                logger.debug(f"❌ Failed to check {test_url}: {e}")
                continue
        
        # Check robots.txt for sitemap declarations with enhanced parsing
        try:
            robots_url = f"{base_url.rstrip('/')}/robots.txt"
            response = session.get(robots_url, timeout=15)
            if response.status_code == 200:
                for line in response.text.split('\n'):
                    line = line.strip()
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        # Use flexible domain matching
                        if self._is_valid_domain_url(sitemap_url, valid_domains):
                            sitemaps.append(sitemap_url)
                            logger.debug(f"✅ Found sitemap in robots.txt: {sitemap_url}")
        except Exception as e:
            logger.debug(f"❌ Failed to check robots.txt: {e}")
        
        # Clean up session
        session.close()
        
        # Remove duplicates and validate URLs
        unique_sitemaps = []
        seen = set()
        for sitemap in sitemaps:
            if sitemap not in seen and sitemap.startswith(('http://', 'https://')):
                unique_sitemaps.append(sitemap)
                seen.add(sitemap)
        
        logger.info(f"🔍 Discovered {len(unique_sitemaps)} sitemap(s) from {len(common_locations)} locations + robots.txt")
        return unique_sitemaps
    
    async def _parse_sitemap(self, sitemap_url: str, valid_domains: set) -> set:
        """
        Parse sitemap XML and extract URLs.
        Production-level with memory management, security checks, and comprehensive error handling.
        """
        urls = set()
        
        try:
            import requests
            import xml.etree.ElementTree as ET
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            import gzip
            import io
            
            logger.debug(f"📄 Parsing sitemap: {sitemap_url}")
            
            # Create robust HTTP session
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Security: Set reasonable size limits
            MAX_SITEMAP_SIZE = 50 * 1024 * 1024  # 50MB limit
            
            response = session.get(
                sitemap_url, 
                timeout=30,
                stream=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; LLMsTxtGenerator/1.0)',
                    'Accept': 'application/xml,text/xml,*/*',
                    'Accept-Encoding': 'gzip, deflate'
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"⚠️ Failed to fetch sitemap: {sitemap_url} (status: {response.status_code})")
                session.close()
                return urls
            
            # Check content size for security
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_SITEMAP_SIZE:
                logger.warning(f"⚠️ Sitemap too large: {sitemap_url} ({content_length} bytes)")
                session.close()
                return urls
            
            # Read content with size limit
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > MAX_SITEMAP_SIZE:
                    logger.warning(f"⚠️ Sitemap exceeded size limit: {sitemap_url}")
                    session.close()
                    return urls
            
            session.close()
            
            # Handle gzipped content only if actually gzipped
            content_encoding = response.headers.get('content-encoding', '').lower()
            is_gzipped = (sitemap_url.endswith('.gz') or 
                         'gzip' in content_encoding or
                         content.startswith(b'\x1f\x8b'))  # gzip magic bytes
            
            if is_gzipped:
                try:
                    content = gzip.decompress(content)
                    logger.debug("📦 Decompressed gzipped sitemap")
                except Exception as e:
                    logger.debug(f"⚠️ Content not actually gzipped, using as-is: {e}")
                    # Content is not actually gzipped, continue with original content
            
            # Parse XML with security measures
            try:
                # Safely parse XML content
                root = ET.fromstring(content)
            except ET.ParseError as e:
                logger.warning(f"⚠️ Invalid XML in sitemap {sitemap_url}: {e}")
                return urls
            except Exception as e:
                logger.warning(f"⚠️ XML parsing error in sitemap {sitemap_url}: {e}")
                return urls
            
            # Register namespaces to handle different sitemap formats
            namespaces = {
                'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                'news': 'http://www.google.com/schemas/sitemap-news/0.9',
                'image': 'http://www.google.com/schemas/sitemap-image/1.1'
            }
            
            # Register the namespaces with ElementTree
            for prefix, uri in namespaces.items():
                ET.register_namespace(prefix, uri)
            
            # Check if this is a sitemap index (contains other sitemaps)
            # Try both with and without namespace prefix
            sitemap_elements = (root.findall('.//sitemap:sitemap', namespaces) or 
                               root.findall('.//sitemap') or 
                               root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'))
            
            if sitemap_elements:
                logger.debug(f"📋 Found sitemap index with {len(sitemap_elements)} sub-sitemaps")
                for sitemap_elem in sitemap_elements:
                    # Try multiple ways to find the loc element
                    loc_elem = (sitemap_elem.find('sitemap:loc', namespaces) or 
                               sitemap_elem.find('loc') or 
                               sitemap_elem.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'))
                    
                    if loc_elem is not None and loc_elem.text:
                        sub_sitemap_url = loc_elem.text.strip()
                        if self._is_valid_domain_url(sub_sitemap_url, valid_domains):
                            # Recursively parse sub-sitemaps
                            sub_urls = await self._parse_sitemap(sub_sitemap_url, valid_domains)
                            urls.update(sub_urls)
            
            # Extract regular URLs - try multiple approaches
            url_elements = (root.findall('.//sitemap:url', namespaces) or 
                           root.findall('.//url') or 
                           root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'))
            
            if url_elements:
                logger.debug(f"🔗 Found {len(url_elements)} URL elements in sitemap")
                for url_elem in url_elements:
                    # Try multiple ways to find the loc element
                    loc_elem = (url_elem.find('sitemap:loc', namespaces) or 
                               url_elem.find('loc') or 
                               url_elem.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'))
                    
                    if loc_elem is not None and loc_elem.text:
                        url = loc_elem.text.strip()
                        if self._is_valid_domain_url(url, valid_domains):
                            urls.add(url)
                            logger.debug(f"  ✅ Added URL: {url}")
            
            # Enhanced fallback: try to extract URLs without namespace 
            if not urls:
                logger.debug("🔍 Trying fallback URL extraction...")
                for elem in root.iter():
                    if elem.tag.endswith('loc') and elem.text:
                        url = elem.text.strip()
                        if self._is_valid_domain_url(url, valid_domains) and url.startswith('http'):
                            urls.add(url)
                            logger.debug(f"  🔄 Fallback added URL: {url}")
            
            logger.info(f"📊 Extracted {len(urls)} URLs from sitemap: {sitemap_url}")
            
            # Debug information
            if len(urls) == 0:
                logger.debug(f"🔍 Debug: No URLs found. XML structure analysis:")
                logger.debug(f"  Root tag: {root.tag}")
                logger.debug(f"  Root attributes: {root.attrib}")
                logger.debug(f"  Number of child elements: {len(list(root))}")
                logger.debug(f"  All element tags found: {set(elem.tag for elem in root.iter())}")
                
                # Try alternative parsing approach
                logger.debug("🔍 Trying alternative parsing without namespaces...")
                for elem in root.iter():
                    if 'loc' in elem.tag.lower() and elem.text:
                        url = elem.text.strip()
                        if self._is_valid_domain_url(url, valid_domains):
                            urls.add(url)
                            logger.debug(f"  🔄 Alternative method found URL: {url}")
                
                if len(urls) > 0:
                    logger.info(f"📊 Alternative parsing found {len(urls)} URLs")
            
        except Exception as e:
            logger.warning(f"⚠️ Error parsing sitemap {sitemap_url}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        return urls
    
    async def _crawl_url_list(self, urls: List[str], base_url: str, valid_domains: set) -> List[Dict[str, Any]]:
        """
        Crawl a specific list of URLs efficiently.
        Production-level with memory management, progress tracking, and error recovery.
        """
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
            from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
            import asyncio
            import gc
            
            pages = []
            total_urls = len(urls)
            processed = 0
            errors = 0
            
            logger.info(f"🚀 Starting sitemap-based crawling of {total_urls} URLs")
            
            # Memory management: Process in smaller batches for large sitemaps
            BATCH_SIZE = min(100, max(10, total_urls // 10))  # Adaptive batch size
            
            # Configure for efficient batch crawling
            crawler_config = CrawlerRunConfig(
                scraping_strategy=LXMLWebScrapingStrategy(),
                word_count_threshold=50,  # Skip very short pages
                page_timeout=15000,  # Shorter timeout for batch processing
                cache_mode=CacheMode.BYPASS,
                verbose=False,  # Reduce verbosity for large batches
                only_text=True,  # Reduce memory usage
                exclude_external_links=True,  # Focus on content
                exclude_social_media_links=True  # Reduce noise
            )
            
            # Process URLs in batches with progress tracking
            async with AsyncWebCrawler(verbose=False) as crawler:
                for batch_start in range(0, total_urls, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, total_urls)
                    batch_urls = urls[batch_start:batch_end]
                    
                    logger.info(f"📦 Processing batch {batch_start//BATCH_SIZE + 1}/{(total_urls-1)//BATCH_SIZE + 1} ({len(batch_urls)} URLs)")
                    
                    # Process batch
                    for i, url in enumerate(batch_urls, batch_start + 1):
                        try:
                            if i % 50 == 0:  # Progress update every 50 URLs
                                logger.info(f"📊 Progress: {i}/{total_urls} URLs processed ({i/total_urls*100:.1f}%)")
                            
                            logger.debug(f"📄 Crawling [{i}/{total_urls}]: {url}")
                            result = await crawler.arun(url, config=crawler_config)
                            
                            if result and result.success:
                                # Extract content
                                content = ""
                                if hasattr(result, 'markdown') and result.markdown:
                                    content = result.markdown.fit_markdown or result.markdown.raw_markdown
                                elif hasattr(result, 'cleaned_html') and result.cleaned_html:
                                    content = result.cleaned_html
                                
                                word_count = len(content.split()) if content else 0
                                
                                if word_count >= crawler_config.word_count_threshold:
                                    pages.append({
                                        'url': result.url,
                                        'title': self._extract_title_v6(result),
                                        'content': content,
                                        'word_count': word_count,
                                        'score': 1.0,  # All sitemap URLs are considered equally important
                                        'depth': 0,  # From sitemap, so depth is not relevant
                                        'session_id': self.session_id,
                                        'metadata': result.metadata or {},
                                        'crawl_timestamp': datetime.now().isoformat(),
                                        'source': 'sitemap'
                                    })
                                    
                                    processed += 1
                                    if i % 10 == 0:  # Progress update every 10 pages
                                        logger.info(f"✅ Progress: {processed} pages processed from {i} URLs crawled")
                                else:
                                    logger.debug(f"⏭️ Skipping low-content page ({word_count} words): {url}")
                            else:
                                errors += 1
                                logger.debug(f"❌ Failed to crawl: {url}")
                                
                        except Exception as e:
                            errors += 1
                            logger.debug(f"❌ Error crawling {url}: {e}")
                            continue
                    
                    # Memory cleanup after each batch
                    if batch_start > 0 and batch_start % (BATCH_SIZE * 5) == 0:
                        gc.collect()
                        logger.debug(f"🧹 Memory cleanup after processing {batch_end} URLs")
                    
                    # Small delay between batches to prevent overwhelming the server
                    if batch_end < total_urls:
                        await asyncio.sleep(1)
            
            success_rate = (processed / total_urls * 100) if total_urls > 0 else 0
            logger.info(f"✅ Sitemap crawling completed: {processed}/{total_urls} pages successfully crawled ({success_rate:.1f}% success rate)")
            
            if errors > 0:
                logger.warning(f"⚠️ {errors} URLs failed to crawl")
            
            return pages
            
        except ImportError as e:
            logger.error(f"❌ Crawl4AI import error: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ URL list crawling failed: {e}", exc_info=True)
            return [] 