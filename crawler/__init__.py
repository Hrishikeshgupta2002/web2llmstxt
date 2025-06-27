"""
Crawler package for LLMs.txt Generator.

This package contains web crawling functionality using Crawl4AI.
"""

try:
    from .web_crawler import WebCrawler
except ImportError:
    from crawler.web_crawler import WebCrawler

__all__ = ['WebCrawler'] 