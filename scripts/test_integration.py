#!/usr/bin/env python3
"""
Test script to demonstrate the integration of the llmsgen package
with the WebCrawler functionality from generate-llmstxt-crawl4ai.py
"""

import asyncio
import sys
import os

# Add the parent directory to the Python path to import llmsgen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmsgen import logger, ModelManager, AIClient, WebCrawler, LLMsTxtGenerator


async def test_web_crawler():
    """Test the WebCrawler functionality"""
    print("🧪 Testing WebCrawler Integration")
    print("=" * 50)
    
    # Initialize crawler
    crawler = WebCrawler()
    
    # Test URL (use a small, well-known site)
    test_url = "https://example.com"
    max_pages = 3
    
    try:
        print(f"🚀 Testing crawler on: {test_url}")
        print(f"📊 Max pages: {max_pages}")
        
        # Test the basic crawl_website method
        pages = await crawler.crawl_website(test_url, max_pages=max_pages)
        
        print(f"✅ Crawl completed successfully!")
        print(f"📄 Pages found: {len(pages)}")
        
        for i, page in enumerate(pages[:3], 1):  # Show first 3 pages
            print(f"\n📝 Page {i}:")
            print(f"   URL: {page['url']}")
            print(f"   Title: {page['title']}")
            print(f"   Words: {page['word_count']}")
            print(f"   Score: {page.get('score', 'N/A')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install Crawl4AI: pip install 'crawl4ai[all]>=0.6.0'")
        return False
    except Exception as e:
        print(f"❌ Error during crawling: {e}")
        return False


def test_package_imports():
    """Test that all package components can be imported correctly"""
    print("🧪 Testing Package Imports")
    print("=" * 50)
    
    try:
        # Test main imports
        print("✅ ModelManager imported successfully")
        print("✅ AIClient imported successfully") 
        print("✅ WebCrawler imported successfully")
        print("✅ LLMsTxtGenerator imported successfully")
        print("✅ Logger imported successfully")
        
        # Test initialization
        model_manager = ModelManager()
        ai_client = AIClient(model_manager)
        crawler = WebCrawler()
        generator = LLMsTxtGenerator()
        
        print("✅ All components initialized successfully")
        
        # Test that generator has the crawler
        assert hasattr(generator, 'crawler'), "Generator should have crawler attribute"
        assert isinstance(generator.crawler, WebCrawler), "Generator crawler should be WebCrawler instance"
        
        print("✅ Generator-Crawler integration verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_model_manager():
    """Test ModelManager functionality"""
    print("🧪 Testing ModelManager")
    print("=" * 50)
    
    try:
        model_manager = ModelManager()
        
        # Test Ollama status check
        ollama_status = model_manager.check_ollama_status()
        print(f"🤖 Ollama status: {'🟢 Running' if ollama_status else '🔴 Not running'}")
        
        # Test model listing
        models = model_manager.list_models()
        print(f"📋 Available models: {len(models)}")
        
        for model_id, config in list(models.items())[:3]:  # Show first 3 models
            print(f"   • {config.display_name} ({config.provider})")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelManager test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🎯 LLMsGen Package Integration Test")
    print("=" * 80)
    print()
    
    # Run all tests
    test_results = []
    
    # Test 1: Package imports
    test_results.append(test_package_imports())
    print()
    
    # Test 2: ModelManager
    test_results.append(test_model_manager())
    print()
    
    # Test 3: WebCrawler (async)
    test_results.append(await test_web_crawler())
    print()
    
    # Summary
    print("📊 Test Results Summary")
    print("=" * 50)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed! Integration successful!")
        print("\n💡 The llmsgen package is now properly configured to use")
        print("   the WebCrawler functionality from generate-llmstxt-crawl4ai.py")
    else:
        print("❌ Some tests failed. Please check the configuration.")
    
    print()
    print("🚀 Next steps:")
    print("   1. Install dependencies: pip install 'crawl4ai[all]>=0.6.0'")
    print("   2. Set up your API keys in .env file")
    print("   3. Use: from llmsgen import LLMsTxtGenerator")


if __name__ == "__main__":
    asyncio.run(main()) 