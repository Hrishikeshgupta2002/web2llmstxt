#!/usr/bin/env python3
"""
Startup script for LLMsGen SDK.
This script works both when the package is installed and when running from source.
"""

import sys
import os
import asyncio
from pathlib import Path

def setup_imports():
    """Setup imports to work both from source and installed package"""
    # Add current directory to Python path for running from source
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

def check_dependencies():
    """Check if required dependencies are installed"""
    required_modules = [
        'requests',
        'beautifulsoup4',
        'lxml',
        'google.generativeai',
        'dotenv',
        'yaml',
        'tenacity',
        'tqdm',
        'psutil'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            if module == 'google.generativeai':
                import google.generativeai
            elif module == 'beautifulsoup4':
                import bs4
            elif module == 'dotenv':
                import dotenv
            elif module == 'yaml':
                import yaml
            else:
                __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required dependencies:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n📦 Please install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n   Or install individually:")
        for module in missing_modules:
            if module == 'google.generativeai':
                print(f"   pip install google-generativeai")
            elif module == 'beautifulsoup4':
                print(f"   pip install beautifulsoup4")
            elif module == 'dotenv':
                print(f"   pip install python-dotenv")
            elif module == 'yaml':
                print(f"   pip install pyyaml")
            else:
                print(f"   pip install {module}")
        return False
    
    return True

async def main():
    """Main function to run the LLMsGen SDK"""
    
    print("🚀 LLMsGen SDK - AI-Powered Website Content Extraction")
    print("=" * 60)
    
    # Setup imports
    setup_imports()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    try:
        # Try importing the modules - handle both installed and source scenarios
        try:
            from llmsgen.models.client import ModelManager, AIClient
            from llmsgen.generator.llms_generator import LLMsTxtGenerator
            from llmsgen.utils.file_utils import ensure_output_dir
            print("📦 Running from installed package")
        except ImportError:
            try:
                from models.client import ModelManager, AIClient
                from generator.llms_generator import LLMsTxtGenerator
                from utils.file_utils import ensure_output_dir
                print("📁 Running from source directory")
            except ImportError as e:
                print(f"❌ Could not import required modules: {e}")
                print("Make sure you're in the correct directory or the package is installed properly")
                return
        
        # Ensure output directory exists
        ensure_output_dir()
        
        # Initialize components
        print("🔧 Initializing components...")
        model_manager = ModelManager()
        generator = LLMsTxtGenerator()
        
        # Interactive URL input
        print("\n🌐 Website URL Input")
        print("=" * 30)
        while True:
            url_input = input("Enter the website URL to crawl: ").strip()
            if url_input:
                if not url_input.startswith(('http://', 'https://')):
                    url_input = 'https://' + url_input
                break
            print("❌ Please enter a valid URL.")
        
        # Interactive crawling mode selection
        crawling_mode, comprehensive_crawl, sitemap_url = generator.interactive_crawling_mode_selection()
        
        # Interactive model selection
        selected_model = generator.interactive_model_selection()
        if not selected_model:
            print("❌ No model selected. Exiting.")
            return
        
        # Initialize AI client with selected model
        ai_client = AIClient(model_manager)
        ai_client.set_model(selected_model)
        generator.set_clients(model_manager, ai_client)
        
        # Set configuration
        max_pages = 50  # Default
        workers = 5     # Default
        batch_size = 10 # Default
        
        # Ask for basic configuration
        print(f"\n⚙️ Configuration")
        print("=" * 20)
        try:
            max_pages_input = input(f"Max pages to crawl (default: {max_pages}): ").strip()
            if max_pages_input:
                max_pages = int(max_pages_input)
        except ValueError:
            print("Using default max pages")
        
        # Adjust max_pages for comprehensive modes
        effective_max_pages = max_pages
        if crawling_mode in ["comprehensive", "sitemap"]:
            effective_max_pages = 999999  # Effectively unlimited for comprehensive crawling
            max_pages_display = "Unlimited (discovers all pages)"
        else:
            max_pages_display = str(effective_max_pages)
        
        print(f"\n🌐 Target URL: {url_input}")
        print(f"📊 Max Pages: {max_pages_display}")
        print(f"🔧 Workers: {workers}")
        print(f"📦 Batch Size: {batch_size}")
        print(f"📄 Format: text")
        print(f"🤖 Model: {selected_model.display_name}")
        print(f"🕷️ Crawling Mode: {crawling_mode}")
        if sitemap_url and sitemap_url != "auto":
            print(f"🗺️ Sitemap URL: {sitemap_url}")
        
        print(f"\n🚀 Starting generation...")
        
        # Generate llms.txt
        await generator.generate_llmstxt(
            base_url=url_input,
            max_pages=effective_max_pages,
            export_format='text',
            include_full_text=True,
            parallel_workers=workers,
            batch_size=batch_size,
            crawl_strategy=crawling_mode,
            comprehensive_crawl=comprehensive_crawl,
            sitemap_url=sitemap_url
        )
        
        print("\n🎉 Generation completed successfully!")
        print("📁 Check the ./output/ directory for your files.")
        
    except KeyboardInterrupt:
        print("\n❌ Generation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        print("Please check the logs for more details.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Handle Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the main function
    asyncio.run(main()) 