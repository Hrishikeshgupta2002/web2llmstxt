#!/usr/bin/env python3
"""
Main entry point for LLMsGen SDK when run as module.
This allows running the package with: python -m llmsgen
"""

import sys
import asyncio
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

async def main():
    """Main function to run the LLMsGen SDK"""
    
    print("🚀 LLMsGen SDK - AI-Powered Website Content Extraction")
    print("=" * 60)
    
    try:
        # Import the required modules
        from models.client import ModelManager, AIClient
        from generator.llms_generator import LLMsTxtGenerator
        from utils.file_utils import ensure_output_dir
        
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
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    # Handle Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the main function
    asyncio.run(main()) 