#!/usr/bin/env python3
"""
Main script to generate llms.txt files using the modular llmsgen package.

This script provides a command-line interface for generating llms.txt and llms-full.txt
files from websites using AI-powered content analysis.
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path

# Add the parent directory to the path so we can import the llmsgen package
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmsgen.models.client import ModelManager, AIClient
from llmsgen.generator.llms_generator import LLMsTxtGenerator
from llmsgen.utils.file_utils import ensure_output_dir, create_sample_env_file

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging with enhanced formatting
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('llmstxt_generator.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Fix Windows UTF-8 encoding for emojis
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except (TypeError, AttributeError):
            logger.info("Could not reconfigure stdout for UTF-8. Emojis may not render correctly.")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate llms.txt files from websites using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_llms.py                              # Interactive mode - prompts for URL
  python scripts/generate_llms.py https://example.com          # Direct URL mode
  python scripts/generate_llms.py https://docs.python.org --max-pages 100 --format json
  python scripts/generate_llms.py https://blog.example.com --workers 10 --batch-size 20
  python scripts/generate_llms.py https://example.com --full-text-only
        """
    )
    
    # URL argument - make it optional for interactive mode
    parser.add_argument('url', nargs='?', help='Website URL to crawl and generate llms.txt for (will prompt if not provided)')
    
    # Optional arguments
    parser.add_argument('--max-pages', type=int, default=50,
                       help='Maximum number of pages to crawl (default: 50)')
    parser.add_argument('--format', choices=['text', 'json', 'yaml'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--workers', type=int, default=5,
                       help='Number of parallel workers for processing (default: 5)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing pages (default: 10)')
    parser.add_argument('--no-full-text', action='store_true',
                       help='Skip generating full-text file')
    parser.add_argument('--full-text-only', action='store_true',
                       help='Generate only full-text file, skip llms.txt')
    parser.add_argument('--max-full-text-pages', type=int,
                       help='Limit full-text to specified number of pages')
    parser.add_argument('--crawl-strategy', choices=['systematic', 'comprehensive'], 
                       default='systematic',
                       help='Crawling strategy (default: systematic)')
    parser.add_argument('--safety-limit', type=int,
                       help='Safety limit for crawling (overrides max-pages)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--setup-env', action='store_true',
                       help='Create a sample .env file and exit')
    
    return parser.parse_args()

async def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle --setup-env flag
    if args.setup_env:
        create_sample_env_file()
        print("✅ Sample .env file created. Please edit it with your API keys.")
        return
    
    # Ensure output directory exists
    ensure_output_dir()
    
    print("🚀 LLMs.txt Generator")
    print("=" * 50)
    
    try:
        # Initialize components
        logger.info("🔧 Initializing components...")
        model_manager = ModelManager()
        generator = LLMsTxtGenerator()
        
        # Interactive URL input if not provided
        if not args.url:
            print("\n🌐 Interactive Mode - Website URL Input")
            print("=" * 50)
            while True:
                url_input = input("Enter the website URL to crawl: ").strip()
                if url_input:
                    args.url = url_input
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
        
        # Validate URL
        if not args.url.startswith(('http://', 'https://')):
            args.url = 'https://' + args.url
        
        # Adjust max_pages for comprehensive modes
        effective_max_pages = args.max_pages
        if crawling_mode in ["comprehensive", "sitemap"]:
            effective_max_pages = 999999  # Effectively unlimited for comprehensive crawling
            max_pages_display = "Unlimited (discovers all pages)"
        else:
            max_pages_display = str(effective_max_pages)
        
        print(f"\n🌐 Target URL: {args.url}")
        print(f"📊 Max Pages: {max_pages_display}")
        print(f"🔧 Workers: {args.workers}")
        print(f"📦 Batch Size: {args.batch_size}")
        print(f"📄 Format: {args.format}")
        print(f"🤖 Model: {selected_model.display_name}")
        print(f"🕷️ Crawling Mode: {crawling_mode}")
        if sitemap_url and sitemap_url != "auto":
            print(f"🗺️ Sitemap URL: {sitemap_url}")
        
        # Generate llms.txt
        await generator.generate_llmstxt(
            base_url=args.url,
            max_pages=effective_max_pages,
            export_format=args.format,
            include_full_text=not args.no_full_text,
            parallel_workers=args.workers,
            batch_size=args.batch_size,
            max_full_text_pages=args.max_full_text_pages,
            full_text_only=args.full_text_only,
            crawl_strategy=crawling_mode,  # Use the selected crawling mode
            safety_limit=args.safety_limit,
            comprehensive_crawl=comprehensive_crawl,
            sitemap_url=sitemap_url
        )
        
        print("\n🎉 Generation completed successfully!")
        print("📁 Check the ./output/ directory for your files.")
        
    except KeyboardInterrupt:
        print("\n❌ Generation cancelled by user.")
        logger.info("Generation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Handle Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the main function
    asyncio.run(main()) 