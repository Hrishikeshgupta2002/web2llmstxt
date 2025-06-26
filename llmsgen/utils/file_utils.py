import os
import json
import yaml
import tempfile
import shutil
from typing import Dict, List, Any
from datetime import datetime

from llmsgen.config import logger
# Assuming text_utils will provide extract_domain_from_url and other text processing.
from .text_utils import extract_domain_from_url, extract_site_name_for_llmstxt, generate_site_summary_for_llmstxt, categorize_llmstxt_entries, remove_page_separators

def create_sample_env_file(file_path: str = 'env.example.llmstxt'):
    """Create a sample .env file for users to reference."""
    sample_content = """# LLMs.txt Generator Configuration
# Copy values and update as needed

# =============================================================================
# AI MODEL CONFIGURATION
# =============================================================================
# Gemini API Configuration
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Ollama Configuration (for local models)
# Default: http://localhost:11434
OLLAMA_BASE_URL=http://localhost:11434

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================
# Output directory for generated files
# Default: ./output
OUTPUT_DIR=./output

# Maximum tokens for AI generation
# Default: 1024
MAX_GEN_OUTPUT_TOKENS=1024

# Enable/disable description caching
# Default: true
CACHE_DESCRIPTIONS=true

# Default number of parallel workers for processing
# Default: 3 (reduced automatically for local models)
DEFAULT_PARALLEL_WORKERS=3
# =============================================================================
# SECURITY NOTES
# =============================================================================
# - Never commit your .env file with real API keys to version control
# - Add .env to your .gitignore file
"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        logger.info(f"Sample .env file created at {file_path}")
        return True
    except Exception as e:
        logger.warning(f"Could not create sample .env file at {file_path}: {e}")
        return False

def save_api_key_to_env(api_key: str, env_file_path: str = '.env'):
    """Safely save API key to .env file with atomic write operation."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         dir=os.path.dirname(os.path.abspath(env_file_path)) or '.',
                                         prefix='.env_tmp_') as tmp_file:
            tmp_path = tmp_file.name
            existing_lines = []
            if os.path.exists(env_file_path):
                with open(env_file_path, 'r', encoding='utf-8') as f:
                    existing_lines = f.readlines()

            gemini_key_found = False
            for line in existing_lines:
                if line.strip().startswith('GEMINI_API_KEY='):
                    tmp_file.write(f'GEMINI_API_KEY={api_key}\n')
                    gemini_key_found = True
                else:
                    tmp_file.write(line)
            if not gemini_key_found:
                tmp_file.write(f'GEMINI_API_KEY={api_key}\n')

        if os.name == 'nt' and os.path.exists(env_file_path): os.remove(env_file_path)
        shutil.move(tmp_path, env_file_path)
        logger.info(f"✅ API key saved to {env_file_path}")
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
        logger.error(f"Failed to save API key to {env_file_path}: {e}")
        raise

async def write_text_output(
    output_dir: str,
    base_url: str,
    llms_entries: List[Dict],
    pages_data: List[Dict], # Renamed from 'pages' for clarity
    metadata: Dict,
    include_full_text: bool,
    full_text_only: bool
):
    """Write traditional text format output (llms.txt and llms-full.txt)."""
    domain = extract_domain_from_url(base_url)
    os.makedirs(output_dir, exist_ok=True)

    # --- llms.txt (Index file with descriptions) ---
    if not full_text_only:
        llms_filename = os.path.join(output_dir, f'{domain}-llms.txt')
        with open(llms_filename, 'w', encoding='utf-8') as f:
            site_name = extract_site_name_for_llmstxt(base_url, pages_data)
            f.write(f"# {site_name}\n\n")

            site_summary = generate_site_summary_for_llmstxt(pages_data)
            f.write(f"> {site_summary}\n\n")

            total_pages_crawled = metadata.get('total_pages_crawled', len(pages_data))
            generated_date = metadata.get('generated_at', datetime.now().isoformat()).split('T')[0]
            f.write(f"Generated from {total_pages_crawled} pages on {generated_date} using automated crawling.\n\n")

            # Create a map of URL to page content for efficient lookup in categorization
            page_content_map = {p.get('url'): p.get('content', '') for p in pages_data}
            categorized_entries = categorize_llmstxt_entries(llms_entries, page_content_map)

            for category, entries_in_cat in categorized_entries.items():
                if entries_in_cat:
                    f.write(f"## {category}\n\n")
                    for entry in entries_in_cat:
                        desc = entry.get('description', 'N/A')
                        # If description is placeholder or missing, try to get a snippet from content
                        if desc == 'N/A (full_text_only mode)' or not desc:
                            page_content = page_content_map.get(entry.get('url',''),'')
                            desc = ' '.join(page_content.split()[:30]) + '...' if page_content else 'No content available.'
                        f.write(f"- [{entry.get('title', 'Untitled')}]({entry.get('url', '#')}): {desc}\n")
                    f.write("\n")
        logger.info(f"📄 Created llms.txt: {llms_filename}")

    # --- llms-full.txt (Full content file) ---
    if include_full_text or full_text_only:
        full_filename = os.path.join(output_dir, f'{domain}-llms-full.txt')
        with open(full_filename, 'w', encoding='utf-8') as f:
            f.write(f"# {base_url} llms-full.txt\n")
            f.write(f"# Generated on {metadata.get('generated_at', '')} using {metadata.get('model_used', 'N/A')}\n")
            f.write(f"# Total pages crawled: {metadata.get('total_pages_crawled', len(pages_data))}\n")
            f.write(f"# Processing time: {metadata.get('processing_time_seconds', 'N/A')}s\n")

            pages_to_write = pages_data
            max_full_pages = metadata.get('max_full_text_pages')
            if max_full_pages is not None and len(pages_data) > max_full_pages:
                pages_to_write = pages_data[:max_full_pages]
                f.write(f"# Displaying {len(pages_to_write)} of {len(pages_data)} crawled pages due to limit.\n")
            f.write("\n")

            for i, page in enumerate(pages_to_write, 1):
                # Use a simple separator for llms-full.txt, not the Crawl4AI specific one
                f.write(f"## Page {i}: {page.get('title', 'Untitled')}\n")
                f.write(f"**URL:** {page.get('url', '#')}\n\n")
                f.write(page.get('content', 'No content available.'))
                f.write('\n\n---\n\n') # Standard markdown separator
        logger.info(f"📄 Created llms-full.txt: {full_filename}")

        # Consider if a "clean" version (without any separators) is still needed.
        # The current llms-full.txt uses standard markdown, which is generally fine.
        # If `remove_page_separators` was for Crawl4AI specific tags, it might not be needed here.

async def write_json_output(
    output_dir: str,
    domain: str,
    llms_entries: List[Dict],
    pages_data: List[Dict],
    metadata: Dict,
    include_full_text: bool
):
    """Write JSON format output."""
    os.makedirs(output_dir, exist_ok=True)
    output_data = {'metadata': metadata, 'llms_entries': llms_entries}
    if include_full_text:
        output_data['full_content_pages'] = pages_data # Use 'pages_data' for consistency

    json_filename = os.path.join(output_dir, f'{domain}-llms.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"📄 Created JSON output: {json_filename}")

async def write_yaml_output(
    output_dir: str,
    domain: str,
    llms_entries: List[Dict],
    pages_data: List[Dict],
    metadata: Dict,
    include_full_text: bool
):
    """Write YAML format output."""
    os.makedirs(output_dir, exist_ok=True)
    output_data = {'metadata': metadata, 'llms_entries': llms_entries}
    if include_full_text:
        output_data['full_content_pages'] = pages_data

    yaml_filename = os.path.join(output_dir, f'{domain}-llms.yaml')
    with open(yaml_filename, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, indent=2)
    logger.info(f"📄 Created YAML output: {yaml_filename}")

async def write_output_files(
    output_dir:str,
    base_url: str,
    llms_entries: List[Dict],
    pages_data: List[Dict],
    metadata: Dict,
    export_format: str,
    include_full_text: bool,
    full_text_only: bool = False
):
    """Orchestrates writing output files in the specified format."""
    domain = extract_domain_from_url(base_url)

    if export_format.lower() == 'json':
        await write_json_output(output_dir, domain, llms_entries, pages_data, metadata, include_full_text)
    elif export_format.lower() == 'yaml':
        await write_yaml_output(output_dir, domain, llms_entries, pages_data, metadata, include_full_text)
    else: # Default to text
        await write_text_output(output_dir, base_url, llms_entries, pages_data, metadata, include_full_text, full_text_only)
