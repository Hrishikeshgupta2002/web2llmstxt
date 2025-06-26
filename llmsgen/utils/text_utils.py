import re
from typing import Dict, List, Any
from urllib.parse import urlparse

from llmsgen.config import logger # Assuming logger is configured in config.py

# --- Text Cleaning and Formatting ---
def remove_page_separators(text: str) -> str:
    """Remove page separators and excessive newlines from text."""
    text = re.sub(r'<\|crawl4ai-page-\d+-lllmstxt\|>\n', '', text)
    text = re.sub(r'<!-- .* -->\n', '', text)
    text = re.sub(r'\n---\n\n', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def limit_pages_in_full_text(full_text: str, max_pages: int) -> str:
    """Limit the number of pages in a pre-formatted full text string."""
    pages = full_text.split('<|crawl4ai-page-')
    if len(pages) <= 1: # No page separators or only one page
        return full_text

    header = pages[0]
    content_pages = pages[1:]

    # Add up to max_pages
    limited_content_pages = content_pages[:max_pages]

    return header + ''.join(['<|crawl4ai-page-' + p for p in limited_content_pages])

def clean_title(title: str) -> str:
    """Clean and normalize page titles."""
    if not title: return "Untitled Page"
    title_str = str(title)
    title_str = re.sub(r'&[a-zA-Z0-9#]+;', ' ', title_str)  # HTML entities
    title_str = re.sub(r'<[^>]+>', '', title_str)          # HTML tags

    # Principle: Remove suffix after the last separator (|, -, –, —)
    # This regex matches: optional spaces, a separator, optional spaces,
    # then one or more characters that are NOT separators, until the end of the string.
    title_str = re.sub(r'\s*[-|–—]\s*[^-|–—]+$', '', title_str)

    title_str = ' '.join(title_str.split()) # Normalize whitespace

    if title_str and (title_str.islower() or title_str.isupper()):
        title_str = title_str.title()
    return title_str.strip() if title_str.strip() else "Untitled Page"

def clean_content_text_for_processing(content: str) -> str:
    """Clean raw content text for better NLP processing or summarization."""
    if not content: return ""
    content = re.sub(r'\s+', ' ', content) # Normalize whitespace
    # Remove common UI/navigation patterns that might interfere with summarization
    ui_patterns = [
        r'\b(?:click here|read more|learn more|sign up|log in|subscribe|search|filter|sort by|view all)\b',
        r'\b(?:menu|navigation|nav|header|footer|sidebar|skip to|go to|back to|return to)\b',
        r'\b(?:cookie(?:s)?\s+(?:policy|notice|consent)|privacy\s+policy|terms\s+(?:of\s+)?(?:service|use))\b',
        r'\bcopyright\s*(?:©|&copy;)?\s*\d{4}\b', r'\ball\s+rights\s+reserved\b',
        r'\bfollow\s+us\s+on\b',
    ]
    for pattern in ui_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    return content.strip()

def is_navigation_text(text: str, threshold_ratio: float = 0.3) -> bool:
    """Check if text is likely navigation/UI content based on keyword density."""
    text_lower = text.lower()
    nav_indicators = [
        'click here', 'read more', 'learn more', 'sign up', 'log in', 'subscribe',
        'newsletter', 'follow us', 'social media', 'cookie', 'privacy policy',
        'terms of service', 'copyright', 'all rights reserved', 'menu',
        'navigation', 'back to top', 'skip to content', 'search', 'filter', 'sort by'
    ]
    nav_count = sum(1 for indicator in nav_indicators if indicator in text_lower)
    word_count = len(text.split())
    if word_count == 0: return False # Avoid division by zero
    return (nav_count / word_count) > threshold_ratio

def is_good_sentence_for_summary(sentence: str, min_len: int = 15, max_len: int = 200,
                                 alphanum_ratio: float = 0.7) -> bool:
    """Check if a sentence is suitable for inclusion in a summary."""
    sentence = sentence.strip()
    if not (min_len <= len(sentence) <= max_len): return False
    if is_navigation_text(sentence, threshold_ratio=0.5): return False # Higher threshold for single sentences

    num_alphanum = sum(1 for char in sentence if char.isalnum() or char.isspace())
    if len(sentence) > 0 and (num_alphanum / len(sentence)) < alphanum_ratio: return False

    # Check for presence of at least one common verb or auxiliary verb if sentence is short
    meaningful_words = ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will', 'provides', 'offers', 'describes']
    if len(sentence.split()) < 5 and not any(word in sentence.lower() for word in meaningful_words):
        return False
    return True

def clean_sentence_for_output(sentence: str) -> str:
    """Clean an individual sentence for final output."""
    if not sentence: return ""
    sentence = ' '.join(sentence.strip().split()) # Normalize whitespace
    sentence = re.sub(r'[.!?]{2,}$', '.', sentence) # Remove trailing punctuation repetition
    if sentence and sentence[0].islower(): # Ensure proper capitalization
        sentence = sentence[0].upper() + sentence[1:]
    if sentence and not sentence.endswith(('.', '!', '?')):
        sentence += '.'
    return sentence

# --- URL and Domain Utilities ---
def extract_domain_from_url(url: str) -> str:
    """Extract a clean domain name from URL for filename generation or display."""
    try:
        if not url: # Handle empty string case
            return "unknown_domain"

        # Prepend http:// if no scheme is present to help urlparse
        if '://' not in url:
            url_to_parse = 'http://' + url
        else:
            url_to_parse = url

        parsed_url = urlparse(url_to_parse)
        domain = parsed_url.netloc.replace("www.", "")

        # If domain is empty or doesn't look like a domain (e.g., no dots, or is the same as original path-like input)
        if not domain or ('.' not in domain and domain == url_to_parse.split('://')[-1].split('/')[0]):
            # Check if the original input (if it had no scheme) was treated as path by urlparse
            if '://' not in url and parsed_url.path == url and not parsed_url.netloc:
                 return "unknown_domain"
            if not domain: # If still no domain after initial checks
                return "unknown_domain"
            # If domain has no dots, it's likely not a valid TLD-based domain.
            if '.' not in domain:
                return "unknown_domain"

        # Clean domain for safe filename use
        return re.sub(r'[^\w\-_.]', '_', domain)
    except Exception: # Fallback for truly malformed URLs that urlparse might still struggle with
        return "unknown_domain"

# --- Content Analysis and Description Generation Helpers ---
def detect_hallucination(description: str, title: str, content_sample: str) -> bool:
    """Detect if the AI generated hallucinated or unrelated content."""
    description_lower = description.lower()
    title_lower = title.lower()
    content_sample_lower = content_sample[:500].lower() # Use a sample of content

    hallucination_indicators = [
        'game character', 'tasks a-j', 'proof by contradiction', 'tree of thought',
        'let\'s say we have', 'consider each of these tasks', 'shortest path',
        'deploy tool a', 'tool b', 'tool c', 'character can only work',
        'proof by exhaustion', 'direct proof and inductive logic',
        # Add more specific irrelevant phrases if observed
    ]
    if any(indicator in description_lower for indicator in hallucination_indicators):
        logger.warning(f"🚨 Hallucination indicator found in description for '{title}'")
        return True

    # Check for relevance (simple keyword overlap)
    common_words = {'the','and','or','but','in','on','at','to','for','of','with','by','a','an','is','are','was','were','this','that'}
    desc_words = set(description_lower.split()) - common_words
    title_words = set(title_lower.split()) - common_words
    content_words = set(content_sample_lower.split()) - common_words

    title_overlap = len(desc_words & title_words) / len(title_words) if title_words else 0
    content_overlap = len(desc_words & content_words) / len(content_words) if content_words else 0

    if title_overlap < 0.1 and content_overlap < 0.05: # Thresholds might need tuning
        logger.warning(f"🚨 Low relevance detected in description for '{title}'")
        return True
    return False

def extract_key_sentences_from_content(title: str, content: str, num_sentences: int = 3, max_chars_per_sentence: int = 150) -> str:
    """Extract key sentences from page content for summarization or direct use."""
    cleaned_content = clean_content_text_for_processing(content)
    if not cleaned_content or len(cleaned_content.strip()) < 50:
        return clean_title(title) if title else "Website content summary."

    sentences = []
    # Try paragraphs first
    paragraphs = [p.strip() for p in cleaned_content.split('\n\n') if p.strip()]
    for para in paragraphs[:5]: # Check first few paragraphs
        if len(sentences) >= num_sentences: break
        para_sentences = re.split(r'(?<=[.!?])\s+', para) # Split sentences
        for sent in para_sentences[:2]: # Take first 1-2 sentences from good paragraphs
            if is_good_sentence_for_summary(sent, max_len=max_chars_per_sentence):
                sentences.append(clean_sentence_for_output(sent))
                if len(sentences) >= num_sentences: break

    # Try list items if not enough sentences
    if len(sentences) < num_sentences:
        list_items = re.findall(r'(?:^|\n)[-*•]\s*([^\n]{20,'+str(max_chars_per_sentence)+'})', cleaned_content, re.MULTILINE)
        for item in list_items:
            if len(sentences) >= num_sentences: break
            if is_good_sentence_for_summary(item, max_len=max_chars_per_sentence):
                sentences.append(clean_sentence_for_output(item))

    if sentences:
        return ' '.join(sentences)

    # Fallback if no good sentences found
    return create_initial_content_description(title, content)


def create_initial_content_description(title: str, content_sample: str) -> str:
    """Create a very basic description based on title and content type hints."""
    content_lower = content_sample[:1000].lower() # Analyze a larger sample for type
    ct = clean_title(title)

    if 'api' in content_lower and ('documentation' in content_lower or 'docs' in content_lower): return f"API documentation for {ct}."
    if 'tutorial' in content_lower or 'guide' in content_lower: return f"Tutorial and guide for {ct}."
    if 'pricing' in content_lower or 'plans' in content_lower: return f"Pricing information for {ct}."
    if 'features' in content_lower or 'capabilities' in content_lower: return f"Features and capabilities of {ct}."
    if 'blog' in content_lower or 'news' in content_lower or 'article' in content_lower : return f"Blog post or article about {ct}."
    if 'about' in content_lower or 'company' in content_lower : return f"Information about {ct}."
    return f"Content related to {ct}."

def create_smart_fallback_description(title: str, content_sample: str, url: str) -> str:
    """Create a more intelligent fallback description if AI summarization fails."""
    # This can be more sophisticated, using more keywords or patterns
    domain = extract_domain_from_url(url)
    base_desc = create_initial_content_description(title, content_sample)
    if domain != "unknown_domain" and domain not in base_desc.lower():
        return f"{base_desc} From {domain}."
    return base_desc

# --- LLMS.TXT Specific Formatting ---
def extract_site_name_for_llmstxt(base_url: str, pages_data: List[Dict[str, Any]]) -> str:
    """Extract a clean site name for llms.txt H1 header."""
    if pages_data: # Check if pages_data is not empty
        main_page = next((p for p in pages_data if p.get('url') == base_url or p.get('url') == base_url.rstrip('/')), None)
        if main_page and main_page.get('title'):
            title = clean_title(main_page['title']) # Use the cleaned title
            # Further simplify if it's a common pattern like "Homepage - Site Name"
            if title and len(title) > 3: return title

    # Fallback to domain name
    domain = extract_domain_from_url(base_url)
    return domain.replace('_', ' ').title() if domain != "unknown_domain" else "Website"


def generate_site_summary_for_llmstxt(pages_data: List[Dict[str, Any]], num_page_samples: int = 5) -> str:
    """Generate a concise site summary for the llms.txt blockquote."""
    if not pages_data: return "A website with various content and resources."

    # Analyze content from a sample of pages
    content_sample = ' '.join([
        (p.get('content', '')[:500] or "").lower()
        for p in pages_data[:num_page_samples]
    ])

    if not content_sample.strip(): return "A website with various content and resources."

    if any(k in content_sample for k in ['api', 'documentation', 'docs', 'developer', 'reference']):
        return "Software documentation, API references, and developer resources."
    if any(k in content_sample for k in ['pricing', 'plans', 'subscription', 'buy', 'purchase', 'checkout']):
        return "Information on products, services, and pricing plans."
    if any(k in content_sample for k in ['blog', 'article', 'news', 'post', 'insights']):
        return "A collection of articles, blog posts, and news updates."
    if any(k in content_sample for k in ['tutorial', 'guide', 'how to', 'learn', 'course']):
        return "Educational content, tutorials, and learning materials."
    if any(k in content_sample for k in ['product', 'service', 'solution', 'tool', 'feature']):
        return "Details about products, services, and their features."
    if any(k in content_sample for k in ['about us', 'company', 'mission', 'team']):
        return "Information about the company, its mission, and team."
    return "A comprehensive website offering information and resources on various topics."


def categorize_llmstxt_entries(
    entries: List[Dict[str, Any]],
    pages_content_map: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize llms.txt entries into logical sections based on URL and content hints."""
    categories: Dict[str, List[Dict[str, Any]]] = {
        "Key Documentation": [], "API & Technical Reference": [],
        "Products & Services": [], "Guides & Tutorials": [],
        "Blog & Resources": [], "General Information": [], "Other Pages": []
    }

    for entry in entries:
        url_lower = entry.get('url', '').lower()
        title_lower = entry.get('title', '').lower()
        # Use pre-fetched content sample for categorization
        content_sample = (pages_content_map.get(entry.get('url', ''), '')[:300] or "").lower()

        assigned = False
        if any(k in url_lower or k in title_lower or k in content_sample for k in ['/api', '/reference', 'api docs', 'developer.']):
            categories["API & Technical Reference"].append(entry); assigned = True
        elif any(k in url_lower or k in title_lower or k in content_sample for k in ['/docs', '/documentation', 'readme', 'manual']):
            categories["Key Documentation"].append(entry); assigned = True
        elif any(k in url_lower or k in title_lower or k in content_sample for k in ['/guide', '/tutorial', 'how-to', 'learn', 'getting-started']):
            categories["Guides & Tutorials"].append(entry); assigned = True
        elif any(k in url_lower or k in title_lower or k in content_sample for k in ['/product', '/service', '/feature', 'pricing', 'plans', 'tool']):
            categories["Products & Services"].append(entry); assigned = True
        elif any(k in url_lower or k in title_lower or k in content_sample for k in ['/blog', '/news', '/article', '/resource']):
            categories["Blog & Resources"].append(entry); assigned = True
        elif any(k in url_lower or k in title_lower for k in ['about', 'contact', 'company', 'team', 'mission']):
            categories["General Information"].append(entry); assigned = True

        if not assigned:
            if any(k in url_lower for k in ['?page=', '/page/', '/compare', '/vs', 'tag/', 'category/']):
                 categories["Other Pages"].append(entry)
            else: # Default if no strong signal
                 categories["General Information"].append(entry)

    return {k: v for k, v in categories.items() if v} # Remove empty categories
