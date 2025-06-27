#!/usr/bin/env python3
"""
Text utility functions for processing and cleaning content.
"""

import re
import string
from typing import List, Optional


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Remove excessive newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Clean up common web artifacts
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    
    return text.strip()


def extract_key_sentences(title: str, content: str, max_sentences: int = 3) -> str:
    """Extract key sentences from content for description generation"""
    if not content:
        return ""
    
    # Clean the content first
    clean_content = clean_text(content)
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', clean_content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return ""
    
    # Filter good sentences
    good_sentences = []
    title_words = set(title.lower().split()) if title else set()
    
    for sentence in sentences:
        if _is_good_sentence(sentence, title_words):
            good_sentences.append(sentence)
        
        # Stop when we have enough good sentences
        if len(good_sentences) >= max_sentences:
            break
    
    if not good_sentences:
        # Fallback to first few sentences if no "good" ones found
        good_sentences = sentences[:max_sentences]
    
    # Join and clean up the result
    result = '. '.join(good_sentences)
    if result and not result.endswith('.'):
        result += '.'
    
    return result


def _is_good_sentence(sentence: str, title_words: set) -> bool:
    """Check if a sentence is good for content description"""
    sentence = sentence.strip()
    
    # Basic length check
    if len(sentence) < 20 or len(sentence) > 300:
        return False
    
    # Check for navigation text
    if _is_navigation_text(sentence):
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
    
    # Prefer sentences that contain title words
    sentence_words = set(sentence.lower().split())
    title_overlap = len(title_words & sentence_words) / max(len(title_words), 1)
    
    # Bonus for sentences with title words
    if title_overlap > 0.3:
        return True
    
    # Check for informative content indicators
    informative_indicators = [
        'explain', 'describe', 'overview', 'introduction', 'guide', 'tutorial',
        'learn', 'understand', 'concept', 'principle', 'method', 'approach',
        'technique', 'strategy', 'solution', 'benefit', 'advantage', 'feature'
    ]
    
    sentence_lower = sentence.lower()
    has_informative_content = any(indicator in sentence_lower for indicator in informative_indicators)
    
    return has_informative_content


def _is_navigation_text(text: str) -> bool:
    """Check if text is likely navigation/menu content"""
    nav_indicators = [
        'menu', 'navigation', 'nav', 'breadcrumb', 'sidebar',
        'footer', 'header', 'skip to', 'toggle', 'dropdown',
        'click here', 'read more', 'learn more', 'see all',
        'view all', 'show more', 'load more', 'back to top',
        'home', 'contact', 'about us', 'privacy policy',
        'terms of service', 'cookie policy'
    ]
    
    text_lower = text.lower().strip()
    if len(text_lower) < 5:
        return True
        
    # Short texts with nav indicators are likely navigation
    if len(text_lower) < 50:
        return any(indicator in text_lower for indicator in nav_indicators)
    
    return False


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """Truncate text to a maximum length"""
    if not text or len(text) <= max_length:
        return text
    
    if add_ellipsis and max_length > 3:
        return text[:max_length-3] + "..."
    else:
        return text[:max_length]


def extract_words(text: str, min_length: int = 3) -> List[str]:
    """Extract meaningful words from text"""
    if not text:
        return []
    
    # Remove punctuation and split
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator)
    words = clean_text.split()
    
    # Filter by length and remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'can', 'this', 'that', 'these', 'those', 'a', 'an'
    }
    
    meaningful_words = [
        word.lower() for word in words 
        if len(word) >= min_length and word.lower() not in stop_words
    ]
    
    return meaningful_words


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity based on word overlap"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(extract_words(text1))
    words2 = set(extract_words(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0 