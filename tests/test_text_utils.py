import unittest
import sys
import os

# Adjust path to import from llmsgen
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llmsgen.utils.text_utils import (
    clean_title,
    extract_domain_from_url,
    is_navigation_text,
    # Placeholder for other functions if we add more tests
)

class TestTextUtils(unittest.TestCase):

    def test_clean_title(self):
        self.assertEqual(clean_title("  My Awesome Page | SiteName  "), "My Awesome Page")
        self.assertEqual(clean_title("my awesome page - site name"), "My Awesome Page")
        self.assertEqual(clean_title("Untitled Page"), "Untitled Page")
        self.assertEqual(clean_title("PAGE IN CAPS"), "Page In Caps")
        self.assertEqual(clean_title("page with &amp; entity"), "Page With Entity") # Corrected assertion
        self.assertEqual(clean_title(""), "Untitled Page")
        self.assertEqual(clean_title(None), "Untitled Page")
        self.assertEqual(clean_title("Simple Title"), "Simple Title")
        self.assertEqual(clean_title("Title - Suffix"), "Title") # Suffix removal
        self.assertEqual(clean_title("Prefix - Title"), "Prefix") # Suffix removal principle applied


    def test_extract_domain_from_url(self):
        self.assertEqual(extract_domain_from_url("https://www.example.com/path?query=1"), "example.com")
        self.assertEqual(extract_domain_from_url("http://sub.example.co.uk/path"), "sub.example.co.uk")
        self.assertEqual(extract_domain_from_url("ftp://example.com"), "example.com")
        self.assertEqual(extract_domain_from_url("example.com/path"), "example.com") # Assumes http if no scheme
        self.assertEqual(extract_domain_from_url("invalid_url"), "unknown_domain") # Invalid URL
        self.assertEqual(extract_domain_from_url("https://www.test-site.com"), "test-site.com")

    def test_is_navigation_text(self):
        self.assertFalse(is_navigation_text("click here to read more about our services")) # 2 indicators / 8 words = 0.25. 0.25 > 0.3 is False.
        self.assertTrue(is_navigation_text("menu navigation sidebar footer")) # 4 indicators / 4 words = 1.0. 1.0 > 0.3 is True.
        self.assertFalse(is_navigation_text("Learn more about our privacy policy and terms of service.")) # 3 indicators / 10 words = 0.3. 0.3 > 0.3 is False.
        self.assertFalse(is_navigation_text("This is a regular sentence about the main content of the page."))
        self.assertFalse(is_navigation_text("An important heading for a section."))
        self.assertTrue(is_navigation_text("Search filter sort by view all", threshold_ratio=0.2)) # Test threshold
        self.assertFalse(is_navigation_text("Short", threshold_ratio=0.5)) # Edge case with short text
        self.assertFalse(is_navigation_text(""))


if __name__ == '__main__':
    unittest.main()
