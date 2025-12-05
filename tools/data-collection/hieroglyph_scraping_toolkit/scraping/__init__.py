"""
Web Scraping Package

This package provides web scraping functionality for hieroglyphic databases
and archaeological websites, with specialized support for the AKU-PAL database.

Modules:
- base_scraper: Base scraper class with common functionality
- aku_scraper: Specialized scraper for AKU-PAL website

Author: Margot
Date: September 2024
"""

# Import classes explicitly when needed to avoid dependency issues:
# from .base_scraper import BaseScraper
# from .aku_scraper import AKUPALScraper

__all__ = ["BaseScraper", "AKUPALScraper"]


# Lazy import to avoid selenium dependency issues
def __getattr__(name):
    if name == "BaseScraper":
        from .base_scraper import BaseScraper

        return BaseScraper
    elif name == "AKUPALScraper":
        from .aku_scraper import AKUPALScraper

        return AKUPALScraper
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
