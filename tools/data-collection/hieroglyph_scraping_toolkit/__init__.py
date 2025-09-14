"""
Hieroglyph Scraping Toolkit

A comprehensive toolkit for hieroglyphic text processing, web scraping,
and digital reconstruction of ancient Egyptian texts.

This package provides modular functionality for:
- Configuration management
- Web scraping from hieroglyphic databases
- SVG processing and analysis
- Gardiner expression parsing
- Spatial encoding and layout
- Text reconstruction and rendering

Author: Margot
Date: September 2024
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Margot"
__email__ = ""
__description__ = "Comprehensive toolkit for hieroglyphic text processing and reconstruction"

# Core modules
from .config import (
    ConfigManager,
    PathConfig,
    LayoutConfiguration,
    WebScrapingConfig,
    get_config_manager,
    create_default_config_file
)

from .utils import (
    setup_logging,
    ensure_directory,
    safe_file_read,
    safe_file_write,
    safe_json_load,
    safe_json_save,
    timing_decorator,
    retry_on_exception,
    performance_tracker,
    PerformanceTracker,
    progress_bar
)

# Sub-packages are available but not automatically imported
# Import them explicitly when needed:
# from hieroglyph_scraping_toolkit.scraping import AKUPALScraper
# from hieroglyph_scraping_toolkit.svg import SVGProcessor
# from hieroglyph_scraping_toolkit.hieroglyphs import GardinerExpressionParser

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__description__',
    
    # Configuration
    'ConfigManager',
    'PathConfig', 
    'LayoutConfiguration',
    'WebScrapingConfig',
    'get_config_manager',
    'create_default_config_file',
    
    # Utilities
    'setup_logging',
    'ensure_directory',
    'safe_file_read',
    'safe_file_write',
    'safe_json_load',
    'safe_json_save',
    'timing_decorator',
    'retry_on_exception',
    'performance_tracker',
    'PerformanceTracker',
    'progress_bar',
]
