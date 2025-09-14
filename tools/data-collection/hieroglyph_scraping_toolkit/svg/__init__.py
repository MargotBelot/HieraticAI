"""
SVG Processing Package

This package provides comprehensive SVG processing functionality for
hieroglyphic content, including parsing, analysis, manipulation, and conversion.

Modules:
- processor: Main SVG processor with comprehensive functionality

Author: Margot
Date: September 2024
"""

# Import classes explicitly when needed to avoid dependency issues:
# from .processor import SVGProcessor

__all__ = [
    'SVGProcessor'
]

# Lazy import to avoid cairosvg dependency issues
def __getattr__(name):
    if name == 'SVGProcessor':
        from .processor import SVGProcessor
        return SVGProcessor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
