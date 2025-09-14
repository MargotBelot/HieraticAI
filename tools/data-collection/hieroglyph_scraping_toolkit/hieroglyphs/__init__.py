"""
Hieroglyphic Processing Package

This package provides advanced functionality for processing hieroglyphic content,
including Gardiner expression parsing, spatial encoding, and layout algorithms.

Modules:
- parser: Gardiner expression parser with enhanced error handling

Author: Margot
Date: September 2024
"""

# Import classes explicitly when needed:
# from .parser import GardinerExpressionParser, ParsedNode

__all__ = [
    'GardinerExpressionParser',
    'ParsedNode'
]

# Lazy import
def __getattr__(name):
    if name == 'GardinerExpressionParser':
        from .parser import GardinerExpressionParser
        return GardinerExpressionParser
    elif name == 'ParsedNode':
        from .parser import ParsedNode
        return ParsedNode
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
