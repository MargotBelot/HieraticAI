"""
HieraticAI: AI-powered hieratic character recognition and validation

This package provides tools for detecting, classifying, and validating hieratic 
characters in ancient Egyptian manuscripts, with specialized focus on the Middle 
Kingdom Westcar Papyrus.

Modules:
    core: Core functionality for model training and inference
    utils: Utility functions for dataset validation and processing  
    models: Model architectures and configurations
    data: Data loading and preprocessing utilities
"""

__version__ = "1.0.0"
__author__ = "Margot Belot, Domino Colyer"
__email__ = "margotbelot@icloud.com"

# Core imports
from .core import *
from .utils import *
