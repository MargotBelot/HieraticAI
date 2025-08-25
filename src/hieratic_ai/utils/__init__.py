"""
Utility modules for HieraticAI

This package contains utility functions for dataset validation, category mapping fixes,
and data consistency verification.

Modules:
    dataset_validator: Comprehensive dataset validation system
    fix_category_id_offset: Category ID offset correction utilities
    verify_dataset_consistency: Dataset consistency verification tools
"""

from .dataset_validator import DatasetValidator
from .fix_category_id_offset import *
from .verify_dataset_consistency import *

__all__ = [
    'DatasetValidator',
    'fix_prediction_category_ids',
    'validate_fixed_predictions', 
    'create_corrected_visualization',
    'load_annotation_data',
    'verify_category_consistency',
    'check_annotation_distribution'
]
