#!/usr/bin/env python3
"""
Comprehensive Dataset Validation System
Prevents category mapping issues and ensures dataset integrity before training.
"""

import json
import os
import cv2
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Comprehensive dataset validation with category mapping safeguards"""
    
    def __init__(self, dataset_path: str = "hieroglyphs_dataset"):
        self.dataset_path = dataset_path
        self.splits = ['train', 'val', 'test']
        self.validation_results = {}
        
    def validate_all(self) -> bool:
        """Run comprehensive validation on all dataset aspects"""
        logger.info("🔍 Starting comprehensive dataset validation...")
        
        all_valid = True
        
        # 1. File structure validation
        if not self._validate_file_structure():
            all_valid = False
            
        # 2. Category consistency validation  
        if not self._validate_category_consistency():
            all_valid = False
            
        # 3. Category ID validation (critical for preventing off-by-one errors)
        if not self._validate_category_ids():
            all_valid = False
            
        # 4. Annotation validation
        if not self._validate_annotations():
            all_valid = False
            
        # 5. Image validation
        if not self._validate_images():
            all_valid = False
            
        # 6. Data distribution validation
        if not self._validate_data_distribution():
            all_valid = False
            
        # 7. Detectron2 compatibility validation
        if not self._validate_detectron2_compatibility():
            all_valid = False
            
        # Generate validation report
        self._generate_validation_report()
        
        if all_valid:
            logger.info("✅ All dataset validations passed!")
        else:
            logger.error("❌ Dataset validation failed - fix issues before training!")
            
        return all_valid
    
    def _validate_file_structure(self) -> bool:
        """Validate that all required files and directories exist"""
        logger.info("📁 Validating file structure...")
        
        required_paths = []
        for split in self.splits:
            required_paths.extend([
                f"{self.dataset_path}/{split}/images",
                f"{self.dataset_path}/{split}/annotations.json"
            ])
        
        all_exist = True
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                logger.error(f"❌ Missing required path: {path}")
                all_exist = False
                missing_paths.append(path)
            else:
                logger.debug(f"✅ Found: {path}")
        
        self.validation_results['file_structure'] = {
            'valid': all_exist,
            'required_paths': required_paths,
            'missing_paths': missing_paths
        }
        
        return all_exist
    
    def _validate_category_consistency(self) -> bool:
        """Validate category mappings are identical across all splits"""
        logger.info("🏷️  Validating category consistency...")
        
        category_mappings = {}
        
        # Load category mappings from each split
        for split in self.splits:
            ann_file = f"{self.dataset_path}/{split}/annotations.json"
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                categories = {cat['id']: cat['name'] for cat in data['categories']}
                category_mappings[split] = categories
                logger.debug(f"   {split}: {len(categories)} categories")
        
        # Check consistency
        if len(category_mappings) < 2:
            logger.error("❌ Need at least 2 splits to verify consistency")
            return False
        
        base_split = list(category_mappings.keys())[0]
        base_mapping = category_mappings[base_split]
        
        consistent = True
        for split, mapping in category_mappings.items():
            if split == base_split:
                continue
                
            if mapping != base_mapping:
                logger.error(f"❌ Category mapping differs between {base_split} and {split}")
                consistent = False
                
                # Log specific differences
                base_set = set(base_mapping.items())
                split_set = set(mapping.items())
                
                only_base = base_set - split_set
                only_split = split_set - base_set
                
                if only_base:
                    logger.error(f"   Only in {base_split}: {len(only_base)} categories")
                if only_split:
                    logger.error(f"   Only in {split}: {len(only_split)} categories")
        
        if consistent:
            logger.info("✅ Category mappings are consistent across all splits")
        
        self.validation_results['category_consistency'] = {
            'valid': consistent,
            'mappings': category_mappings,
            'base_mapping': base_mapping if consistent else None
        }
        
        return consistent
    
    def _validate_category_ids(self) -> bool:
        """Critical validation for category IDs to prevent off-by-one errors"""
        logger.info("🔢 Validating category IDs (critical for Detectron2)...")
        
        valid = True
        
        # Load category data from first split
        ann_file = f"{self.dataset_path}/{self.splits[0]}/annotations.json"
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        category_ids = [cat['id'] for cat in data['categories']]
        min_id = min(category_ids)
        max_id = max(category_ids)
        num_categories = len(category_ids)
        
        logger.info(f"   Category ID range: {min_id} to {max_id}")
        logger.info(f"   Number of categories: {num_categories}")
        
        # Critical checks for Detectron2 compatibility
        issues = []
        
        # Check 1: IDs should be continuous
        expected_range = set(range(min_id, max_id + 1))
        actual_range = set(category_ids)
        gaps = expected_range - actual_range
        
        if gaps:
            gap_str = str(sorted(gaps)[:10]) + ('...' if len(gaps) > 10 else '')
            issues.append(f"Missing category IDs: {gap_str}")
            logger.error(f"❌ Category ID gaps detected: {len(gaps)} missing IDs")
            valid = False
        
        # Check 2: No duplicate IDs
        if len(category_ids) != len(set(category_ids)):
            duplicates = [id for id, count in Counter(category_ids).items() if count > 1]
            issues.append(f"Duplicate category IDs: {duplicates}")
            logger.error(f"❌ Duplicate category IDs: {duplicates}")
            valid = False
        
        # Check 3: Warn about starting ID (Detectron2 expects 0-based)
        if min_id == 1:
            logger.warning("⚠️  Category IDs start from 1 (dataset format)")
            logger.warning("   Detectron2 expects 0-based IDs - will need remapping during training")
            logger.warning("   This is the ROOT CAUSE of the previous off-by-one errors!")
        elif min_id == 0:
            logger.info("✅ Category IDs start from 0 (Detectron2 compatible)")
        else:
            issues.append(f"Category IDs start from {min_id} (neither 0 nor 1)")
            logger.error(f"❌ Unusual category ID start: {min_id}")
            valid = False
        
        # Check 4: Reasonable range
        if max_id > 10000:
            issues.append(f"Very high category IDs (max: {max_id})")
            logger.warning(f"⚠️  Very high category ID: {max_id}")
        
        if valid and not issues:
            logger.info("✅ Category IDs are valid")
        
        self.validation_results['category_ids'] = {
            'valid': valid,
            'min_id': min_id,
            'max_id': max_id,
            'num_categories': num_categories,
            'detectron2_compatible': min_id == 0,
            'needs_remapping': min_id == 1,
            'issues': issues
        }
        
        return valid
    
    def _validate_detectron2_compatibility(self) -> bool:
        """Validate Detectron2-specific compatibility requirements"""
        logger.info("🔧 Validating Detectron2 compatibility...")
        
        valid = True
        issues = []
        
        # Check category ID compatibility (most critical)
        if 'category_ids' in self.validation_results:
            cat_results = self.validation_results['category_ids']
            if not cat_results['detectron2_compatible']:
                if cat_results['needs_remapping']:
                    issues.append("Category IDs start from 1 - need 0-based remapping for Detectron2")
                    logger.warning("⚠️  Category IDs need remapping: 1-based -> 0-based")
                else:
                    issues.append(f"Category IDs start from {cat_results['min_id']} - not Detectron2 compatible")
                    logger.error(f"❌ Category IDs start from {cat_results['min_id']} - fix required")
                    valid = False
        
        self.validation_results['detectron2_compatibility'] = {
            'valid': valid,
            'issues': issues,
            'needs_category_remapping': not self.validation_results.get('category_ids', {}).get('detectron2_compatible', True)
        }
        
        if valid and not issues:
            logger.info("✅ Detectron2 compatibility checks passed")
        elif issues:
            logger.warning(f"⚠️  Detectron2 compatibility issues (can be fixed): {len(issues)}")
        
        return valid
    
    def _validate_annotations(self) -> bool:
        """Validate annotation format and content"""
        logger.info("📝 Validating annotations...")
        
        valid = True
        
        for split in self.splits:
            ann_file = f"{self.dataset_path}/{split}/annotations.json"
            if not os.path.exists(ann_file):
                continue
                
            logger.debug(f"   Checking {split} annotations...")
            
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Validate COCO format structure
            required_keys = ['images', 'annotations', 'categories']
            for key in required_keys:
                if key not in data:
                    logger.error(f"❌ Missing '{key}' in {split} annotations")
                    valid = False
            
            if 'annotations' in data:
                # Validate annotation fields (check first 5 for performance)
                for i, ann in enumerate(data['annotations'][:5]):
                    required_ann_keys = ['id', 'image_id', 'category_id', 'bbox', 'area', 'iscrowd']
                    for key in required_ann_keys:
                        if key not in ann:
                            logger.error(f"❌ Missing '{key}' in annotation {i} of {split}")
                            valid = False
                            break
                    
                    # Validate bbox format [x, y, width, height]
                    if 'bbox' in ann:
                        bbox = ann['bbox']
                        if not (isinstance(bbox, list) and len(bbox) == 4):
                            logger.error(f"❌ Invalid bbox format in {split}: {bbox}")
                            valid = False
                        elif any(v < 0 for v in bbox):
                            logger.warning(f"⚠️  Negative bbox values in {split}: {bbox}")
            
            # Check category ID references
            if 'categories' in data and 'annotations' in data:
                valid_cat_ids = set(cat['id'] for cat in data['categories'])
                used_cat_ids = set(ann['category_id'] for ann in data['annotations'])
                invalid_refs = used_cat_ids - valid_cat_ids
                
                if invalid_refs:
                    logger.error(f"❌ Invalid category ID references in {split}: {sorted(invalid_refs)}")
                    valid = False
        
        self.validation_results['annotations'] = {'valid': valid}
        
        if valid:
            logger.info("✅ Annotations are valid")
        
        return valid
    
    def _validate_images(self) -> bool:
        """Validate image files and their correspondence with annotations"""
        logger.info("🖼️  Validating images...")
        
        valid = True
        
        for split in self.splits:
            img_dir = f"{self.dataset_path}/{split}/images"
            ann_file = f"{self.dataset_path}/{split}/annotations.json"
            
            if not (os.path.exists(img_dir) and os.path.exists(ann_file)):
                continue
            
            logger.debug(f"   Checking {split} images...")
            
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Get image filenames from annotations
            if 'images' in data:
                ann_images = {img['file_name']: img for img in data['images']}
                
                # Check first few images for performance
                for filename, img_info in list(ann_images.items())[:3]:
                    img_path = os.path.join(img_dir, filename)
                    if not os.path.exists(img_path):
                        logger.error(f"❌ Missing image file: {img_path}")
                        valid = False
                    else:
                        # Validate image can be loaded
                        try:
                            img = cv2.imread(img_path)
                            if img is None:
                                logger.error(f"❌ Cannot load image: {img_path}")
                                valid = False
                            else:
                                h, w = img.shape[:2]
                                # Check if dimensions match annotation
                                if 'width' in img_info and 'height' in img_info:
                                    if w != img_info['width'] or h != img_info['height']:
                                        logger.error(f"❌ Image dimensions mismatch: {filename}")
                                        logger.error(f"   File: {w}x{h}, Annotation: {img_info['width']}x{img_info['height']}")
                                        valid = False
                        except Exception as e:
                            logger.error(f"❌ Error loading {img_path}: {e}")
                            valid = False
        
        self.validation_results['images'] = {'valid': valid}
        
        if valid:
            logger.info("✅ Images are valid")
        
        return valid
    
    def _validate_data_distribution(self) -> bool:
        """Validate data distribution and identify potential training issues"""
        logger.info("📊 Validating data distribution...")
        
        valid = True
        warnings = []
        
        for split in self.splits:
            ann_file = f"{self.dataset_path}/{split}/annotations.json"
            if not os.path.exists(ann_file):
                continue
                
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            if 'annotations' not in data:
                continue
            
            # Count annotations per category
            cat_counts = Counter(ann['category_id'] for ann in data['annotations'])
            
            # Check for categories with very few examples
            min_samples_threshold = 3
            low_sample_cats = {cat_id: count for cat_id, count in cat_counts.items() 
                             if count < min_samples_threshold}
            
            if low_sample_cats:
                warning_msg = f"{split}: {len(low_sample_cats)} categories with <{min_samples_threshold} samples"
                warnings.append(warning_msg)
                logger.warning(f"⚠️  {warning_msg}")
            
            # Check for completely missing categories
            if 'categories' in data:
                all_cat_ids = set(cat['id'] for cat in data['categories'])
                used_cat_ids = set(cat_counts.keys())
                unused_cats = all_cat_ids - used_cat_ids
                
                if unused_cats:
                    warning_msg = f"{split}: {len(unused_cats)} categories with 0 samples"
                    warnings.append(warning_msg)
                    logger.warning(f"⚠️  {warning_msg}")
        
        self.validation_results['data_distribution'] = {
            'valid': True,  # Distribution issues are warnings, not failures
            'warnings': warnings
        }
        
        if not warnings:
            logger.info("✅ Data distribution looks good")
        
        return valid
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("📋 Generating validation report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': self.dataset_path,
            'validation_results': self.validation_results,
            'summary': self._generate_summary()
        }
        
        # Save report
        with open('dataset_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("💾 Validation report saved: dataset_validation_report.json")
    
    def _generate_summary(self) -> Dict:
        """Generate validation summary"""
        summary = {
            'overall_valid': all(result.get('valid', False) for result in self.validation_results.values()),
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Collect critical issues
        for check, result in self.validation_results.items():
            if not result.get('valid', True):
                summary['critical_issues'].append(f"{check}: validation failed")
        
        # Collect warnings
        if 'data_distribution' in self.validation_results:
            summary['warnings'].extend(self.validation_results['data_distribution'].get('warnings', []))
        
        if 'detectron2_compatibility' in self.validation_results:
            summary['warnings'].extend(self.validation_results['detectron2_compatibility'].get('issues', []))
        
        # Generate recommendations
        if self.validation_results.get('category_ids', {}).get('needs_remapping', False):
            summary['recommendations'].append(
                "Use category ID remapping during training to convert 1-based IDs to 0-based (prevents off-by-one errors)"
            )
        
        if summary['critical_issues']:
            summary['recommendations'].append(
                "Fix all critical issues before training to prevent model failures"
            )
        
        if not summary['critical_issues'] and not summary['warnings']:
            summary['recommendations'].append(
                "Dataset is ready for training - no issues detected"
            )
        
        return summary

def main():
    """Main validation function"""
    print("🏺 Comprehensive Dataset Validation")
    print("=" * 60)
    
    validator = DatasetValidator()
    
    # Run validation
    is_valid = validator.validate_all()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    
    if is_valid:
        print("✅ Dataset validation PASSED")
        print("   Dataset is ready for training")
    else:
        print("❌ Dataset validation FAILED")
        print("   Fix critical issues before training")
    
    # Show key recommendations
    summary = validator._generate_summary()
    
    if summary['critical_issues']:
        print(f"\n🚨 Critical Issues ({len(summary['critical_issues'])}):")    
        for issue in summary['critical_issues']:
            print(f"   - {issue}")
    
    if summary['warnings']:
        print(f"\n⚠️  Warnings ({len(summary['warnings'])}):")    
        for warning in summary['warnings']:
            print(f"   - {warning}")
    
    if summary['recommendations']:
        print(f"\n💡 Recommendations:")    
        for rec in summary['recommendations']:
            print(f"   - {rec}")
    
    print(f"\n📋 Detailed report: dataset_validation_report.json")
    
    return is_valid

if __name__ == "__main__":
    main()
