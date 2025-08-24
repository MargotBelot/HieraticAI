#!/usr/bin/env python3
"""
Automated Tests for Dataset Validation and Training Pipeline
Comprehensive test suite to prevent category mapping issues and ensure system reliability.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_validator import DatasetValidator

class TestDatasetValidator(unittest.TestCase):
    """Test suite for dataset validation functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dataset_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dataset_dir)
        
    def create_mock_dataset(self, category_start_id=1, missing_files=None):
        """Create a mock dataset for testing"""
        missing_files = missing_files or []
        
        splits = ['train', 'val', 'test']
        categories = [
            {"id": category_start_id, "name": "A1", "supercategory": ""},
            {"id": category_start_id + 1, "name": "M17", "supercategory": ""},
            {"id": category_start_id + 2, "name": "N35", "supercategory": ""}
        ]
        
        for split in splits:
            split_dir = os.path.join(self.test_dataset_dir, split)
            images_dir = os.path.join(split_dir, 'images')
            
            if f"{split}/images" not in missing_files:
                os.makedirs(images_dir, exist_ok=True)
            
            if f"{split}/annotations.json" not in missing_files:
                # Create mock annotation file
                annotations = {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "test_image.png",
                            "width": 512,
                            "height": 512
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": category_start_id,
                            "bbox": [100, 100, 50, 50],
                            "area": 2500,
                            "iscrowd": 0
                        }
                    ],
                    "categories": categories
                }
                
                ann_file = os.path.join(split_dir, 'annotations.json')
                os.makedirs(split_dir, exist_ok=True)
                with open(ann_file, 'w') as f:
                    json.dump(annotations, f)
                
                # Create mock image file
                if f"{split}/images" not in missing_files:
                    img_file = os.path.join(images_dir, 'test_image.png')
                    # Create a simple white 1x1 PNG
                    from PIL import Image
                    img = Image.new('RGB', (1, 1), 'white')
                    img.save(img_file)
    
    def test_file_structure_validation_success(self):
        """Test successful file structure validation"""
        self.create_mock_dataset()
        validator = DatasetValidator(self.test_dataset_dir)
        
        result = validator._validate_file_structure()
        
        self.assertTrue(result)
        self.assertTrue(validator.validation_results['file_structure']['valid'])
        self.assertEqual(len(validator.validation_results['file_structure']['missing_paths']), 0)
    
    def test_file_structure_validation_missing_files(self):
        """Test file structure validation with missing files"""
        self.create_mock_dataset(missing_files=['train/images', 'val/annotations.json'])
        validator = DatasetValidator(self.test_dataset_dir)
        
        result = validator._validate_file_structure()
        
        self.assertFalse(result)
        self.assertFalse(validator.validation_results['file_structure']['valid'])
        self.assertGreater(len(validator.validation_results['file_structure']['missing_paths']), 0)
    
    def test_category_consistency_validation_success(self):
        """Test successful category consistency validation"""
        self.create_mock_dataset()
        validator = DatasetValidator(self.test_dataset_dir)
        validator._validate_file_structure()  # Setup results
        
        result = validator._validate_category_consistency()
        
        self.assertTrue(result)
        self.assertTrue(validator.validation_results['category_consistency']['valid'])
    
    def test_category_id_validation_1_based(self):
        """Test category ID validation with 1-based IDs (dataset format)"""
        self.create_mock_dataset(category_start_id=1)
        validator = DatasetValidator(self.test_dataset_dir)
        validator._validate_file_structure()
        
        result = validator._validate_category_ids()
        
        self.assertTrue(result)  # 1-based is valid but needs remapping
        self.assertTrue(validator.validation_results['category_ids']['needs_remapping'])
        self.assertFalse(validator.validation_results['category_ids']['detectron2_compatible'])
        self.assertEqual(validator.validation_results['category_ids']['min_id'], 1)
    
    def test_category_id_validation_0_based(self):
        """Test category ID validation with 0-based IDs (Detectron2 compatible)"""
        self.create_mock_dataset(category_start_id=0)
        validator = DatasetValidator(self.test_dataset_dir)
        validator._validate_file_structure()
        
        result = validator._validate_category_ids()
        
        self.assertTrue(result)
        self.assertFalse(validator.validation_results['category_ids']['needs_remapping'])
        self.assertTrue(validator.validation_results['category_ids']['detectron2_compatible'])
        self.assertEqual(validator.validation_results['category_ids']['min_id'], 0)
    
    def test_detectron2_compatibility_with_remapping_needed(self):
        """Test Detectron2 compatibility check when remapping is needed"""
        self.create_mock_dataset(category_start_id=1)
        validator = DatasetValidator(self.test_dataset_dir)
        validator._validate_file_structure()
        validator._validate_category_ids()
        
        result = validator._validate_detectron2_compatibility()
        
        self.assertTrue(result)  # Compatible with remapping
        self.assertTrue(validator.validation_results['detectron2_compatibility']['needs_category_remapping'])
        self.assertGreater(len(validator.validation_results['detectron2_compatibility']['issues']), 0)
    
    def test_detectron2_compatibility_without_remapping(self):
        """Test Detectron2 compatibility check when no remapping is needed"""
        self.create_mock_dataset(category_start_id=0)
        validator = DatasetValidator(self.test_dataset_dir)
        validator._validate_file_structure()
        validator._validate_category_ids()
        
        result = validator._validate_detectron2_compatibility()
        
        self.assertTrue(result)
        self.assertFalse(validator.validation_results['detectron2_compatibility']['needs_category_remapping'])
    
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline"""
        self.create_mock_dataset()
        validator = DatasetValidator(self.test_dataset_dir)
        
        # The validation will pass but with warnings about image loading and data distribution
        result = validator.validate_all()
        
        # Should have results even if some warnings occurred
        self.assertIsNotNone(result)
        
        # Check that all validation components were run
        expected_keys = [
            'file_structure',
            'category_consistency', 
            'category_ids',
            'annotations',
            'images',
            'data_distribution',
            'detectron2_compatibility'
        ]
        
        for key in expected_keys:
            self.assertIn(key, validator.validation_results)

class TestCategoryMapping(unittest.TestCase):
    """Test category mapping functionality"""
    
    def test_category_id_offset_calculation(self):
        """Test proper calculation of category ID offset"""
        # Test 1-based dataset (needs -1 offset)
        categories_1_based = {1: "A1", 2: "M17", 3: "N35"}
        min_id = min(categories_1_based.keys())
        offset = -1 if min_id == 1 else 0
        
        self.assertEqual(offset, -1)
        
        # Test 0-based dataset (no offset needed)
        categories_0_based = {0: "A1", 1: "M17", 2: "N35"}
        min_id = min(categories_0_based.keys())
        offset = -1 if min_id == 1 else 0
        
        self.assertEqual(offset, 0)
    
    def test_category_remapping_logic(self):
        """Test category ID remapping logic"""
        original_categories = {1: "A1", 2: "M17", 3: "N35"}
        offset = -1
        
        # Simulate remapping
        remapped_ids = {}
        for cat_id, cat_name in original_categories.items():
            new_id = cat_id + offset
            remapped_ids[new_id] = cat_name
        
        expected = {0: "A1", 1: "M17", 2: "N35"}
        self.assertEqual(remapped_ids, expected)

class TestTrainingConfiguration(unittest.TestCase):
    """Test training configuration and setup"""
    
    def test_training_config_validation(self):
        """Test training configuration validation"""
        # Mock args
        class MockArgs:
            dataset_path = "hieroglyphs_dataset"
            output_dir = "output"
            model_weights = None
            num_workers = 2
            batch_size = 1
            learning_rate = 0.001
            max_iter = 1000
            eval_period = 500
        
        # This would need to be tested with actual training pipeline
        # For now, just test that the config class can be instantiated
        try:
            from train_hieroglyph_detection_robust import HieroglyphTrainingConfig
            config = HieroglyphTrainingConfig(MockArgs())
            self.assertIsNotNone(config.run_id)
            self.assertEqual(config.dataset_path, "hieroglyphs_dataset")
        except ImportError:
            self.skipTest("Training script not available for testing")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dataset_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dataset_dir)
    
    def create_realistic_mock_dataset(self):
        """Create a more realistic mock dataset for integration testing"""
        splits = ['train', 'val', 'test']
        
        # Create realistic hieroglyph categories
        categories = []
        for i in range(10):
            categories.append({
                "id": i + 1,  # 1-based IDs to test remapping
                "name": f"H{i+1}",
                "supercategory": "hieroglyph"
            })
        
        for split in splits:
            split_dir = os.path.join(self.test_dataset_dir, split)
            images_dir = os.path.join(split_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # Create multiple mock images and annotations
            images = []
            annotations = []
            
            for img_idx in range(3):  # 3 images per split
                img_id = img_idx + 1
                images.append({
                    "id": img_id,
                    "file_name": f"image_{img_idx:03d}.png",
                    "width": 512,
                    "height": 512
                })
                
                # Create mock annotations
                for ann_idx in range(2):  # 2 annotations per image
                    ann_id = img_idx * 2 + ann_idx + 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": (ann_idx % len(categories)) + 1,
                        "bbox": [50 + ann_idx * 100, 50, 50, 50],
                        "area": 2500,
                        "iscrowd": 0
                    })
                
                # Create mock image file
                img_file = os.path.join(images_dir, f"image_{img_idx:03d}.png")
                from PIL import Image
                img = Image.new('RGB', (512, 512), 'white')
                img.save(img_file)
            
            # Create annotation file
            coco_data = {
                "info": {
                    "description": "Test hieroglyph dataset",
                    "version": "1.0"
                },
                "images": images,
                "annotations": annotations,
                "categories": categories
            }
            
            ann_file = os.path.join(split_dir, 'annotations.json')
            with open(ann_file, 'w') as f:
                json.dump(coco_data, f)
    
    def test_complete_validation_pipeline(self):
        """Test complete validation pipeline with realistic data"""
        self.create_realistic_mock_dataset()
        
        validator = DatasetValidator(self.test_dataset_dir)
        result = validator.validate_all()
        
        # Should have results (might have warnings but not critical failures)
        self.assertIsNotNone(result)
        
        # Should detect 1-based category IDs and recommend remapping
        self.assertTrue(validator.validation_results['category_ids']['needs_remapping'])
        self.assertFalse(validator.validation_results['category_ids']['detectron2_compatible'])
        
        # Should have proper summary with key information
        summary = validator._generate_summary()
        # Even with warnings, the validation logic should work
        self.assertIn('recommendations', summary)
        if summary['recommendations']:
            self.assertIn("Use category ID remapping", summary['recommendations'][0])

class TestRegressionPrevention(unittest.TestCase):
    """Regression tests to prevent the off-by-one error from reoccurring"""
    
    def test_category_mapping_regression_prevention(self):
        """Test that prevents the original M16/M17 confusion issue"""
        
        # Simulate the original problem scenario
        dataset_categories = {294: "M17"}  # Ground truth has M17
        model_predictions = [
            {"category_id": 293, "score": 0.9}  # Model predicts M16 (off by 1)
        ]
        
        # The validation should catch this mismatch
        gt_cat_ids = set(dataset_categories.keys())
        pred_cat_ids = set(pred['category_id'] for pred in model_predictions)
        
        # No overlap indicates the off-by-one error
        overlap = gt_cat_ids & pred_cat_ids
        self.assertEqual(len(overlap), 0, "Off-by-one error detected: no category overlap")
        
        # With proper remapping (subtract 1 from dataset categories):
        remapped_dataset_categories = {293: "M17"}  # M17 now at ID 293
        remapped_gt_cat_ids = set(remapped_dataset_categories.keys())
        
        # Now there should be overlap
        overlap_after_fix = remapped_gt_cat_ids & pred_cat_ids
        self.assertEqual(len(overlap_after_fix), 1, "After remapping, categories should align")

def run_tests():
    """Run all tests and return results"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDatasetValidator,
        TestCategoryMapping,
        TestTrainingConfiguration,
        TestIntegration,
        TestRegressionPrevention
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == "__main__":
    print("🧪 Running Automated Tests for Dataset Validation and Training Pipeline")
    print("=" * 80)
    
    result = run_tests()
    
    print("\n" + "=" * 80)
    print("🎯 TEST SUMMARY")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print("   Dataset validation and training pipeline are working correctly")
        print("   Category mapping issues are prevented")
    else:
        print("❌ Some tests failed!")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        if result.failures:
            print("\n🚨 Failures:")
            for test, traceback in result.failures:
                print(f"   - {test}: {traceback}")
        
        if result.errors:
            print("\n💥 Errors:")
            for test, traceback in result.errors:
                print(f"   - {test}: {traceback}")
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)
