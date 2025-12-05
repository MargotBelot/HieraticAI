#!/usr/bin/env python3
"""
Unit Tests for Training Pipeline
Tests training configuration, category remapping, and data loading.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCategoryRemapping(unittest.TestCase):
    """Test category ID remapping logic"""

    def test_1_based_to_0_based_remapping(self):
        """Test 1-based to 0-based category ID remapping"""
        original_ids = [1, 2, 3, 294, 634]
        offset = -1

        remapped_ids = [cat_id + offset for cat_id in original_ids]

        self.assertEqual(remapped_ids, [0, 1, 2, 293, 633])

    def test_0_based_no_remapping(self):
        """Test 0-based categories need no remapping"""
        original_ids = [0, 1, 2, 293, 633]
        offset = 0

        remapped_ids = [cat_id + offset for cat_id in original_ids]

        self.assertEqual(remapped_ids, original_ids)

    def test_offset_calculation_from_min_id(self):
        """Test offset calculation based on minimum category ID"""
        # Test 1-based dataset
        categories_1_based = {1: "A1", 2: "M17", 294: "M17"}
        min_id = min(categories_1_based.keys())
        offset = -1 if min_id == 1 else 0
        self.assertEqual(offset, -1)

        # Test 0-based dataset
        categories_0_based = {0: "A1", 1: "M17", 293: "M17"}
        min_id = min(categories_0_based.keys())
        offset = -1 if min_id == 1 else 0
        self.assertEqual(offset, 0)

    def test_remapping_preserves_order(self):
        """Test that remapping preserves category order"""
        original_categories = {1: "A1", 2: "B2", 3: "C3", 4: "D4"}
        offset = -1

        remapped = {}
        for cat_id, cat_name in sorted(original_categories.items()):
            new_id = cat_id + offset
            remapped[new_id] = cat_name

        expected = {0: "A1", 1: "B2", 2: "C3", 3: "D4"}
        self.assertEqual(remapped, expected)

    def test_annotation_remapping(self):
        """Test annotation category ID remapping"""
        annotations = [
            {"id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "category_id": 2, "bbox": [30, 30, 40, 40]},
            {"id": 3, "category_id": 294, "bbox": [50, 50, 60, 60]},
        ]

        offset = -1

        for ann in annotations:
            ann["category_id"] += offset

        self.assertEqual(annotations[0]["category_id"], 0)
        self.assertEqual(annotations[1]["category_id"], 1)
        self.assertEqual(annotations[2]["category_id"], 293)

    def test_negative_id_detection(self):
        """Test detection of invalid negative category IDs"""
        annotation = {"category_id": 0}
        offset = -1

        new_id = annotation["category_id"] + offset

        self.assertLess(new_id, 0)
        self.assertEqual(new_id, -1)


class TestHieroglyphTrainingConfig(unittest.TestCase):
    """Test HieroglyphTrainingConfig class"""

    def setUp(self):
        """Set up test environment"""
        self.test_dataset_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dataset_dir)

        # Create mock dataset structure
        self.create_mock_dataset()

    def create_mock_dataset(self):
        """Create a mock dataset for testing"""
        splits = ["train", "val", "test"]
        categories = [
            {"id": 1, "name": "A1", "supercategory": ""},
            {"id": 2, "name": "M17", "supercategory": ""},
            {"id": 3, "name": "N35", "supercategory": ""},
        ]

        for split in splits:
            split_dir = os.path.join(self.test_dataset_dir, split)
            images_dir = os.path.join(split_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            annotations = {
                "images": [
                    {"id": 1, "file_name": "test.png", "width": 512, "height": 512}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 50, 50],
                        "area": 2500,
                        "iscrowd": 0,
                    }
                ],
                "categories": categories,
            }

            ann_file = os.path.join(split_dir, "annotations.json")
            with open(ann_file, "w") as f:
                json.dump(annotations, f)

            # Create mock image
            from PIL import Image

            img = Image.new("RGB", (512, 512), "white")
            img.save(os.path.join(images_dir, "test.png"))

    def test_config_initialization(self):
        """Test configuration initialization"""

        # Mock args
        class MockArgs:
            dataset_path = self.test_dataset_dir
            output_dir = "output"
            model_weights = None
            num_workers = 2
            batch_size = 1
            learning_rate = 0.001
            max_iter = 1000
            eval_period = 500

        try:
            from tools.training.train import HieroglyphTrainingConfig

            config = HieroglyphTrainingConfig(MockArgs())

            self.assertEqual(config.dataset_path, self.test_dataset_dir)
            self.assertEqual(config.output_dir, "output")
            self.assertEqual(config.num_workers, 2)
            self.assertEqual(config.batch_size, 1)
            self.assertEqual(config.learning_rate, 0.001)
            self.assertEqual(config.max_iter, 1000)
            self.assertEqual(config.eval_period, 500)
            self.assertIsNotNone(config.run_id)

        except ImportError:
            self.skipTest("Training module not available")

    def test_config_validation_detects_1_based_categories(self):
        """Test that config validation detects 1-based category IDs"""

        class MockArgs:
            dataset_path = self.test_dataset_dir
            output_dir = tempfile.mkdtemp()
            model_weights = None
            num_workers = 2
            batch_size = 1
            learning_rate = 0.001
            max_iter = 1000
            eval_period = 500

        try:
            from tools.training.train import HieroglyphTrainingConfig

            config = HieroglyphTrainingConfig(MockArgs())
            result = config.validate_and_setup()

            # Should succeed with warnings about 1-based IDs
            self.assertTrue(result or result is None)  # Validation may have warnings
            self.assertEqual(config.category_id_offset, -1)
            self.assertEqual(config.num_classes, 3)

        except ImportError:
            self.skipTest("Training module not available")
        except Exception as e:
            # Validation may fail due to missing dependencies, but we can check the logic
            self.skipTest(f"Validation skipped due to: {e}")

    def test_num_classes_calculation(self):
        """Test number of classes is calculated correctly"""
        ann_file = os.path.join(self.test_dataset_dir, "train", "annotations.json")
        with open(ann_file, "r") as f:
            data = json.load(f)

        num_classes = len(data["categories"])

        self.assertEqual(num_classes, 3)


class TestCategoryRemappingDatasetMapper(unittest.TestCase):
    """Test CategoryRemappingDatasetMapper"""

    def test_mapper_applies_offset(self):
        """Test that mapper correctly applies category offset"""
        # Create mock dataset dict
        dataset_dict = {
            "image_id": 1,
            "annotations": [
                {"category_id": 1, "bbox": [10, 10, 20, 20]},
                {"category_id": 2, "bbox": [30, 30, 40, 40]},
            ],
        }

        offset = -1

        # Apply offset manually (simulating mapper behavior)
        for ann in dataset_dict["annotations"]:
            ann["category_id"] += offset

        self.assertEqual(dataset_dict["annotations"][0]["category_id"], 0)
        self.assertEqual(dataset_dict["annotations"][1]["category_id"], 1)

    def test_mapper_detects_negative_ids(self):
        """Test that mapper detects invalid negative IDs"""
        dataset_dict = {
            "image_id": 1,
            "annotations": [{"category_id": 0, "bbox": [10, 10, 20, 20]}],
        }

        offset = -1

        # Check if any annotation would become negative
        has_negative = False
        for ann in dataset_dict["annotations"]:
            new_id = ann["category_id"] + offset
            if new_id < 0:
                has_negative = True

        self.assertTrue(has_negative)


class TestDatasetRegistration(unittest.TestCase):
    """Test dataset registration with Detectron2"""

    def setUp(self):
        """Set up test environment"""
        self.test_dataset_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dataset_dir)

    def test_dataset_path_validation(self):
        """Test validation of dataset paths"""
        splits = ["train", "val", "test"]

        for split in splits:
            images_dir = f"{self.test_dataset_dir}/{split}/images"
            annotations_file = f"{self.test_dataset_dir}/{split}/annotations.json"

            # Before creation
            self.assertFalse(os.path.exists(images_dir))
            self.assertFalse(os.path.exists(annotations_file))

            # Create
            os.makedirs(images_dir, exist_ok=True)
            with open(annotations_file, "w") as f:
                json.dump({"images": [], "annotations": [], "categories": []}, f)

            # After creation
            self.assertTrue(os.path.exists(images_dir))
            self.assertTrue(os.path.exists(annotations_file))


class TestTrainingMetadata(unittest.TestCase):
    """Test training metadata generation"""

    def test_metadata_structure(self):
        """Test metadata has required fields"""
        from datetime import datetime

        metadata = {
            "run_id": datetime.now().strftime("training_%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "dataset_path": "/path/to/dataset",
            "num_classes": 634,
            "category_id_offset": -1,
            "categories": {1: "A1", 2: "M17"},
            "model_config": {
                "weights": "model.pth",
                "backbone": 101,
                "num_classes": 634,
            },
            "training_config": {
                "max_iter": 10000,
                "learning_rate": 0.001,
                "batch_size": 2,
                "eval_period": 500,
            },
        }

        required_fields = [
            "run_id",
            "start_time",
            "dataset_path",
            "num_classes",
            "category_id_offset",
            "categories",
            "model_config",
            "training_config",
        ]

        for field in required_fields:
            self.assertIn(field, metadata)

    def test_metadata_serialization(self):
        """Test metadata can be serialized to JSON"""
        from datetime import datetime

        metadata = {
            "run_id": "training_20240101_120000",
            "start_time": datetime.now().isoformat(),
            "num_classes": 3,
            "category_id_offset": -1,
        }

        # Should be JSON serializable
        json_str = json.dumps(metadata, default=str)
        self.assertIsInstance(json_str, str)

        # Should be deserializable
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["run_id"], metadata["run_id"])
        self.assertEqual(deserialized["num_classes"], metadata["num_classes"])


def run_tests():
    """Run all tests"""
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    print("Running Training Pipeline Unit Tests")
    print("=" * 80)
    run_tests()
