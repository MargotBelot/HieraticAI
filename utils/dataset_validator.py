#!/usr/bin/env python3
"""
Dataset Validator for HieraticAI
Validates dataset structure, category mappings, and Detectron2 compatibility.
Prevents category ID mismatches and other training issues.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """
    Comprehensive dataset validator for COCO-format hieroglyph datasets.
    Ensures proper structure, category consistency, and Detectron2 compatibility.
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the dataset validator.

        Args:
            dataset_path: Path to the dataset directory containing train/val/test splits
        """
        self.dataset_path = Path(dataset_path)
        self.validation_results = {
            "file_structure": {},
            "category_consistency": {},
            "category_ids": {},
            "annotations": {},
            "images": {},
            "data_distribution": {},
            "detectron2_compatibility": {},
        }

        self.splits = ["train", "val", "test"]
        self.category_data = {}  # Store category information from each split

    def validate_all(self) -> bool:
        """
        Run all validation checks.

        Returns:
            bool: True if all validations pass (or pass with warnings), False for critical failures
        """
        logger.info("=" * 60)
        logger.info("Starting Dataset Validation")
        logger.info("=" * 60)

        # Run all validation steps
        steps = [
            ("File Structure", self._validate_file_structure),
            ("Category Consistency", self._validate_category_consistency),
            ("Category IDs", self._validate_category_ids),
            ("Annotations", self._validate_annotations),
            ("Images", self._validate_images),
            ("Data Distribution", self._validate_data_distribution),
            ("Detectron2 Compatibility", self._validate_detectron2_compatibility),
        ]

        all_passed = True

        for step_name, step_func in steps:
            logger.info(f"\n{step_name}...")
            try:
                result = step_func()
                if not result:
                    logger.warning(
                        f"  [WARNING]  {step_name} validation completed with warnings"
                    )
                else:
                    logger.info(f"  [PASS] {step_name} validation passed")
            except Exception as e:
                logger.error(f"  [FAIL] {step_name} validation failed: {e}")
                all_passed = False

        # Generate summary
        summary = self._generate_summary()
        self._print_summary(summary)

        return all_passed

    def _validate_file_structure(self) -> bool:
        """Validate that all required files and directories exist."""
        required_paths = []
        missing_paths = []

        for split in self.splits:
            split_dir = self.dataset_path / split
            images_dir = split_dir / "images"
            annotations_file = split_dir / "annotations.json"

            required_paths.extend(
                [
                    (split_dir, f"{split}/ directory"),
                    (images_dir, f"{split}/images/ directory"),
                    (annotations_file, f"{split}/annotations.json file"),
                ]
            )

        for path, description in required_paths:
            if not path.exists():
                missing_paths.append(description)
                logger.error(f"  Missing: {description}")

        self.validation_results["file_structure"] = {
            "valid": len(missing_paths) == 0,
            "missing_paths": missing_paths,
            "checked_paths": len(required_paths),
        }

        return len(missing_paths) == 0

    def _validate_category_consistency(self) -> bool:
        """Validate that categories are consistent across splits."""
        categories_by_split = {}

        for split in self.splits:
            annotations_file = self.dataset_path / split / "annotations.json"

            if not annotations_file.exists():
                continue

            try:
                with open(annotations_file, "r") as f:
                    data = json.load(f)

                categories = {
                    cat["id"]: cat["name"] for cat in data.get("categories", [])
                }
                categories_by_split[split] = categories
                self.category_data[split] = data.get("categories", [])

            except Exception as e:
                logger.error(f"  Error reading {split} annotations: {e}")
                return False

        # Check consistency across splits
        if not categories_by_split:
            logger.error("  No valid category data found in any split")
            return False

        # Use the first split as reference
        reference_split = list(categories_by_split.keys())[0]
        reference_categories = categories_by_split[reference_split]

        inconsistencies = []
        for split, categories in categories_by_split.items():
            if split == reference_split:
                continue

            if categories != reference_categories:
                inconsistencies.append(
                    f"{split} has different categories than {reference_split}"
                )
                logger.warning(f"  {split} categories differ from {reference_split}")

        self.validation_results["category_consistency"] = {
            "valid": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "num_categories": len(reference_categories),
            "reference_split": reference_split,
        }

        return len(inconsistencies) == 0

    def _validate_category_ids(self) -> bool:
        """Validate category ID ranges and detect if remapping is needed."""
        if not self.category_data:
            logger.error("  No category data available for validation")
            return False

        # Get categories from first available split
        first_split = list(self.category_data.keys())[0]
        categories = self.category_data[first_split]

        if not categories:
            logger.error("  No categories found")
            return False

        category_ids = [cat["id"] for cat in categories]
        min_id = min(category_ids)
        max_id = max(category_ids)
        num_categories = len(category_ids)

        # Check for 1-based vs 0-based indexing
        needs_remapping = False
        detectron2_compatible = False

        if min_id == 1:
            logger.warning(
                "  [WARNING]  Categories use 1-based indexing (starting at 1)"
            )
            logger.warning(
                "  [WARNING]  Detectron2 expects 0-based indexing (starting at 0)"
            )
            logger.warning(
                "  [WARNING]  Category ID remapping is required during training"
            )
            needs_remapping = True
            detectron2_compatible = False
        elif min_id == 0:
            logger.info(
                "  [PASS] Categories use 0-based indexing (Detectron2 compatible)"
            )
            needs_remapping = False
            detectron2_compatible = True
        else:
            logger.error(f"  Unusual category ID start: {min_id} (expected 0 or 1)")
            return False

        # Check for gaps in IDs
        expected_ids = set(range(min_id, min_id + num_categories))
        actual_ids = set(category_ids)
        missing_ids = expected_ids - actual_ids

        if missing_ids:
            logger.warning(f"  [WARNING]  Gaps in category IDs: {sorted(missing_ids)}")

        self.validation_results["category_ids"] = {
            "valid": True,
            "min_id": min_id,
            "max_id": max_id,
            "num_categories": num_categories,
            "needs_remapping": needs_remapping,
            "detectron2_compatible": detectron2_compatible,
            "has_gaps": len(missing_ids) > 0,
            "missing_ids": list(missing_ids),
        }

        return True

    def _validate_annotations(self) -> bool:
        """Validate annotation structure and content."""
        total_annotations = 0
        total_images = 0
        issues = []

        for split in self.splits:
            annotations_file = self.dataset_path / split / "annotations.json"

            if not annotations_file.exists():
                continue

            try:
                with open(annotations_file, "r") as f:
                    data = json.load(f)

                annotations = data.get("annotations", [])
                images = data.get("images", [])
                categories = data.get("categories", [])

                total_annotations += len(annotations)
                total_images += len(images)

                # Validate annotation structure
                category_ids = set(cat["id"] for cat in categories)

                for ann in annotations:
                    # Check required fields
                    required_fields = ["id", "image_id", "category_id", "bbox"]
                    for field in required_fields:
                        if field not in ann:
                            issues.append(
                                f"{split}: Annotation {ann.get('id', 'unknown')} missing '{field}'"
                            )

                    # Check category_id validity
                    if ann.get("category_id") not in category_ids:
                        issues.append(
                            f"{split}: Annotation {ann['id']} has invalid category_id {ann['category_id']}"
                        )

            except Exception as e:
                issues.append(f"Error reading {split} annotations: {e}")

        self.validation_results["annotations"] = {
            "valid": len(issues) == 0,
            "total_annotations": total_annotations,
            "total_images": total_images,
            "issues": issues,
        }

        if issues:
            for issue in issues[:5]:  # Show first 5 issues
                logger.warning(f"  {issue}")
            if len(issues) > 5:
                logger.warning(f"  ... and {len(issues) - 5} more issues")

        return len(issues) == 0

    def _validate_images(self) -> bool:
        """Validate that image files exist and match annotations."""
        issues = []
        total_image_files = 0

        for split in self.splits:
            annotations_file = self.dataset_path / split / "annotations.json"
            images_dir = self.dataset_path / split / "images"

            if not annotations_file.exists() or not images_dir.exists():
                continue

            try:
                with open(annotations_file, "r") as f:
                    data = json.load(f)

                images = data.get("images", [])

                for img_info in images:
                    img_path = images_dir / img_info["file_name"]

                    if not img_path.exists():
                        issues.append(
                            f"{split}: Image file missing: {img_info['file_name']}"
                        )
                    else:
                        total_image_files += 1

            except Exception as e:
                issues.append(f"Error validating {split} images: {e}")

        self.validation_results["images"] = {
            "valid": len(issues) == 0,
            "total_image_files": total_image_files,
            "issues": issues,
        }

        if issues:
            for issue in issues[:5]:
                logger.warning(f"  {issue}")
            if len(issues) > 5:
                logger.warning(f"  ... and {len(issues) - 5} more issues")

        return len(issues) == 0

    def _validate_data_distribution(self) -> bool:
        """Validate data distribution across splits."""
        distribution = {}

        for split in self.splits:
            annotations_file = self.dataset_path / split / "annotations.json"

            if not annotations_file.exists():
                continue

            try:
                with open(annotations_file, "r") as f:
                    data = json.load(f)

                num_images = len(data.get("images", []))
                num_annotations = len(data.get("annotations", []))

                distribution[split] = {
                    "images": num_images,
                    "annotations": num_annotations,
                    "avg_annotations_per_image": (
                        num_annotations / num_images if num_images > 0 else 0
                    ),
                }

            except Exception as e:
                logger.warning(f"  Error analyzing {split} distribution: {e}")

        self.validation_results["data_distribution"] = distribution

        # Log distribution
        for split, stats in distribution.items():
            logger.info(
                f"  {split:5s}: {stats['images']:4d} images, {stats['annotations']:4d} annotations "
                f"({stats['avg_annotations_per_image']:.1f} ann/img)"
            )

        return True

    def _validate_detectron2_compatibility(self) -> bool:
        """Validate Detectron2-specific requirements."""
        issues = []
        needs_category_remapping = self.validation_results.get("category_ids", {}).get(
            "needs_remapping", False
        )

        if needs_category_remapping:
            issues.append(
                "Category IDs need remapping from 1-based to 0-based for Detectron2"
            )
            logger.warning(
                "  [WARNING]  Use category ID remapping during training (CategoryRemappingDatasetMapper)"
            )

        # Check for other potential issues
        category_result = self.validation_results.get("category_ids", {})
        if category_result.get("has_gaps"):
            issues.append("Category IDs have gaps which may cause issues")

        self.validation_results["detectron2_compatibility"] = {
            "compatible": True,  # Can be made compatible with remapping
            "needs_category_remapping": needs_category_remapping,
            "issues": issues,
        }

        return True

    def _generate_summary(self) -> Dict:
        """Generate validation summary."""
        summary = {
            "overall_status": "PASS",
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # Check for critical issues
        if not self.validation_results["file_structure"]["valid"]:
            summary["overall_status"] = "FAIL"
            summary["critical_issues"].append("Missing required files or directories")

        if not self.validation_results["category_consistency"]["valid"]:
            summary["warnings"].append("Category inconsistencies across splits")

        # Add recommendations
        if self.validation_results["category_ids"].get("needs_remapping"):
            summary["recommendations"].append(
                "Use category ID remapping during training (offset=-1 to convert 1-based to 0-based)"
            )

        if self.validation_results["detectron2_compatibility"].get(
            "needs_category_remapping"
        ):
            summary["recommendations"].append(
                "Enable CategoryRemappingDatasetMapper in training script"
            )

        return summary

    def _print_summary(self, summary: Dict):
        """Print validation summary."""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        status_emoji = "[PASS]" if summary["overall_status"] == "PASS" else "[FAIL]"
        logger.info(f"Overall Status: {status_emoji} {summary['overall_status']}")

        if summary["critical_issues"]:
            logger.error("\nCritical Issues:")
            for issue in summary["critical_issues"]:
                logger.error(f"  [FAIL] {issue}")

        if summary["warnings"]:
            logger.warning("\nWarnings:")
            for warning in summary["warnings"]:
                logger.warning(f"  [WARNING]  {warning}")

        if summary["recommendations"]:
            logger.info("\nRecommendations:")
            for rec in summary["recommendations"]:
                logger.info(f"  -> {rec}")

        logger.info("\n" + "=" * 60)

    def get_category_mapping_info(self) -> Dict:
        """
        Get category mapping information for training.

        Returns:
            Dict with category mapping details
        """
        return {
            "needs_remapping": self.validation_results.get("category_ids", {}).get(
                "needs_remapping", False
            ),
            "offset": (
                -1
                if self.validation_results.get("category_ids", {}).get(
                    "needs_remapping", False
                )
                else 0
            ),
            "num_categories": self.validation_results.get("category_ids", {}).get(
                "num_categories", 0
            ),
        }
