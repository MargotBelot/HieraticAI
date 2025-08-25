#!/usr/bin/env python3
"""
Dataset Consistency Verification
Critical script to identify the exact cause of category mapping errors.
"""

import json
import os
from collections import Counter, defaultdict
import random

def load_annotation_data():
    """Load all annotation files and extract category mappings"""
    splits = ['train', 'val', 'test']
    data = {}
    
    for split in splits:
        ann_file = f'hieroglyphs_dataset/{split}/annotations.json'
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                data[split] = json.load(f)
        else:
            print(f"Missing annotation file: {ann_file}")
            data[split] = None
    
    return data

def verify_category_consistency(data):
    """Check if category mappings are identical across splits"""
    print("Verifying category consistency across splits...")
    
    splits_with_data = {k: v for k, v in data.items() if v is not None}
    
    if len(splits_with_data) < 2:
        print("Need at least 2 splits to verify consistency")
        return False
    
    # Extract category mappings
    category_mappings = {}
    for split, split_data in splits_with_data.items():
        categories = {cat['id']: cat['name'] for cat in split_data['categories']}
        category_mappings[split] = categories
        print(f"{split}: {len(categories)} categories")
    
    # Check consistency
    base_split = list(splits_with_data.keys())[0]
    base_mapping = category_mappings[base_split]
    
    all_consistent = True
    for split, mapping in category_mappings.items():
        if split == base_split:
            continue
            
        if mapping != base_mapping:
            print(f"Category mapping differs between {base_split} and {split}")
            
            # Find specific differences
            base_set = set(base_mapping.items())
            split_set = set(mapping.items())
            
            only_base = base_set - split_set
            only_split = split_set - base_set
            
            if only_base:
                print(f"Only in {base_split}: {len(only_base)} categories")
                for cat_id, name in sorted(only_base):
                    print(f"ID {cat_id}: {name}")
            
            if only_split:
                print(f"Only in {split}: {len(only_split)} categories")
                for cat_id, name in sorted(only_split):
                    print(f"ID {cat_id}: {name}")
            
            all_consistent = False
        else:
            print(f"{split} categories match {base_split}")
    
    if all_consistent:
        print("All splits have identical category mappings")
        return True, category_mappings[base_split]
    else:
        print("Category mappings are inconsistent between splits")
        return False, None

def check_annotation_distribution(data):
    """Analyze annotation distribution across splits"""
    print("\n Analyzing annotation distribution...")
    
    for split, split_data in data.items():
        if split_data is None:
            continue
            
        annotations = split_data['annotations']
        images = split_data['images']
        categories = {cat['id']: cat['name'] for cat in split_data['categories']}
        
        print(f"\n{split.upper()} Split:")
        print(f"Images: {len(images)}")
        print(f"Annotations: {len(annotations)}")
        print(f"Categories: {len(categories)}")
        
        # Count annotations per category
        cat_counts = Counter(ann['category_id'] for ann in annotations)
        used_categories = len(cat_counts)
        
        print(f"Used categories: {used_categories}/{len(categories)} ({used_categories/len(categories)*100:.1f}%)")
        
        # Show top/bottom categories
        sorted_cats = cat_counts.most_common()
        
        print(f"Top 5 categories:")
        for cat_id, count in sorted_cats[:5]:
            cat_name = categories.get(cat_id, f"Unknown_{cat_id}")
            print(f"{cat_name} (ID {cat_id}): {count}")
        
        print(f"Bottom 5 categories (with annotations):")
        for cat_id, count in sorted_cats[-5:]:
            cat_name = categories.get(cat_id, f"Unknown_{cat_id}")  
            print(f"{cat_name} (ID {cat_id}): {count}")
        
        # Categories with no annotations
        unused_cats = set(categories.keys()) - set(cat_counts.keys())
        print(f"Unused categories: {len(unused_cats)}")

def sample_training_images(data):
    """Sample random training images for manual verification"""
    print("\nSampling training images for manual verification...")
    
    if 'train' not in data or data['train'] is None:
        print("No training data available")
        return
    
    train_data = data['train']
    annotations = train_data['annotations']
    images = {img['id']: img for img in train_data['images']}
    categories = {cat['id']: cat['name'] for cat in train_data['categories']}
    
    # Group annotations by image
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann['image_id']].append(ann)
    
    # Sample 10 random images with annotations
    image_ids = list(ann_by_image.keys())
    sample_size = min(10, len(image_ids))
    sample_images = random.sample(image_ids, sample_size)
    
    print(f" Manual verification sample ({sample_size} images):")
    print("Please manually verify these training images have correct labels:")
    
    verification_list = []
    
    for i, img_id in enumerate(sample_images):
        img_info = images[img_id]
        img_anns = ann_by_image[img_id]
        
        print(f"\n   {i+1}. IMAGE: hieroglyphs_dataset/train/images/{img_info['file_name']}")
        print(f" Dimensions: {img_info['width']} x {img_info['height']}")
        print(f" Annotations: {len(img_anns)}")
        
        # Show all annotations for this image
        for j, ann in enumerate(img_anns[:5]):  # Limit to first 5
            cat_name = categories.get(ann['category_id'], f"Unknown_{ann['category_id']}")
            bbox = ann['bbox']  # [x, y, width, height]
            print(f"{j+1}. {cat_name} (ID {ann['category_id']}) at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        
        if len(img_anns) > 5:
            print(f"... and {len(img_anns)-5} more annotations")
        
        verification_list.append({
            'image_path': f"hieroglyphs_dataset/train/images/{img_info['file_name']}",
            'annotations': [
                {
                    'category_id': ann['category_id'],
                    'category_name': categories.get(ann['category_id'], f"Unknown_{ann['category_id']}"),
                    'bbox': ann['bbox']
                } for ann in img_anns
            ]
        })
    
    # Save verification list
    with open('manual_verification_sample.json', 'w') as f:
        json.dump(verification_list, f, indent=2)
    
    print(f"\nVerification sample saved to: manual_verification_sample.json")
    print("Use this file to systematically verify image labels")

def check_category_id_ranges(data):
    """Check for suspicious category ID patterns"""
    print("\nChecking category ID ranges and patterns...")
    
    for split, split_data in data.items():
        if split_data is None:
            continue
        
        categories = {cat['id']: cat['name'] for cat in split_data['categories']}
        cat_ids = list(categories.keys())
        
        print(f"\n{split.upper()} Split category IDs:")
        print(f"Range: {min(cat_ids)} - {max(cat_ids)}")
        print(f"Count: {len(cat_ids)}")
        
        # Check for gaps
        expected_range = set(range(min(cat_ids), max(cat_ids) + 1))
        actual_range = set(cat_ids)
        gaps = expected_range - actual_range
        
        if gaps:
            print(f" Missing IDs: {len(gaps)} gaps in range")
            if len(gaps) <= 20:
                print(f" Missing: {sorted(gaps)}")
            else:
                print(f" Missing: {sorted(gaps)[:10]} ... (and {len(gaps)-10} more)")
        else:
            print(f"No gaps in ID range")
        
        # Check for suspicious patterns
        if 0 in cat_ids:
            print(f" Category ID 0 present (often indicates background/unknown)")
        
        if min(cat_ids) > 1:
            print(f" Category IDs don't start at 0 or 1 (starts at {min(cat_ids)})")
        
        # Look for duplicates
        duplicates = len(cat_ids) - len(set(cat_ids))
        if duplicates > 0:
            print(f"DUPLICATE category IDs detected: {duplicates}")

def compare_with_predictions():
    """Compare dataset categories with actual predictions"""
    print("\nComparing dataset categories with model predictions...")
    
    # Load predictions
    results_file = "output/improved_training_20250822_200344/coco_instances_results.json"
    if not os.path.exists(results_file):
        print("Predictions file not found")
        return
    
    with open(results_file, 'r') as f:
        predictions = json.load(f)
    
    pred_cat_ids = set(pred['category_id'] for pred in predictions)
    
    # Load test categories (ground truth)
    with open('hieroglyphs_dataset/test/annotations.json') as f:
        test_data = json.load(f)
    
    gt_cat_ids = set(ann['category_id'] for ann in test_data['annotations'])
    all_cat_ids = set(cat['id'] for cat in test_data['categories'])
    
    print(f"All categories in dataset: {len(all_cat_ids)}")
    print(f"Categories in test ground truth: {len(gt_cat_ids)}")
    print(f"Categories in predictions: {len(pred_cat_ids)}")
    
    # Analyze overlaps
    correct_predictions = pred_cat_ids & gt_cat_ids
    wrong_predictions = pred_cat_ids - all_cat_ids  # Predicting non-existent categories
    missing_predictions = gt_cat_ids - pred_cat_ids
    
    print(f"Correct category predictions: {len(correct_predictions)}")
    print(f"Wrong category predictions (non-existent): {len(wrong_predictions)}")
    print(f" Missing category predictions: {len(missing_predictions)}")
    
    if wrong_predictions:
        print(f"Non-existent category IDs being predicted: {sorted(wrong_predictions)}")
        if 0 in wrong_predictions:
            print("  Category 0 predictions suggest background/unknown confusion")

def main():
    print(" Dataset Consistency Verification")
    print("=" * 60)
    
    # Load all annotation data
    data = load_annotation_data()
    
    # Verify category consistency
    consistent, base_categories = verify_category_consistency(data)
    
    # Analyze annotation distribution
    check_annotation_distribution(data)
    
    # Check category ID patterns
    check_category_id_ranges(data)
    
    # Sample training images for manual verification
    sample_training_images(data)
    
    # Compare with predictions
    compare_with_predictions()
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if consistent:
        print("Category mappings are consistent across splits")
        print("BUT: Model has systematic prediction errors")
        print("\nNext steps:")
        print("1. Manually verify the sample images listed above")
        print("2. Check training configuration and data loading")
        print("3. Look for off-by-one errors or data corruption")
    else:
        print("CRITICAL: Category mappings are inconsistent between splits")
        print("\nNext steps:")
        print("1. Fix category mapping consistency issues first")
        print("2. Then manually verify training data")
        print("3. Retrain model with corrected mappings")
    
    print(f"\nManual verification sample: manual_verification_sample.json")
    print("Open these images and verify the labels are correct!")

if __name__ == "__main__":
    main()
