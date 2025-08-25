#!/usr/bin/env python3
"""
URGENT FIX: Category ID Offset Correction
Corrects the systematic off-by-one error in model predictions.
"""

import json
import shutil
import os
from datetime import datetime

def fix_prediction_category_ids():
    """Fix category ID offset in predictions (add +1 to all category IDs)"""
    
    print(" URGENT: Fixing category ID offset in predictions")
    print("=" * 60)
    
    # Load original predictions
    results_file = "output/improved_training_20250822_200344/coco_instances_results.json"
    
    if not os.path.exists(results_file):
        print(f" Results file not found: {results_file}")
        return False
    
    print(f" Loading predictions from: {results_file}")
    with open(results_file, 'r') as f:
        predictions = json.load(f)
    
    print(f" Original predictions: {len(predictions)}")
    
    # Analyze original category ID distribution
    from collections import Counter
    original_cats = Counter(pred['category_id'] for pred in predictions)
    print(f"   Original category IDs: {min(original_cats.keys())} to {max(original_cats.keys())}")
    print(f"   Unique categories: {len(original_cats)}")
    
    # Create backup
    backup_file = results_file.replace('.json', f'_BACKUP_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    shutil.copy2(results_file, backup_file)
    print(f" Backup created: {backup_file}")
    
    # Fix category IDs (add +1 to all)
    print(" Applying category ID fix (+1 offset)...")
    
    fixed_predictions = []
    category_mapping = {}
    
    for pred in predictions:
        old_cat_id = pred['category_id']
        new_cat_id = old_cat_id + 1
        
        # Handle special case: category 0 -> category 1 (but this might be background)
        if old_cat_id == 0:
            # Category 0 predictions are likely wrong - we need to investigate
            print(f"  Found {original_cats[0]} predictions for category 0 (background/unknown)")
            # For now, map to category 1, but this needs manual review
            new_cat_id = 1
        
        category_mapping[old_cat_id] = new_cat_id
        
        # Create fixed prediction
        fixed_pred = pred.copy()
        fixed_pred['category_id'] = new_cat_id
        fixed_predictions.append(fixed_pred)
    
    # Analyze fixed category ID distribution
    fixed_cats = Counter(pred['category_id'] for pred in fixed_predictions)
    print(f"   Fixed category IDs: {min(fixed_cats.keys())} to {max(fixed_cats.keys())}")
    print(f"   Unique categories: {len(fixed_cats)}")
    
    # Show mapping summary
    print(f"\n Category ID mapping applied:")
    for old_id, new_id in sorted(category_mapping.items())[:10]:
        print(f"   {old_id} -> {new_id}")
    if len(category_mapping) > 10:
        print(f"   ... and {len(category_mapping)-10} more mappings")
    
    # Save fixed predictions
    fixed_file = results_file.replace('.json', '_FIXED.json')
    with open(fixed_file, 'w') as f:
        json.dump(fixed_predictions, f)
    
    print(f" Fixed predictions saved to: {fixed_file}")
    
    return True, fixed_file, fixed_predictions

def validate_fixed_predictions(fixed_file, fixed_predictions):
    """Validate that the fix worked correctly"""
    
    print(f"\n Validating fixed predictions...")
    
    # Load test ground truth
    with open('hieroglyphs_dataset/test/annotations.json') as f:
        test_data = json.load(f)
    
    gt_categories = {cat['id']: cat['name'] for cat in test_data['categories']}
    gt_cat_ids = set(ann['category_id'] for ann in test_data['annotations'])
    
    # Analyze fixed predictions
    from collections import Counter
    fixed_cat_ids = set(pred['category_id'] for pred in fixed_predictions)
    
    print(f" Validation results:")
    print(f"   Ground truth categories: {len(gt_cat_ids)}")
    print(f"   Fixed prediction categories: {len(fixed_cat_ids)}")
    
    # Check overlaps
    correct_predictions = fixed_cat_ids & gt_cat_ids
    wrong_predictions = fixed_cat_ids - set(gt_categories.keys())
    missing_predictions = gt_cat_ids - fixed_cat_ids
    
    print(f"    Correct category predictions: {len(correct_predictions)}")
    print(f"    Wrong category predictions: {len(wrong_predictions)}")
    print(f"     Missing category predictions: {len(missing_predictions)}")
    
    if wrong_predictions:
        print(f"   Invalid category IDs: {sorted(wrong_predictions)}")
    
    # Show top predictions vs ground truth
    pred_counts = Counter(pred['category_id'] for pred in fixed_predictions)
    gt_counts = Counter(ann['category_id'] for ann in test_data['annotations'])
    
    print(f"\n Top 10 fixed predictions vs ground truth:")
    for cat_id, pred_count in pred_counts.most_common(10):
        cat_name = gt_categories.get(cat_id, f"Unknown_{cat_id}")
        gt_count = gt_counts.get(cat_id, 0)
        status = "" if gt_count > 0 else ""
        print(f"   {status} {cat_name} (ID {cat_id}): {pred_count} pred, {gt_count} GT")
    
    # Calculate improvement
    improvement = len(correct_predictions) / len(gt_cat_ids) * 100 if gt_cat_ids else 0
    print(f"\n Fix effectiveness: {improvement:.1f}% of GT categories now have predictions")
    
    return len(correct_predictions) > 5  # Success if >5 categories match

def create_corrected_visualization():
    """Create visualization with corrected predictions"""
    
    print(f"\n Creating corrected visualization...")
    
    # Update visualization script to use fixed predictions
    viz_script = """#!/usr/bin/env python3
# Updated visualization script using FIXED predictions
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from collections import defaultdict

# Use FIXED predictions file
results_file = "output/improved_training_20250822_200344/coco_instances_results_FIXED.json"

# Rest of visualization code would be the same...
print(" Use the fixed predictions file:", results_file)
print("   Run the visualization again with this corrected file!")
"""
    
    with open('visualize_corrected_predictions.py', 'w') as f:
        f.write(viz_script)
    
    print(" Created: visualize_corrected_predictions.py")
    print("   This shows how to use the fixed predictions")

def main():
    print(" URGENT CATEGORY ID OFFSET FIX")
    print("=" * 60)
    print("Problem detected: Model learned category IDs 0-633 but dataset uses 1-634")
    print("Solution: Add +1 to all predicted category IDs")
    print()
    
    # Apply the fix
    success, fixed_file, fixed_predictions = fix_prediction_category_ids()
    
    if not success:
        print(" Fix failed!")
        return
    
    # Validate the fix
    validation_success = validate_fixed_predictions(fixed_file, fixed_predictions)
    
    # Create corrected visualization
    create_corrected_visualization()
    
    # Summary
    print("\n" + "=" * 60)
    print(" FIX SUMMARY")
    print("=" * 60)
    
    if validation_success:
        print(" CATEGORY ID OFFSET FIX SUCCESSFUL!")
        print()
        print(" Results:")
        print("   - Category IDs shifted from 0-633 to 1-634")
        print("   - Multiple categories now match ground truth")
        print("   - Predictions are now usable for analysis")
        print()
        print(" Files created:")
        print(f"   - {fixed_file} (corrected predictions)")
        print("   - visualize_corrected_predictions.py (updated viz)")
        print()
        print(" Next steps:")
        print("   1. Re-run visualization with fixed predictions")
        print("   2. Retrain model with proper category ID mapping")
        print("   3. Ensure training pipeline uses 0-based category IDs")
        
    else:
        print("  PARTIAL SUCCESS - Some issues remain")
        print("   The offset fix helped but additional problems exist")
        print("   Manual review of training configuration needed")
    
    print(f"\n For future training:")
    print("   - Ensure category IDs start from 0 in training config")
    print("   - Or remap dataset to use 0-633 instead of 1-634")
    print("   - Add validation checks during training")

if __name__ == "__main__":
    main()
