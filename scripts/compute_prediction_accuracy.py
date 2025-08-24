#!/usr/bin/env python3
"""
Compute prediction accuracy metrics by comparing model predictions against ground truth.

This script provides complementary metrics to mAP by computing direct prediction accuracy,
which is more interpretable from an Egyptological perspective.

Author: HieraticAI Project
Date: August 2025
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available, visualizations will be skipped")


class PredictionAccuracyAnalyzer:
    """Analyze prediction accuracy by comparing model outputs with ground truth."""
    
    def __init__(self, dataset_path: str, predictions_file: str, iou_threshold: float = 0.5):
        """
        Initialize the accuracy analyzer.
        
        Args:
            dataset_path: Path to COCO dataset directory
            predictions_file: Path to COCO predictions JSON file
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.dataset_path = Path(dataset_path)
        self.predictions_file = Path(predictions_file)
        self.iou_threshold = iou_threshold
        
        # Load data
        self.test_annotations = self._load_test_annotations()
        self.predictions = self._load_predictions()
        self.category_map = self._load_category_mapping()
        
        print(f"📊 Loaded {len(self.test_annotations)} ground truth annotations")
        print(f"📊 Loaded {len(self.predictions)} predictions")
        print(f"📊 Using IoU threshold: {self.iou_threshold}")
    
    def _load_test_annotations(self) -> List[Dict]:
        """Load ground truth annotations from test set."""
        test_ann_file = self.dataset_path / "test" / "annotations.json"
        
        if not test_ann_file.exists():
            raise FileNotFoundError(f"Test annotations not found: {test_ann_file}")
        
        with open(test_ann_file, 'r') as f:
            data = json.load(f)
        
        return data['annotations']
    
    def _load_predictions(self) -> List[Dict]:
        """Load model predictions."""
        if not self.predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_file}")
        
        with open(self.predictions_file, 'r') as f:
            predictions = json.load(f)
        
        return predictions
    
    def _load_category_mapping(self) -> Dict[int, str]:
        """Load category ID to name mapping."""
        test_ann_file = self.dataset_path / "test" / "annotations.json"
        
        with open(test_ann_file, 'r') as f:
            data = json.load(f)
        
        return {cat['id']: cat['name'] for cat in data['categories']}
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes in COCO format [x, y, w, h]."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to [x1, y1, x2, y2] format
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Compute intersection
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute_accuracy_metrics(self) -> Dict:
        """
        Compute comprehensive accuracy metrics.
        
        Returns:
            Dictionary containing various accuracy metrics
        """
        print("🔍 Computing accuracy metrics...")
        # Group annotations and predictions by image
        gt_by_image = defaultdict(list)
        pred_by_image = defaultdict(list)
        
        for ann in self.test_annotations:
            gt_by_image[ann['image_id']].append(ann)
        
        for pred in self.predictions:
            pred_by_image[pred['image_id']].append(pred)
        
        # Compute metrics for each image
        total_gt = 0
        total_predictions = 0
        correct_detections = 0
        correct_classifications = 0
        
        per_category_stats = defaultdict(lambda: {
            'gt_count': 0, 
            'pred_count': 0, 
            'correct_detections': 0,
            'correct_classifications': 0
        })
        
        confidence_scores = []
        correct_confidence_scores = []
        
        image_results = []
        
        for image_id in gt_by_image.keys():
            gt_annotations = gt_by_image[image_id]
            predictions = pred_by_image.get(image_id, [])
            
            total_gt += len(gt_annotations)
            total_predictions += len(predictions)
            
            # Match predictions to ground truth
            matched_gt = set()
            image_correct_det = 0
            image_correct_cls = 0
            
            for pred in predictions:
                pred_bbox = pred['bbox']
                pred_category = pred['category_id']
                pred_score = pred['score']
                
                confidence_scores.append(pred_score)
                
                best_iou = 0
                best_match_idx = -1
                
                # Find best matching ground truth
                for i, gt in enumerate(gt_annotations):
                    if i in matched_gt:
                        continue
                    
                    gt_bbox = gt['bbox']
                    iou = self._compute_iou(pred_bbox, gt_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                
                # Check if we have a valid match
                if best_iou >= self.iou_threshold and best_match_idx >= 0:
                    matched_gt.add(best_match_idx)
                    correct_detections += 1
                    image_correct_det += 1
                    
                    # Check classification accuracy
                    gt_category = gt_annotations[best_match_idx]['category_id']
                    if pred_category == gt_category:
                        correct_classifications += 1
                        image_correct_cls += 1
                        correct_confidence_scores.append(pred_score)
                    
                    # Per-category stats
                    gt_cat_name = self.category_map.get(gt_category, f"cat_{gt_category}")
                    per_category_stats[gt_cat_name]['correct_detections'] += 1
                    if pred_category == gt_category:
                        per_category_stats[gt_cat_name]['correct_classifications'] += 1
            
            # Count ground truth categories
            for gt in gt_annotations:
                gt_cat_name = self.category_map.get(gt['category_id'], f"cat_{gt['category_id']}")
                per_category_stats[gt_cat_name]['gt_count'] += 1
            
            # Count predicted categories  
            for pred in predictions:
                pred_cat_name = self.category_map.get(pred['category_id'], f"cat_{pred['category_id']}")
                per_category_stats[pred_cat_name]['pred_count'] += 1
            
            # Store image-level results
            image_results.append({
                'image_id': image_id,
                'gt_count': len(gt_annotations),
                'pred_count': len(predictions),
                'correct_detections': image_correct_det,
                'correct_classifications': image_correct_cls
            })
        
        # Compute overall metrics
        detection_recall = correct_detections / total_gt if total_gt > 0 else 0
        detection_precision = correct_detections / total_predictions if total_predictions > 0 else 0
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0
        
        classification_accuracy = correct_classifications / correct_detections if correct_detections > 0 else 0
        overall_accuracy = correct_classifications / total_gt if total_gt > 0 else 0
        
        # Confidence analysis
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        avg_correct_confidence = np.mean(correct_confidence_scores) if correct_confidence_scores else 0
        
        results = {
            'overall_metrics': {
                'total_ground_truth': total_gt,
                'total_predictions': total_predictions,
                'correct_detections': correct_detections,
                'correct_classifications': correct_classifications,
                'detection_recall': detection_recall,
                'detection_precision': detection_precision,
                'detection_f1': detection_f1,
                'classification_accuracy': classification_accuracy,
                'overall_accuracy': overall_accuracy,
                'average_confidence': avg_confidence,
                'average_correct_confidence': avg_correct_confidence
            },
            'per_category_metrics': dict(per_category_stats),
            'per_image_results': image_results,
            'confidence_scores': confidence_scores,
            'correct_confidence_scores': correct_confidence_scores
        }
        
        return results
    
    def print_accuracy_report(self, results: Dict) -> None:
        """Print a comprehensive accuracy report."""
        metrics = results['overall_metrics']
        
        print("\n" + "="*80)
        print("🏺 HIERATIC CHARACTER PREDICTION ACCURACY REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL DETECTION PERFORMANCE")
        print(f"   Total Ground Truth Annotations: {metrics['total_ground_truth']:,}")
        print(f"   Total Model Predictions: {metrics['total_predictions']:,}")
        print(f"   Correct Detections (IoU≥{self.iou_threshold}): {metrics['correct_detections']:,}")
        print(f"   Correct Classifications: {metrics['correct_classifications']:,}")
        
        print(f"\n🎯 ACCURACY METRICS")
        print(f"   Detection Recall: {metrics['detection_recall']:.3f} ({metrics['detection_recall']*100:.1f}%)")
        print(f"   Detection Precision: {metrics['detection_precision']:.3f} ({metrics['detection_precision']*100:.1f}%)")
        print(f"   Detection F1-Score: {metrics['detection_f1']:.3f}")
        print(f"   Classification Accuracy*: {metrics['classification_accuracy']:.3f} ({metrics['classification_accuracy']*100:.1f}%)")
        print(f"   Overall Accuracy**: {metrics['overall_accuracy']:.3f} ({metrics['overall_accuracy']*100:.1f}%)")
        
        print(f"\n📈 CONFIDENCE ANALYSIS")
        print(f"   Average Prediction Confidence: {metrics['average_confidence']:.3f}")
        print(f"   Average Confidence (Correct): {metrics['average_correct_confidence']:.3f}")
        
        print(f"\n📝 METRIC DEFINITIONS")
        print(f"   * Classification Accuracy: Correct categories among detected hieroglyphs")
        print(f"   ** Overall Accuracy: Correct detections+classifications among all ground truth")
        print(f"   Detection uses IoU threshold ≥ {self.iou_threshold} for matching")
        
        # Top performing categories
        per_cat = results['per_category_metrics']
        category_accuracies = []
        
        for cat_name, stats in per_cat.items():
            if stats['gt_count'] >= 5:  # Only categories with sufficient samples
                accuracy = stats['correct_classifications'] / stats['gt_count']
                category_accuracies.append((cat_name, accuracy, stats['gt_count']))
        
        category_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP 10 BEST PERFORMING CATEGORIES (≥5 samples)")
        for i, (cat_name, accuracy, count) in enumerate(category_accuracies[:10], 1):
            print(f"   {i:2d}. {cat_name:8s}: {accuracy:.3f} ({accuracy*100:.1f}%) - {count} samples")
        
        print(f"\n⚠️  BOTTOM 10 CATEGORIES (≥5 samples)")
        for i, (cat_name, accuracy, count) in enumerate(category_accuracies[-10:], 1):
            print(f"   {i:2d}. {cat_name:8s}: {accuracy:.3f} ({accuracy*100:.1f}%) - {count} samples")
        
        print("\n" + "="*80)
        print("✅ Report complete! Metrics saved to accuracy_analysis_report.json")
        print("="*80)
    
    def save_detailed_report(self, results: Dict, output_file: str = "accuracy_analysis_report.json") -> None:
        """Save detailed results to JSON file."""
        
        # Make results JSON serializable
        serializable_results = {
            'overall_metrics': results['overall_metrics'],
            'per_category_metrics': results['per_category_metrics'],
            'analysis_parameters': {
                'iou_threshold': self.iou_threshold,
                'dataset_path': str(self.dataset_path),
                'predictions_file': str(self.predictions_file),
                'total_categories': len(self.category_map)
            },
            'summary': {
                'date_generated': pd.Timestamp.now().isoformat(),
                'main_finding': f"Overall accuracy: {results['overall_metrics']['overall_accuracy']:.1%}",
                'interpretation': "This represents the percentage of ground truth hieroglyphs that were correctly detected AND classified by the model."
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📁 Detailed results saved to: {output_file}")
    
    def create_visualization(self, results: Dict, output_file: str = "accuracy_analysis_plots.png") -> None:
        """Create visualization of accuracy metrics."""
        
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Cannot create visualizations: matplotlib not available")
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall metrics bar chart
        metrics = results['overall_metrics']
        metric_names = ['Detection\nRecall', 'Detection\nPrecision', 'Classification\nAccuracy', 'Overall\nAccuracy']
        metric_values = [
            metrics['detection_recall'],
            metrics['detection_precision'], 
            metrics['classification_accuracy'],
            metrics['overall_accuracy']
        ]
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.8)
        ax1.set_ylabel('Score')
        ax1.set_title('🏺 Overall Accuracy Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}\n({value*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Confidence distribution
        conf_scores = results['confidence_scores']
        correct_conf_scores = results['correct_confidence_scores']
        
        ax2.hist(conf_scores, bins=30, alpha=0.6, label='All Predictions', color='lightblue', density=True)
        ax2.hist(correct_conf_scores, bins=30, alpha=0.8, label='Correct Predictions', color='darkblue', density=True)
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Density')
        ax2.set_title('📊 Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Per-category accuracy (top categories)
        per_cat = results['per_category_metrics']
        category_data = []
        
        for cat_name, stats in per_cat.items():
            if stats['gt_count'] >= 3:  # Minimum samples for meaningful accuracy
                accuracy = stats['correct_classifications'] / stats['gt_count'] if stats['gt_count'] > 0 else 0
                category_data.append((cat_name, accuracy, stats['gt_count']))
        
        category_data.sort(key=lambda x: x[2], reverse=True)  # Sort by sample count
        top_categories = category_data[:15]  # Top 15 by sample count
        
        cat_names = [item[0] for item in top_categories]
        cat_accuracies = [item[1] for item in top_categories]
        cat_counts = [item[2] for item in top_categories]
        
        y_pos = np.arange(len(cat_names))
        bars = ax3.barh(y_pos, cat_accuracies, color='green', alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(cat_names, fontsize=9)
        ax3.set_xlabel('Accuracy')
        ax3.set_title('🎯 Top Categories by Sample Count\n(Accuracy Shown)', fontsize=14, fontweight='bold')
        ax3.set_xlim(0, 1)
        
        # Add sample counts as text
        for i, (bar, count) in enumerate(zip(bars, cat_counts)):
            ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{count}', va='center', fontsize=8, color='gray')
        
        # 4. Detection vs Classification accuracy comparison
        image_results = results['per_image_results']
        
        detection_rates = []
        classification_rates = []
        
        for img_result in image_results:
            if img_result['gt_count'] > 0:
                det_rate = img_result['correct_detections'] / img_result['gt_count']
                detection_rates.append(det_rate)
                
                if img_result['correct_detections'] > 0:
                    cls_rate = img_result['correct_classifications'] / img_result['correct_detections']
                else:
                    cls_rate = 0
                classification_rates.append(cls_rate)
        
        ax4.scatter(detection_rates, classification_rates, alpha=0.6, s=30)
        ax4.set_xlabel('Detection Rate (per image)')
        ax4.set_ylabel('Classification Rate (per image)')
        ax4.set_title('🔍 Detection vs Classification\n(Per Image)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect correlation')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to: {output_file}")
        
        return fig


def main():
    """Main function to run accuracy analysis."""
    
    if len(sys.argv) < 3:
        print("Usage: python compute_prediction_accuracy.py <dataset_path> <predictions_file> [iou_threshold]")
        print("Example: python compute_prediction_accuracy.py hieroglyphs_dataset output/predictions.json 0.5")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    predictions_file = sys.argv[2]
    iou_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    try:
        # Initialize analyzer
        analyzer = PredictionAccuracyAnalyzer(dataset_path, predictions_file, iou_threshold)
        
        # Compute accuracy metrics
        results = analyzer.compute_accuracy_metrics()
        
        # Print report
        analyzer.print_accuracy_report(results)
        
        # Save detailed results
        analyzer.save_detailed_report(results)
        
        # Create visualizations (if matplotlib is available)
        if MATPLOTLIB_AVAILABLE:
            analyzer.create_visualization(results)
        else:
            print("⚠️  Skipping visualizations (matplotlib not available)")
        
        print(f"\n🎯 Key Finding:")
        overall_acc = results['overall_metrics']['overall_accuracy']
        classification_acc = results['overall_metrics']['classification_accuracy']
        print(f"   • Overall accuracy: {overall_acc:.1%} (correct detections + classifications)")
        print(f"   • Classification accuracy: {classification_acc:.1%} (correct categories among detections)")
        print(f"   • This provides interpretable accuracy metrics complementary to mAP scores")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
