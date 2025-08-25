#!/usr/bin/env python3
"""
Robust Hieroglyph Detection Training Script
Production-ready training with category mapping safeguards and comprehensive validation.

Key Features:
- Automatic category ID remapping (1-based -> 0-based for Detectron2)  
- Built-in dataset validation before training
- Comprehensive error prevention and logging
- Graceful handling of training interruptions
- Automatic model checkpointing and recovery
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import validation
from src.hieratic_ai.utils.dataset_validator import DatasetValidator

# Detectron2 imports
try:
    import torch
    from detectron2 import model_zoo
    from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
    from detectron2.data.datasets import register_coco_instances
    from detectron2.evaluation import COCOEvaluator
    from detectron2.utils.logger import setup_logger
    from detectron2.solver import get_default_optimizer_params
    from detectron2.data import transforms as T
    from detectron2.data import DatasetMapper
except ImportError as e:
    print(f" Failed to import Detectron2: {e}")
    print("Please install Detectron2: https://detectron2.readthedocs.io/en/latest/tutorials/install.html")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HieroglyphTrainingConfig:
    """Configuration class for hieroglyph training with safeguards"""
    
    def __init__(self, args):
        self.args = args
        self.dataset_path = args.dataset_path
        self.output_dir = args.output_dir
        self.model_weights = args.model_weights
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.max_iter = args.max_iter
        self.eval_period = args.eval_period
        
        # Category mapping info (will be set after validation)
        self.num_classes = None
        self.category_id_offset = 0  # Will be -1 if remapping needed
        self.categories = {}
        
        # Training metadata
        self.training_start_time = datetime.now()
        self.run_id = self.training_start_time.strftime("training_%Y%m%d_%H%M%S")
        
    def validate_and_setup(self) -> bool:
        """Validate dataset and setup category mappings"""
        logger.info(" Validating dataset and setting up configuration...")
        
        # Step 1: Run comprehensive dataset validation
        validator = DatasetValidator(self.dataset_path)
        
        if not validator.validate_all():
            logger.error(" Dataset validation failed!")
            return False
        
        # Step 2: Extract category information
        ann_file = f"{self.dataset_path}/train/annotations.json"
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        self.num_classes = len(self.categories)
        
        # Step 3: Check if category remapping is needed
        min_cat_id = min(self.categories.keys())
        
        if min_cat_id == 1:
            logger.warning("  Dataset uses 1-based category IDs")
            logger.warning("   Will remap to 0-based for Detectron2 compatibility")
            self.category_id_offset = -1  # Subtract 1 from all category IDs
        elif min_cat_id == 0:
            logger.info(" Dataset already uses 0-based category IDs")
            self.category_id_offset = 0
        else:
            logger.error(f" Unusual category ID start: {min_cat_id}")
            return False
        
        logger.info(f" Configuration summary:")
        logger.info(f"   Dataset: {self.dataset_path}")
        logger.info(f"   Categories: {self.num_classes}")
        logger.info(f"   Category ID offset: {self.category_id_offset}")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   Run ID: {self.run_id}")
        
        return True

class CategoryRemappingDatasetMapper(DatasetMapper):
    """Custom dataset mapper that handles category ID remapping"""
    
    def __init__(self, cfg, is_train: bool = True, category_offset: int = 0):
        super().__init__(cfg, is_train)
        self.category_offset = category_offset
        
        if category_offset != 0:
            logger.info(f" DatasetMapper will apply category offset: {category_offset}")
    
    def __call__(self, dataset_dict):
        """Apply category ID remapping to annotations"""
        dataset_dict = super().__call__(dataset_dict)
        
        if self.category_offset != 0 and "annotations" in dataset_dict:
            for ann in dataset_dict["annotations"]:
                ann["category_id"] += self.category_offset
                
                # Ensure category IDs are valid
                if ann["category_id"] < 0:
                    logger.error(f" Invalid category ID after remapping: {ann['category_id']}")
                    raise ValueError(f"Category ID remapping resulted in negative ID: {ann['category_id']}")
        
        return dataset_dict

class RobustHieroglyphTrainer(DefaultTrainer):
    """Enhanced trainer with error handling and validation"""
    
    def __init__(self, cfg, training_config: HieroglyphTrainingConfig):
        self.training_config = training_config
        super().__init__(cfg)
        
    @classmethod
    def build_train_loader(cls, cfg, training_config):
        """Build training data loader with category remapping"""
        mapper = CategoryRemappingDatasetMapper(
            cfg, 
            is_train=True, 
            category_offset=training_config.category_id_offset
        )
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod  
    def build_evaluator(cls, cfg, dataset_name):
        """Build evaluator for validation"""
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    
    def train(self):
        """Enhanced training with error handling"""
        logger.info(" Starting robust training...")
        
        try:
            super().train()
            logger.info(" Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.warning("  Training interrupted by user")
            self._save_checkpoint_on_interrupt()
            
        except Exception as e:
            logger.error(f" Training failed with error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._save_checkpoint_on_interrupt()
            raise
    
    def _save_checkpoint_on_interrupt(self):
        """Save checkpoint when training is interrupted"""
        try:
            checkpoint_path = os.path.join(
                self.cfg.OUTPUT_DIR, 
                f"model_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            )
            self.checkpointer.save(checkpoint_path)
            logger.info(f" Saved interruption checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save interruption checkpoint: {e}")

def register_hieroglyph_datasets(config: HieroglyphTrainingConfig):
    """Register datasets with Detectron2 with proper metadata"""
    logger.info(" Registering datasets...")
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        dataset_name = f"hieroglyphs_{split}"
        images_dir = f"{config.dataset_path}/{split}/images"
        annotations_file = f"{config.dataset_path}/{split}/annotations.json"
        
        if os.path.exists(images_dir) and os.path.exists(annotations_file):
            # Register dataset
            register_coco_instances(
                dataset_name, 
                {}, 
                annotations_file, 
                images_dir
            )
            
            # Set up metadata with remapped categories if needed
            if config.category_id_offset != 0:
                # Create remapped category list for metadata
                remapped_categories = []
                for cat_id, cat_name in sorted(config.categories.items()):
                    remapped_id = cat_id + config.category_id_offset
                    remapped_categories.append(cat_name)  # Detectron2 uses index-based naming
                
                MetadataCatalog.get(dataset_name).thing_classes = remapped_categories
            else:
                # Use original categories
                category_names = [config.categories[cat_id] for cat_id in sorted(config.categories.keys())]
                MetadataCatalog.get(dataset_name).thing_classes = category_names
            
            logger.info(f"    Registered {dataset_name}: {len(MetadataCatalog.get(dataset_name).thing_classes)} classes")
    
    logger.info(" Dataset registration complete")

def setup_config(config: HieroglyphTrainingConfig) -> object:
    """Setup Detectron2 configuration"""
    logger.info("  Setting up Detectron2 configuration...")
    
    cfg = get_cfg()
    
    # Model configuration
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
    
    # Critical: Set correct number of classes
    if config.category_id_offset != 0:
        logger.info(f" Configuring for {config.num_classes} classes with ID remapping")
    else:
        logger.info(f" Configuring for {config.num_classes} classes (no remapping)")
    
    # Training configuration
    cfg.DATASETS.TRAIN = ("hieroglyphs_train",)
    cfg.DATASETS.TEST = ("hieroglyphs_val",)
    cfg.DATALOADER.NUM_WORKERS = config.num_workers
    
    # Solver configuration
    cfg.SOLVER.IMS_PER_BATCH = config.batch_size
    cfg.SOLVER.BASE_LR = config.learning_rate
    cfg.SOLVER.MAX_ITER = config.max_iter
    cfg.SOLVER.STEPS = (int(config.max_iter * 0.6), int(config.max_iter * 0.8))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = min(1000, config.max_iter // 10)
    cfg.SOLVER.CHECKPOINT_PERIOD = max(config.eval_period, 2000)
    
    # Evaluation
    cfg.TEST.EVAL_PERIOD = config.eval_period
    
    # Output
    cfg.OUTPUT_DIR = os.path.join(config.output_dir, config.run_id)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Weights
    if config.model_weights:
        cfg.MODEL.WEIGHTS = config.model_weights
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    
    # Data augmentation
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Advanced settings
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f" Configuration complete. Device: {cfg.MODEL.DEVICE}")
    
    return cfg

def save_training_metadata(config: HieroglyphTrainingConfig, cfg):
    """Save training metadata for reproducibility"""
    metadata = {
        'run_id': config.run_id,
        'start_time': config.training_start_time.isoformat(),
        'dataset_path': config.dataset_path,
        'num_classes': config.num_classes,
        'category_id_offset': config.category_id_offset,
        'categories': config.categories,
        'model_config': {
            'weights': cfg.MODEL.WEIGHTS,
            'backbone': cfg.MODEL.RESNETS.DEPTH,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES
        },
        'training_config': {
            'max_iter': cfg.SOLVER.MAX_ITER,
            'learning_rate': cfg.SOLVER.BASE_LR,
            'batch_size': cfg.SOLVER.IMS_PER_BATCH,
            'eval_period': cfg.TEST.EVAL_PERIOD
        },
        'system_info': {
            'device': cfg.MODEL.DEVICE,
            'cuda_available': torch.cuda.is_available(),
            'num_workers': cfg.DATALOADER.NUM_WORKERS
        }
    }
    
    metadata_file = os.path.join(cfg.OUTPUT_DIR, 'training_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f" Training metadata saved: {metadata_file}")

def main(args):
    """Main training function with comprehensive error handling"""
    
    logger.info(" Robust Hieroglyph Detection Training")
    logger.info("=" * 60)
    
    try:
        # Step 1: Initialize and validate configuration
        config = HieroglyphTrainingConfig(args)
        
        if not config.validate_and_setup():
            logger.error(" Configuration validation failed!")
            return False
        
        # Step 2: Register datasets
        register_hieroglyph_datasets(config)
        
        # Step 3: Setup Detectron2 configuration
        cfg = setup_config(config)
        
        # Step 4: Save training metadata
        save_training_metadata(config, cfg)
        
        # Step 5: Initialize trainer
        trainer = RobustHieroglyphTrainer(cfg, config)
        
        # Step 6: Resume from checkpoint if available
        trainer.resume_or_load(resume=args.resume)
        
        # Step 7: Start training
        logger.info(" Launching training...")
        trainer.train()
        
        logger.info(" Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f" Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description="Robust Hieroglyph Detection Training")
    
    parser.add_argument(
        "--dataset-path",
        default="hieroglyphs_dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output-dir", 
        default="output",
        help="Output directory for training results"
    )
    parser.add_argument(
        "--model-weights",
        default=None,
        help="Path to model weights (default: use pretrained COCO weights)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--batch-size",
        type=int, 
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15000,
        help="Maximum training iterations"
    )
    parser.add_argument(
        "--eval-period",
        type=int,
        default=2000,
        help="Evaluation period (iterations)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true", 
        help="Only run dataset validation, don't train"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Setup
    setup_logger()
    args = setup_args()
    
    if args.validate_only:
        # Run validation only
        validator = DatasetValidator(args.dataset_path)
        success = validator.validate_all()
        sys.exit(0 if success else 1)
    else:
        # Run full training
        success = main(args)
        sys.exit(0 if success else 1)
