# Hieroglyph Detection Notebooks

This directory contains comprehensive Jupyter notebooks that walk through the complete hieroglyph detection pipeline. Each notebook focuses on a specific aspect of the project and can be run independently.

## Notebook Overview

### [01_Data_Preparation.ipynb](01_Data_Preparation.ipynb)
**Data Leakage Prevention & Dataset Creation**

- **Purpose**: Prepare training data from annotated papyrus images
- **Key Features**:
  - Patch-based image splitting
  - Spatial grouping to prevent data leakage
  - Heavy data augmentation
  - Train/validation/test split creation
- **Duration**: ~10-15 minutes
- **Prerequisites**: Original annotated image data

### [02_Training.ipynb](02_Training.ipynb)  
**Model Training Pipeline**

- **Purpose**: Train hieroglyph detection models using Detectron2
- **Key Features**:
  - Detectron2 configuration setup
  - Custom dataset registration
  - Training loop with validation
  - Checkpoint management
- **Duration**: ~2-4 hours (depending on hardware)
- **Prerequisites**: Prepared dataset from notebook 01

### [03_Evaluation.ipynb](03_Evaluation.ipynb)
**Model Performance Analysis**

- **Purpose**: Comprehensive evaluation of trained models
- **Key Features**:
  - COCO evaluation metrics (mAP, mAP@0.5, etc.)
  - Per-category performance analysis
  - Confusion matrix generation
  - Error analysis and visualization
- **Duration**: ~5-10 minutes
- **Prerequisites**: Trained model from notebook 02

### [04_Inference.ipynb](04_Inference.ipynb)
**Real-time Hieroglyph Detection**

- **Purpose**: Run inference on new papyrus images
- **Key Features**:
  - HieroglyphDetector class implementation
  - Single image and batch processing
  - Test-time augmentation
  - Confidence threshold tuning
  - Result visualization and export
- **Duration**: ~2-5 minutes per image
- **Prerequisites**: Trained model from notebook 02

### [05_Improved_Training.ipynb](05_Improved_Training.ipynb)
**Advanced Training with Focal Loss**

- **Purpose**: Demonstrate improved training techniques
- **Key Features**:
  - Focal Loss implementation for hard examples
  - Enhanced data augmentation
  - Learning rate scheduling
  - Early stopping with validation monitoring
  - Performance comparison with baseline
- **Duration**: ~3-5 hours (full training)
- **Prerequisites**: Understanding of baseline training from notebook 02

## Quick Start Guide

### For New Users
1. **Start Here**: [01_Data_Preparation.ipynb](01_Data_Preparation.ipynb)
2. **Then**: [02_Training.ipynb](02_Training.ipynb)  
3. **Evaluate**: [03_Evaluation.ipynb](03_Evaluation.ipynb)
4. **Use Model**: [04_Inference.ipynb](04_Inference.ipynb)

### For Advanced Users
- Skip to [04_Inference.ipynb](04_Inference.ipynb) if you have a pre-trained model
- Try [05_Improved_Training.ipynb](05_Improved_Training.ipynb) for advanced techniques
- Use [03_Evaluation.ipynb](03_Evaluation.ipynb) for detailed performance analysis

### For Researchers
- Focus on [05_Improved_Training.ipynb](05_Improved_Training.ipynb) for the latest improvements
- Combine with [03_Evaluation.ipynb](03_Evaluation.ipynb) for comprehensive analysis
- Reference [01_Data_Preparation.ipynb](01_Data_Preparation.ipynb) for data handling best practices

## Running the Notebooks

### Prerequisites
```bash
# Install Jupyter if not already installed
pip install jupyter ipykernel

# Start Jupyter notebook server
jupyter notebook

# Or use Jupyter Lab
pip install jupyterlab
jupyter lab
```

### Environment Setup
```python
# Add this to the beginning of each notebook
import sys
import os
sys.path.append('..')  # Add project root to path

# Verify installation
import torch
import detectron2
print(f"PyTorch: {torch.__version__}")
print(f"Detectron2: {detectron2.__version__}")
```

### Google Colab Usage
All notebooks are designed to work with Google Colab:

1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Mount Google Drive when prompted
4. Install dependencies using the provided cells

## Performance Expectations

 Notebook  Expected Runtime  Memory Usage  GPU Required 
--------------------------------------------------------
 01_Data_Preparation  10-15 min  2-4 GB  No 
 02_Training  2-4 hours  6-8 GB  Yes (recommended) 
 03_Evaluation  5-10 min  2-4 GB  No 
 04_Inference  2-5 min  2-3 GB  Recommended 
 05_Improved_Training  3-5 hours  8-12 GB  Yes 

## Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Solution: Add project root to Python path
import sys
sys.path.append('..')
```

**2. CUDA Memory Issues**
```python
# Solution: Reduce batch size or use CPU
cfg.SOLVER.IMS_PER_BATCH = 2  # Reduce from 4
# or
device = "cpu"  # Force CPU usage
```

**3. Dataset Not Found**
```bash
# Solution: Ensure dataset structure is correct
ls hieroglyphs_dataset/
# Should show: train/ val/ test/ directories
```

**4. Model Loading Issues**
```python
# Solution: Check model path and ensure model exists
model_path = "./output/model_final.pth"
assert os.path.exists(model_path), f"Model not found at {model_path}"
```

## Additional Resources

- **Project Documentation**: `../docs/`
- **Utility Scripts**: `../scripts/`
- **Training Script**: `../train_hieroglyph_detection_robust.py`
- **Dataset Validation**: `../utils/verify_dataset_consistency.py`

## Learning Path

### Beginner Track
1. Read the project README
2. Run [01_Data_Preparation.ipynb](01_Data_Preparation.ipynb)
3. Understand the data structure and format
4. Try [04_Inference.ipynb](04_Inference.ipynb) with pre-trained models

### Intermediate Track  
1. Complete the Beginner Track
2. Run [02_Training.ipynb](02_Training.ipynb)
3. Analyze results with [03_Evaluation.ipynb](03_Evaluation.ipynb)
4. Experiment with different parameters

### Advanced Track
1. Complete the Intermediate Track
2. Study [05_Improved_Training.ipynb](05_Improved_Training.ipynb)
3. Implement custom improvements
4. Contribute to the project

## Tips for Success

1. **Always validate your data** before training
2. **Start with small experiments** before full training
3. **Monitor GPU memory usage** during training
4. **Save checkpoints frequently** during long training runs
5. **Visualize results** to understand model behavior
6. **Compare different approaches** using consistent evaluation

---

**Happy learning!**

For questions or issues, refer to the main project documentation or create an issue in the project repository.
