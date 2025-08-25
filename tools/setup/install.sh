#!/bin/bash

# Hieroglyph Detection Project Installation Script
# This script sets up the complete environment for the hieroglyph detection project

set -e  # Exit on error

echo "Hieroglyph Detection Project Installation"
echo "============================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1  grep -o '[0-9]\+\.[0-9]\+'  head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version""$python_version"sort -V  head -n1)"= "$required_version"]; then
    echo "Python $python_version is compatible (>= 3.8 required)"
else
    echo "Python $python_version is not compatible. Please install Python 3.8 or higher."
    exit 1
fi

# Check if CUDA is available
echo "Checking for CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits  head -1
    echo "CUDA GPU detected"
    CUDA_AVAILABLE=true
else
    echo "No CUDA GPU detected. CPU-only installation will proceed."
    CUDA_AVAILABLE=false
fi

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv"]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch based on CUDA availability
echo "Installing PyTorch..."
if [ "$CUDA_AVAILABLE"= true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install Detectron2
echo "Installing Detectron2..."
if [ "$CUDA_AVAILABLE"= true ]; then
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
else
    # For CPU-only installation, we might need a specific build
    echo "Note: Detectron2 CPU-only support may be limited"
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
fi

# Install other requirements
echo "Installing project dependencies..."
pip install -r requirements.txt

# Verify installations
echo "Verifying installations..."

echo "- Testing torch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

echo "- Testing detectron2..."
python3 -c "import detectron2; print(f'Detectron2 {detectron2.__version__} installed successfully')"

echo "- Testing computer vision libraries..."
python3 -c "import cv2; print(f'OpenCV {cv2.__version__} installed successfully')"

echo "- Testing scientific libraries..."
python3 -c "import numpy as np; import pandas as pd; print('NumPy and Pandas installed successfully')"

# Check dataset
echo "Checking dataset structure..."
if [ -d "hieroglyphs_dataset"]; then
    echo "Dataset directory found"
    if [ -f "hieroglyphs_dataset/train/annotations.json"]; then
        echo "Training annotations found"
    else
        echo "Training annotations not found"
    fi
else
    echo "Dataset directory not found. Please ensure dataset is present."
fi

# Run dataset validation
echo "Running dataset validation..."
if python3 utils/verify_dataset_consistency.py; then
    echo "Dataset validation passed"
else
    echo "Dataset validation failed. Please check the dataset."
fi

# Create output directories
echo "Creating output directories..."
mkdir -p output
mkdir -p docs
mkdir -p logs
echo "Output directories created"

# Set executable permissions on scripts
echo "Setting script permissions..."
find scripts/ -name "*.py"-exec chmod +x {} \;
chmod +x train_hieroglyph_detection_robust.py
echo "Script permissions set"

# Run tests if available
echo "Running tests..."
if [ -d "tests"]; then
    python3 -m pytest tests/ -v
    echo "Tests completed"
else
    echo "No tests directory found"
fi

# Final setup verification
echo "Final verification..."
echo "- Python: $(python3 --version)"
echo "- PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "- CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
if [ "$CUDA_AVAILABLE"= true ]; then
    echo "- GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
fi

echo ""
echo "Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Explore the notebooks: jupyter notebook notebooks/"
echo "3. Run a quick test: python3 scripts/visualize_coco_results.py"
echo "4. Train a model: python3 train_hieroglyph_detection_robust.py"
echo ""
echo "For detailed usage instructions, see:"
echo "- README.md - Project overview"
echo "- docs/GETTING_STARTED.md - Step-by-step guide"
echo "- notebooks/ - Interactive tutorials"
echo ""
echo "Happy hieroglyph detecting!"
