# HieraticAI Training Notebook

## Overview

Complete **training + evaluation pipeline** for Google Colab using an A100 GPU. Implements a clean Y-band spatial split with zero pixel leakage between train/val/test.

**Key techniques:**
- 95 active Gardiner categories (from pWestcar)
- Non-overlapping Y-band spatial split (30px buffer zones)
- RepeatFactorTrainingSampler for rare category oversampling
- Heavy augmentation (rotation ±15°, flip, brightness 0.6–1.4)
- Custom anchors tuned for small hieroglyphic signs
- Test-time augmentation (TTA) at evaluation

## Running on Google Colab

1. Upload `annotations.json` and `train_val.png` to Google Drive under `HieraticAI/data/`
2. Open `01_Training.ipynb` in Colab
3. Update `PROJECT` path variable at notebook top to match your Drive path
4. Run all cells (≈1 hour on A100)

## Results

| Metric | Value |
|--------|-------|
| mAP (standard) | 30.9% |
| mAP (with TTA) | 36.4% |
| AP50 (TTA) | 59.7% |

## Dataset Split

- **Train**: Y ≤ 1325 (455 signs, 90 categories)
- **Val**: Y 1355–1480 (67 signs)
- **Test**: Y > 1510 (58 signs, 26 categories)
