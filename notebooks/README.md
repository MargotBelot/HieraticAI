# HieraticAI Notebooks

## Overview

This notebook implements the complete HieraticAI training and evaluation pipeline using a **clean Y-band spatial split** with 30px buffer zones guaranteeing zero pixel leakage between splits.

### Design
- **95 active categories** — only classes present in pWestcar, not all 634 Gardiner codes
- **Non-overlapping Y-band split** with buffer gaps — verified zero cross-split pixel overlap
- **FREEZE_AT=2** — freeze res1/res2, let res3+ fine-tune on papyrus textures
- **RepeatFactorTrainingSampler** (threshold 0.3) — oversample rare categories
- **Custom anchors** `[16, 32, 64, 128, 256]` — tuned for small hieroglyphic signs
- **Heavy augmentation** (rotation ±15°, flip, brightness 0.6–1.4, contrast, saturation, RandomExtent)
- **LR 0.0002**, 30K iterations with cosine schedule on Faster R-CNN R50-FPN
- **Test-time augmentation** (TTA) at evaluation: multi-scale (400, 512, 600) + flip

## Notebook

### 01_Training.ipynb
Complete training + evaluation pipeline for Google Colab (A100 GPU). Handles data splitting, patch generation, dataset creation, training, standard evaluation, and TTA evaluation in one notebook.

## Running on Google Colab

1. Upload `annotations.json` and `train_val.png` to your Google Drive under a `HieraticAI/data/` folder
2. Open `01_Training.ipynb` in Colab
3. Update the `PROJECT` path variable at the top of the notebook to match your Drive path
4. Training takes ~1 hour on A100

## Split Details

- **Train** (Y ≤ 1325): 455 signs, 90 categories
- **Val** (Y 1355–1480): 67 signs
- **Test** (Y > 1510): 58 signs, 26 categories (96% overlap with train)
- **Buffer** (excluded): 25 signs in the 30px gaps between zones

## Results (May 2026)

- **Standard mAP**: 30.9% (AP50: 50.7%)
- **TTA mAP**: 36.4% (AP50: 59.7%)
