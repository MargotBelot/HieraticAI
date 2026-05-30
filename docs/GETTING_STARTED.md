# Getting Started with HieraticAI

> **Academic Prototype Notice**: This is a methodological prototype developed for the "Ancient Language Processing" seminar at Freie Universität Berlin (Summer 2025). The project demonstrates computational approaches to ancient Egyptian paleographic analysis.

This guide will walk you through the **complete setup process from scratch**. We have multiple installation options depending on your comfort level with this method.

## Table of Contents

- [Quick Installation](#one-click-installation-recommended)
- [Manual Installation](#manual-installation-for-experienced-users)
- [First Use](#your-first-validation-session)
- [Troubleshooting](#troubleshooting)

## One-Click Installation (Recommended)

**The easiest way to install HieraticAI - no technical knowledge required!**

### Step 1: Download HieraticAI
1. Go to [github.com/MargotBelot/HieraticAI](https://github.com/MargotBelot/HieraticAI)
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP file to your Desktop (or anywhere you like)

### Step 2: Run the Automatic Installer

**Important**: Make sure you have **Python 3.8 or newer** installed first!
- **Windows/Mac**: Download from [python.org](https://www.python.org/downloads/)
- **Linux**: Install with `sudo apt install python3 python3-pip` (Ubuntu/Debian)

Open Terminal, navigate to the folder, and run: `python3 install.py`

**That's it!** The installer will:
- Check if you have Python 3.8+ (and tell you if you need to install it)
- Download and install all required software (~2GB)
- Create a simple launcher for you to use
- Test everything to make sure it works

### Step 3: Launch HieraticAI
After installation completes:
- **Mac**: Double-click **HieraticAI** on your Desktop (or run `./start_hieratic_ai.sh` from terminal)
- **Linux**: Run `./start_hieratic_ai.sh` from terminal

Your web browser will open automatically with the HieraticAI interface!

---

## Manual Installation (For Experienced Users)

If you prefer to install manually or already have Python/Git:

```bash
git clone https://github.com/MargotBelot/HieraticAI.git
cd HieraticAI
python3 -m venv hieratic_env
# Windows: hieratic_env\Scripts\activate
# Mac/Linux: source hieratic_env/bin/activate
pip install -r requirements.txt
# Note: This will automatically install Detectron2 and all other dependencies
streamlit run tools/validation/prediction_validator.py
```


## Your First Validation Session

### Interface Layout

- **Top**: Westcar Papyrus with colored bounding boxes
- **Left**: Sign details (image, Gardiner code, TLA data, AKU references)
- **Right**: Validation controls (Correct/Incorrect/Uncertain buttons)
- **Bottom**: Progress statistics and export

### Workflow

1. **Adjust threshold**: Use sidebar slider to set confidence level (default: 0.3)
2. **Select a prediction**: Choose from dropdown (shows confidence score)
3. **Review context**: Check cropped image, Gardiner code, TLA/AKU data
4. **Validate**: Click Correct / Incorrect / Uncertain
5. **Export**: Download CSV results when done

**Tip**: Start with high-confidence predictions (0.8+) to validate obvious ones first.

## Troubleshooting

**Issues with installation?**
- Verify Python 3.8+: `python3 --version`
- Reinstall dependencies: `pip install --upgrade -r requirements.txt`
- Clear Streamlit cache: `streamlit cache clear`

**Interface won't load?**
```bash
streamlit run tools/validation/prediction_validator.py --logger.level=debug
```

**Database connection issues?**
- Check paths exist: `ls data/ model/`
- Verify prediction file: `ls model/eval_test/coco_instances_results.json`

**Need help?** Report issues on [GitHub Issues](https://github.com/MargotBelot/HieraticAI/issues)
  
---
