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

1. **Windows Users**: Double-click `install.py` in the HieraticAI folder
2. **Mac/Linux Users**: Open Terminal, navigate to the folder, and run: `python3 install.py`

**That's it!** The installer will:
- Check if you have Python 3.8+ (and tell you if you need to install it)
- Download and install all required software (~2GB)
- Create a simple launcher for you to use
- Test everything to make sure it works

### Step 3: Launch HieraticAI
After installation completes:
- **Windows**: Double-click `start_hieratic_ai.bat`
- **Mac/Linux**: Run `./start_hieratic_ai.sh`

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
streamlit run tools/validation/prediction_validator.py
```


## Your First Validation Session

### Step 1: Understand the Interface

The interface displays:
- **Left Panel**: Westcar Papyrus image with colored bounding boxes
- **Right Panel**: Validation controls and detailed sign information
- **Bottom**: Progress statistics and export options

### Step 2: Adjust Confidence Threshold

Start with the default threshold (0.3) to see all predictions:
- **High confidence (0.8-1.0)**: Likely correct predictions
- **Medium confidence (0.5-0.8)**: Review carefully  
- **Low confidence (0.0-0.5)**: Often need correction

### Step 3: Select Your First Prediction

1. Look at the manuscript image - you'll see numbered bounding boxes
2. In the right panel, use the dropdown: `Select a prediction...`
3. Choose: `[PENDING] 1. A1 (conf: 0.85)` (example)

### Step 4: Review the Sign Context

For the selected prediction, examine:
- **Cropped Image**: Isolated view of the detected sign
- **Gardiner Info**: Code, Unicode character, description  
- **TLA Data**: Transliteration, translation, frequency
- **AKU References**: Similar signs from the database

### Step 5: Make Your First Validation

Based on your expert assessment:
- Click **Correct** if the AI prediction is accurate
- Click **Incorrect** if wrong classification or bounding box
- Click **Uncertain** for ambiguous or damaged signs

### Step 6: Track Your Progress

Watch the statistics update in real-time:
- **Progress Bar**: Shows validation completion
- **Accuracy Metrics**: Running accuracy percentage
- **Distribution Chart**: Breakdown of validation outcomes

## Best Practices for Validation

**Validation Tips:**
1. **Start with high confidence predictions** - validate obvious correct ones first
2. **Review TLA and AKU data** before making decisions
3. **Export regularly** to save your progress
4. **Take breaks** to maintain concentration

**Export Results:** Click "Export Validation Results" button in the interface to download a CSV with your validation data.

## Troubleshooting

### Common Issues

#### Interface Won't Load
```bash
# Check if all dependencies installed
pip list | grep streamlit
pip list | grep torch

# Try clearing Streamlit cache
streamlit cache clear

# Restart with verbose output
streamlit run tools/validation/prediction_validator.py --logger.level=debug
```

#### Database Connection Issues
```bash
# Check file paths exist
ls -la data/
ls -la "AKU Westcar Scraping/"

# Verify prediction file exists
ls -la output/*/coco_instances_results_FIXED.json
```

#### Memory Issues
```bash
# For low-memory systems:
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200

# Or use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

### Getting Help

If you encounter issues:

1. **Check Prerequisites**: Verify Python version and dependencies
2. **Review Error Messages**: Look for specific error details in terminal
3. **Check File Permissions**: Ensure read/write access to project directories
4. **Update Dependencies**: Try `pip install --upgrade -r requirements.txt`
5. **GitHub Issues**: Report bugs at [GitHub Issues](https://github.com/MargotBelot/HieraticAI/issues)

## Next Steps

Next, explore:

- **[Technical Guide](TECHNICAL_GUIDE.md)**: Advanced features and customization
- **Validation Best Practices**: Develop systematic validation workflows
- **Research Applications**: Integrate results into your research
- **Contributing**: Help improve HieraticAI for the academic community

## Support

Need additional help?
- **Documentation**: Check the Technical Guide for advanced topics
- **Bug Reports**: Use GitHub Issues with detailed error information
- **Feature Requests**: Suggest improvements through GitHub Discussions
  
---
