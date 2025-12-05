# Hieroglyph Scraping Toolkit

**A simple tool to download ancient Egyptian hieroglyph data from the internet.**

## What does this do?

You're studying ancient Egyptian hieroglyphs and you want to:
- Get a list of all available hieroglyphic signs from a research database
- Download detailed information about each sign
- Download the actual images of the hieroglyphs
- Get statistics about the images (size, complexity, etc.)

This toolkit does all of that automatically. Instead of manually copying data from websites, you run a few simple commands and get organized files with all the data you need.

## Where does the data come from?

The data comes from **AKU-PAL**, a research database at the University of Mainz, Germany.

**Visit the database:** [aku-pal.uni-mainz.de](https://aku-pal.uni-mainz.de)

**What's in the database?**
- Thousands of hieroglyphic signs from ancient Egyptian papyri
- High-quality images of each hieroglyph
- Information about each sign (when it was written, where it came from, etc.)

**What you'll get:**
- Lists of all available hieroglyphic signs
- Detailed information about each sign
- Image files of the hieroglyphs
- Statistics about the images

## What do I need to install?

### Step 1: Install Python

Python is the programming language this toolkit uses.

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download Python (get version 3.8 or newer)
3. Install it (just click through the installer)

**Check if it worked:**
Open Terminal (Mac) or Command Prompt (Windows) and type:
```
python3 --version
```
You should see something like "Python 3.9.6"

### Step 2: Install Google Chrome

The toolkit uses Chrome to automatically browse websites.

1. Go to [chrome.google.com](https://www.google.com/chrome/)
2. Download and install Chrome
3. You don't need to do anything else - the toolkit handles the rest

### Step 3: Install the toolkit's dependencies

This installs the extra tools the toolkit needs.

**Copy and paste this command** into Terminal (Mac) or Command Prompt (Windows):

```
pip3 install selenium webdriver-manager beautifulsoup4 requests cairosvg python-docx numpy
```

Press Enter and wait for it to finish.

### Step 4: Test everything works

Run this command to make sure everything is working:

```
python3 list_signs_modular.py --help
```

If you see a help message, everything is set up correctly.

### If something goes wrong

**"python3: command not found"**
- Try typing `python` instead of `python3`

**"pip3: command not found"**  
- Try typing `pip` instead of `pip3`

**"Permission denied"**
- On Mac/Linux: Try adding `sudo ` before the pip command
- On Windows: Run Command Prompt as Administrator

## Quick Installation Checklist

Before you start, run through this checklist:

- Python 3.7+ installed (`python3 --version`)
- Google Chrome browser installed  
- Dependencies installed (`pip3 install selenium webdriver-manager beautifulsoup4 requests cairosvg python-docx numpy`)
- Test command works (`python3 list_signs_modular.py --help`)

## How to use it (Simple 3-step process)

### Step 1: Set up the configuration

First, tell the toolkit where to save your files:

```
python3 list_signs_modular.py --create-config
```

This creates a settings file called `hieroglyph_config.json`. You can edit this file if you want to change where files are saved, but the defaults work fine for most people.

**For beginners:** You don't need to change anything in this file right now.

### Step 2: Get a list of all hieroglyphs

This will automatically browse the research website and collect a list of all available hieroglyphs:

```
python3 list_signs_modular.py --config hieroglyph_config.json
```

**What this does:** Creates files with lists of all hieroglyphic signs available in the database.

**What you get:** Two files (`sign_numbers.json` and `sign_numbers.txt`) with lists of hieroglyphs.

### Step 3: Download the actual data

This downloads detailed information and images for each hieroglyph:

```
python3 metadata_scraper_modular.py --signs-file sign_numbers.txt --config hieroglyph_config.json --download-svg
```

**What this does:** For each hieroglyph in your list, it downloads detailed information (like dating, source papyrus, etc.) and the actual image of the hieroglyph.

**What you get:** 
- `all_metadata.json` - All the detailed information
- `all_metadata.txt` - Same information in a human-readable format  
- `svg/` folder - Image files of all the hieroglyphs

**Note:** This step takes longer because it's downloading lots of data.

### Optional: Get statistics about the images

If you want to analyze the hieroglyph images (like getting size statistics), you can run:

```
python3 svg_analyzer_modular.py --svg-dir svg --config hieroglyph_config.json
```

**What this does:** Analyzes all the hieroglyph images and creates statistics (average size, complexity, etc.)

**What you get:** A file called `svg_analysis_results.json` with detailed statistics about all the images.

## Useful tips

**Want to see what's happening while it runs?** Add `--verbose` to any command:
```
python3 list_signs_modular.py --config hieroglyph_config.json --verbose
```

**Need help with a command?** Add `--help`:
```
python3 list_signs_modular.py --help
```

## What files will I get?

After running the toolkit, you'll have several files with your hieroglyph data:

**From Step 2:**
- `sign_numbers.json` and `sign_numbers.txt` - Lists of all available hieroglyphs

**From Step 3:**  
- `all_metadata.json` - All detailed information about each hieroglyph (for computers)
- `all_metadata.txt` - Same information in readable format (for humans)
- `svg/` folder - Image files of all the hieroglyphs

**From Optional step:**
- `svg_analysis_results.json` - Statistics about the images

**Plus some log files** that show what happened during the process.

## What if something goes wrong?

**"I get an error when I run the command"**
- Make sure you installed Python and all the dependencies
- Try the installation command again: `pip3 install selenium webdriver-manager beautifulsoup4 requests cairosvg python-docx numpy`

**"It says it can't find Chrome"**
- Install Google Chrome from [chrome.google.com](https://www.google.com/chrome/)
- Make sure Chrome is actually installed and you can open it

**"It's taking forever"**
- Downloading thousands of hieroglyphs takes time
- The toolkit includes delays to be respectful to the research website
- Add `--verbose` to see what's happening: `python3 list_signs_modular.py --verbose`

**"I can't find my files"**
- By default, files are saved to the `output/` directory in your project folder
- You can check the `hieroglyph_config.json` file to see where files are being saved
- Look for the `output_directory` setting in the config file


