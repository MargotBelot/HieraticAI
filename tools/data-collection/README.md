# Hieroglyph Scraping Toolkit

Download ancient Egyptian hieroglyph data from [AKU-PAL](https://aku-pal.uni-mainz.de/) research database.

## Prerequisites

```bash
# Python 3.8+
python3 --version

# Google Chrome browser
# Download from: https://www.google.com/chrome/

# Python dependencies
pip3 install selenium webdriver-manager beautifulsoup4 requests cairosvg python-docx numpy
```

## Quick Start

```bash
# 1. Configure
python3 list_signs_modular.py --create-config

# 2. Get list of signs
python3 list_signs_modular.py --config hieroglyph_config.json

# 3. Download data & images
python3 metadata_scraper_modular.py --signs-file sign_numbers.txt \
  --config hieroglyph_config.json --download-svg

# Optional: Analyze images
python3 svg_analyzer_modular.py --svg-dir svg --config hieroglyph_config.json
```

## Output Files

- `sign_numbers.json/txt` — Available signs list
- `all_metadata.json/txt` — Sign information
- `svg/` — Hieroglyph images
- `svg_analysis_results.json` — Image statistics

## Help

```bash
python3 list_signs_modular.py --help    # Command help
python3 list_signs_modular.py --verbose # See progress
```

Output files saved to `output/` by default. Edit `hieroglyph_config.json` to change.
