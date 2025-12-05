# HieraticAI

**AI-powered hieratic character recognition and validation for ancient Egyptian manuscripts**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![Prototype](https://img.shields.io/badge/Status-Prototype-orange.svg)](https://github.com/MargotBelot/HieraticAI)

> **Academic Project Notice**: This is a prototype developed for the "Ancient Language Processing" seminar at Freie Universität Berlin (Summer 2025). The project serves as a methodological exploration rather than a production-ready system.
> 
> **Course Information**: [Ancient Language Processing 2025](https://digitalpasts.github.io/alp-course-2025/) - Hybrid seminar focusing on computational approaches to ancient datasets and digital philology methods.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Interactive Interface](#interactive-validation-interface)
- [Data Collection Tools](#data-collection-tools)
- [Technical Details](#technical-architecture)
- [Documentation](#documentation)

## Project Overview

HieraticAI focuses on the study of ancient Egyptian manuscripts by combining computer vision with Egyptological expertise. This system automatically detects, classifies, and validates hieratic characters in papyrus images, with specialized focus on the **Westcar Papyrus**.

**What makes HieraticAI unique:**
- **AI-First Approach**: Faster R-CNN detection with ResNet-50 backbone across 634 Gardiner code classes
- **Expert Validation**: Interactive interface for Egyptologists to review and correct AI predictions
- **Comprehensive Integration**: Direct links to TLA (Thesaurus Linguae Aegyptiae) and AKU databases
- **Research-Ready**: Exports validated data for paleographic and linguistic research

### The Problem We Solve

Traditional hieratic manuscript analysis is:
- **Time-consuming**: Manual character identification takes a long time for a single manuscript
- **Limited scale**: Impossible to analyze large corpora without automation
- **Subjective**: Difficult to maintain consistency across different scholars

### Our Solution

HieraticAI provides:
- **Automated Detection**: AI identifies and classifies characters in seconds
- **High Accuracy**: Trained specifically on the Westcar papyrus hieratic forms
- **Expert Oversight**: Validation interface ensures rigor
- **Rich Context**: Integrated linguistic and paleographic resources

## Complete Pipeline Overview

```mermaid
graph LR
    %% INPUT
    Start[<b>INPUT</b><br/>Papyrus Westcar<br/>Facsimile Image<br/>Recto VIII, lines 5-24]
    
    %% DATA COLLECTION
    AKU[<b>AKU-PAL Database</b><br/>309 hieroglyphs<br/>781 variants<br/>17th Dynasty<br/>SVG vectors]
    TLA[<b>TLA Database</b><br/>587 lemmas<br/>Transliterations<br/>Translations<br/>92.8% coverage<br/>Fallback strategies]
    
    %% BRANCHING: Two parallel paths from START
    Start --> Path1[<b>PATH A</b><br/>Vector<br/>Reconstruction]
    Start --> Path2[<b>PATH B</b><br/>AI<br/>Recognition]
    
    %% PATH B starts with manual annotation
    Path2 --> CVAT[<b>MANUAL ANNOTATION</b><br/>CVAT Tool<br/>605 signs<br/>Polygonal bounding boxes<br/>Gardiner codes + Unicode]
    
    %% PATH A: VECTOR RECOMPOSITOR
    Path1 --> VR1[<b>Spatial Encoding</b><br/><b>Parser</b><br/>Gardiner expressions<br/>Ligature handling]
    AKU --> VR1
    VR1 --> VR2[<b>Hieratogram</b><br/><b>Matching</b><br/>Period-specific<br/>variants]
    VR2 --> VR3[<b>SVG Line</b><br/><b>Reconstruction</b><br/>Modular<br/>composition<br/>Metadata<br/>embedding]
    VR3 --> OutputA[<b>OUTPUT A</b><br/>Digital Edition<br/>SVG format<br/>20 lines<br/>605 signs]
    
    %% PATH B: AI RECOGNITION (continues from CVAT)
    CVAT --> Patch[<b>Patching</b><br/>Multiple crop views<br/>605 signs to<br/>1,269 instances]
    Patch --> AI1[<b>Spatial Data</b><br/><b>Splitting</b><br/>10 regions<br/>70/20/10 split<br/>Prevents leakage<br/>803 training instances]
    AI1 --> AI2[<b>Data</b><br/><b>Augmentation</b><br/>803 to 4,726<br/>5.9x expansion<br/>Rotation, scaling]
    AI2 --> AI3[<b>Model Training</b><br/>Google Colab A100<br/>Faster R-CNN<br/>ResNet-50<br/>634 categories<br/>15,000 iterations]
    AI3 --> AI5[<b>Trained Model</b><br/>mAP: 31.2%<br/>High-freq: 45-75%<br/>Low-freq: 5-25%]
    
    %% VALIDATION INTERFACE
    AI5 --> Val[<b>VALIDATION</b><br/><b>INTERFACE</b><br/>Streamlit UI<br/>Human-in-the-Loop]
    
    %% Add database context to validation
    AKU -.-> Val
    TLA -.-> Val
    
    Val --> Val2[<b>Expert Review</b><br/>TLA linguistic data<br/>AKU-PAL references<br/>Accept/Reject/Modify<br/>100 signs in<br/>12-15 minutes]
    Val2 --> OutputB[<b>OUTPUT B</b><br/>Validated Dataset<br/>CSV format<br/>Egyptological<br/>validation]
    
    %% FINAL APPLICATIONS
    OutputA --> Apps[<b>RESEARCH</b><br/><b>APPLICATIONS</b>]
    OutputB --> Apps
    Apps --> App1[Digital<br/>Editions]
    Apps --> App2[Paleographic<br/>Analysis]
    Apps --> App3[Large-scale<br/>Corpus Studies]
    
    %% STYLING
    classDef input fill:#95a5a6,stroke:#7f8c8d,stroke-width:3px,color:#fff
    classDef annotation fill:#9b59b6,stroke:#8e44ad,stroke-width:3px,color:#fff
    classDef database fill:#34495e,stroke:#2c3e50,stroke-width:2px,color:#fff
    classDef pathLabel fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px,color:#fff
    classDef vectorComp fill:#3498db,stroke:#2c3e50,stroke-width:2px,color:#fff
    classDef aiComp fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff
    classDef validation fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff
    classDef output fill:#e74c3c,stroke:#c0392b,stroke-width:3px,color:#fff
    classDef apps fill:#16a085,stroke:#138d75,stroke-width:2px,color:#fff
    
    class Start input
    class CVAT annotation
    class AKU,TLA database
    class Path1,Path2 pathLabel
    class VR1,VR2,VR3 vectorComp
    class Patch,AI1,AI2,AI3,AI5 aiComp
    class Val,Val2 validation
    class OutputA,OutputB output
    class Apps,App1,App2,App3 apps
```

## Manuscript Focus: The Westcar Papyrus

**Westcar Papyrus (pBerlin P 3033)** is a significant Middle Kingdom hieratic manuscript containing tales about magicians at the court of King Khufu.

**Key Details:**
- **Location**: Ägyptisches Museum, Berlin
- **Script**: Hieratic (Middle Egyptian)
- **Content**: 5 stories across 12 columns
- **Training Focus**: Recto VIII, lines 5-24
- **Significance**: Primary source for AI model training and hieratic character evolution research

## Quick Start

### One-Click Installation (Recommended)

**Perfect for anyone - no technical experience required!**

**Prerequisites**: You need Python 3.8+ installed on your system. If you don't have it:
- **Windows/Mac**: Download from [python.org](https://www.python.org/downloads/) 
- **Linux**: Install with `sudo apt install python3 python3-pip` (Ubuntu) or equivalent

1. **Download**: Go to [github.com/MargotBelot/HieraticAI](https://github.com/MargotBelot/HieraticAI) and click "Download ZIP"
2. **Extract**: Unzip the file to your Desktop
3. **Install**: Double-click `install.py` (Windows) or run `python3 install.py` (Mac/Linux)
4. **Launch**: Use the created launcher script when installation completes
5. **Validate**: Your browser opens automatically to start validating!

### Manual Installation (For Experienced Users)

**Requires Python 3.8+**

```bash
git clone https://github.com/MargotBelot/HieraticAI.git
cd HieraticAI
python3 -m venv hieratic_env
# Activate: hieratic_env\Scripts\activate (Win) or source hieratic_env/bin/activate (Mac/Linux)
pip install -r requirements.txt
# Note: Detectron2 will be automatically installed from the requirements.txt
streamlit run tools/validation/prediction_validator.py
```

**Then navigate to `http://localhost:8501` in your browser!**

## Interactive Validation Interface

### Real-Time Validation Workflow

The HieraticAI interface provides a validation panel:

#### **Main Interface Layout**

**Manuscript Viewer (Left Panel)**
- **Westcar Papyrus Display**: Testing area view with detected signs of the Westcar papyrus facsimile.
- **Color-coded Predictions**:
  - 🔵 #1 A1 (85%) - Pending validation
  - 🟢 #2 G17 (92%) - High confidence, likely correct
  - 🔴 #3 M17 (76%) - Flagged for review
  - 🟠 #4 D21 (68%) - Medium confidence
  - 🔵 #5 N35 (54%) - Low confidence, needs attention
- **Status Tracking**: "2/5 reviewed" with progress indicator

**Validation Panel (Right Panel)**
- **Current Sign Review**:
  - Cropped image of selected sign
  - Gardiner code: A1
  - Unicode display: 𓀀
- **TLA Linguistic Data**:
  - Transliteration information
  - Lemma details and meanings
  - Related sign forms
- **AKU Reference Signs**:
  - Similar signs from database
  - Quality assessment scores
  - SVG vector displays
- **Validation Actions**:
  - **CORRECT** - Confirm AI prediction
  - **INCORRECT** - Mark as wrong
  - **UNCERTAIN** - Flag for further review
  - **EDIT CODE** - Manual correction

**Control Panel (Bottom)**
- Navigation: Previous/Next/Refresh buttons
- Progress: ████▒▒▒ 67% completion

### Validation Status System

| Color | Status | Meaning |
|-------|--------|---------|
| 🔵 Blue | Pending | Awaiting validation |
| 🟢 Green | Correct | AI prediction is accurate |
| 🔴 Red | Incorrect | AI prediction needs correction |
| 🟠 Orange | Uncertain | Requires expert judgment |

## Database Integration

HieraticAI integrates with two major academic databases to provide context for hieratic character validation:

**AKU-PAL (Altägyptische Kursivschriften)**
- Digital paleography platform from Academy of Sciences, Mainz
- Provides authenticated Westcar Papyrus signs for comparison
- Link: https://aku-pal.uni-mainz.de/

**TLA (Thesaurus Linguae Aegyptiae)**
- Comprehensive lexicographical database of ancient Egyptian
- Provides transliterations, translations, and frequency data
- Ensures 100% coverage through fallback mapping strategies
- Link: https://thesaurus-linguae-aegyptiae.de/home 

## Data Collection Tools

**Included: Hieroglyph Scraping Toolkit**

HieraticAI includes a comprehensive data collection toolkit for downloading hieroglyphic data from academic databases. This toolkit automates the process of gathering training data and reference materials.

**Location**: `tools/data-collection/`

**What it does:**
- Downloads hieroglyphic signs from AKU-PAL database
- Collects detailed metadata for each sign
- Downloads high-quality SVG images
- Generates statistics and analysis reports
- Creates organized datasets for training and research

**Quick Start with Data Collection:**
1. **Set up configuration**:
   ```bash
   cd tools/data-collection
   python3 list_signs_modular.py --create-config
   ```

2. **Collect hieroglyph list**:
   ```bash
   python3 list_signs_modular.py --config hieroglyph_config.json
   ```

3. **Download data and images**:
   ```bash
   python3 metadata_scraper_modular.py --signs-file sign_numbers.txt --config hieroglyph_config.json --download-svg
   ```

**Output files:**
- `sign_numbers.json/txt` - Lists of available hieroglyphs
- `all_metadata.json/txt` - Detailed information about each sign
- `svg/` folder - Vector graphics of all hieroglyphs
- `svg_analysis_results.json` - Image statistics and analysis

**Requirements**: Python 3.8+, Chrome browser, see `tools/data-collection/README.md` for detailed setup instructions.

This toolkit is perfect for:
- Expanding training datasets
- Collecting reference materials
- Building custom hieroglyphic corpora
- Research data gathering

**Integration with HieraticAI Workflow:**

The data collection tools complement the main HieraticAI system by providing fresh training data:

```mermaid
graph LR
    A[Data Collection Tools] --> B[Download AKU Signs]
    B --> C[Generate Training Data]
    C --> D[Train AI Model]
    D --> E[Validation Interface]
    E --> F[Research Output]
    
    style A fill:#f39c12
    style D fill:#3498db
    style E fill:#e74c3c
```

1. **Collect Data**: Use the scraping toolkit to gather hieroglyphic signs
2. **Process Dataset**: Convert downloaded data to training format
3. **Train Model**: Use collected data to improve AI accuracy
4. **Validate Results**: Use the main HieraticAI interface for validation

## Model Performance

| Metric | Value |
|--------|-------|
| **Detection Model** | Faster R-CNN with ResNet-50 backbone |
| **Categories** | 634 Gardiner code classes |
| **mAP Performance** | 31.2% (IoU=0.50:0.95) |
| **Detection Accuracy** | 95% (post category-mapping fix) |
| **TLA Coverage** | 100% (with fallback strategies) |
| **AKU Integration** | Reference signs from Westcar corpus |
| **Validation Interface** | Real-time review |


## Project Structure & File Paths

HieraticAI uses **relative paths** throughout to ensure portability across different systems and users. All paths are resolved relative to the project root directory.

### Required Directory Structure
```
HieraticAI/
├── data/                           # Generated indices and datasets
│   ├── aku_gardiner_index.json     # AKU database index (auto-generated)
│   └── tla_lemma_index.json        # TLA database index (auto-generated)
├── external_data/                  # External database files
│   └── AKU Westcar Scraping/       # AKU Westcar papyrus data
│       ├── json/                   # Metadata files
│       └── svg/                    # Sign vector graphics
├── hieroglyphs_dataset/            # Training dataset
│   ├── train/
│   ├── val/
│   └── test/
├── output/                         # Training outputs and results
│   └── [training_timestamp]/       # Auto-generated training directories
└── tools/                          # Scripts and utilities
    ├── data-collection/            # Hieroglyph scraping toolkit
    │   ├── README.md              # Data collection setup guide
    │   ├── list_signs_modular.py   # Sign list scraper
    │   ├── metadata_scraper_modular.py # Metadata downloader
    │   ├── svg_analyzer_modular.py # Image analysis tool
    │   └── hieroglyph_scraping_toolkit/ # Supporting modules
    └── validation/                 # Validation interface
```

### Path Portability
-  **All code uses relative paths** - works on Windows, macOS, Linux
-  **No hardcoded usernames** - works for any user
-  **Auto-detection of training outputs** - finds most recent results
-  **Helpful error messages** - guides users to missing files

### Fixing Path Issues
If you encounter path-related errors:

1. **Regenerate AKU index** (fixes absolute path issues):
   ```bash
   python regenerate_aku_index.py
   ```

2. **Verify project structure** matches the layout above

3. **Run from project root** - always execute commands from the HieraticAI directory

## Technical Architecture

```mermaid
graph TB
    subgraph "AI Pipeline"
        A[Image Input] --> B[Faster R-CNN Detection]
        B --> C[Gardiner Classification]
        C --> D[Confidence Scoring]
    end
    
    subgraph "Validation Interface"
        D --> E[Streamlit App]
        E --> F[Interactive Review]
        F --> G[Validation]
    end
    
    subgraph "Database Layer"
        G --> H[TLA Integration]
        G --> I[AKU References]
        H --> J[Linguistic Context]
        I --> K[Paleographic Context]
    end
    
    subgraph "Output"
        J --> L[Validated Results]
        K --> L
        L --> M[CSV Export]
        L --> N[Research Data]
    end
    
    style E fill:#e74c3c
    style H fill:#3498db
    style I fill:#f39c12
```

## Testing & Quality Assurance

HieraticAI includes comprehensive testing and automated CI/CD to ensure reliability.

### Running Tests

**Quick start**:
```bash
# Run all tests
bash scripts/run_tests.sh

# Run with pytest (if installed)
pytest tests/ -v

# Run specific test file
python3 tests/test_dataset_validation.py
python3 tests/test_training.py
```

**Test Coverage**:
- Dataset validation tests
- Category remapping tests (prevents off-by-one errors)
- Training configuration tests
- Integration tests for complete pipeline
- Regression prevention tests

### CI/CD Pipeline

Automated testing runs on every commit:
- **Code Quality**: Black, isort, Flake8 linting
- **Multi-Platform Tests**: Ubuntu & macOS, Python 3.8-3.11
- **Security Scanning**: Bandit, Safety vulnerability checks
- **Coverage Reporting**: Minimum 70% code coverage enforced
- **Build Validation**: Package building and verification

See `.github/workflows/ci.yml` for full pipeline configuration.

### Documentation

- **[Getting Started](docs/GETTING_STARTED.md)**: Step-by-step installation and first use
- **[Technical Guide](docs/TECHNICAL_GUIDE.md)**: Advanced usage, customization, and development
- **[Testing Guide](docs/TESTING.md)**: Comprehensive testing documentation

## Acknowledgments

### Academic Context
This prototype was developed as part of the **"Ancient Language Processing" seminar** at **Freie Universität Berlin** (Summer 2025). The course focuses on computational approaches to ancient datasets, digital philology methods, and emerging research questions in ancient Near Eastern studies.

**Course Details:**
- **Institution**: Freie Universität Berlin
- **Course**: Ancient Language Processing 2025
- **Objective**: Methodological exploration of ancient language processing technologies
- **Course Website**: https://digitalpasts.github.io/alp-course-2025/

### Project Status
**Important Note**: This is a **methodological prototype** created for educational and research exploration purposes. It is not intended as a production system but rather as a proof-of-concept for applying modern AI techniques to ancient Egyptian paleographic analysis.

## Contributing

We welcome contributions from Egyptologists, computer vision researchers, and digital humanities scholars, particularly those interested in computational approaches to ancient manuscripts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use HieraticAI in your research, please cite:

```bibtex
@software{belotcolyer2025hieraticai,
  title={HieraticAI: AI-powered hieratic character recognition for ancient Egyptian manuscripts},
  author={Belot, Margot and Colyer, Domino},
  year={2025},
  url={https://github.com/MargotBelot/HieraticAI}
}
```
---
