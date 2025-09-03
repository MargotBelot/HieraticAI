#!/usr/bin/env python3
"""
Setup Enhanced Validation App
=============================

Script to prepare data and run the enhanced validation app with AKU database integration.

Authors: Margot Belot <margotbelot@icloud.com>
         Dominique Colyer <dominic23@zedat.fu-berlin.de>
Date: August 2025
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import subprocess

def create_sample_predictions(output_path: str, num_samples: int = 50):
    """Create sample prediction data for testing."""
    
    # Common Gardiner codes that might exist in AKU database
    gardiner_codes = [
        "A1", "A2", "A21", "B1", "C1", "D1", "D2", "D4", "D21", "D36",
        "E1", "F1", "F18", "F31", "G1", "G4", "G17", "G25", "G36", "G43",
        "H1", "I1", "I9", "I10", "K1", "L1", "M1", "M8", "M17", "M23",
        "N1", "N5", "N14", "N18", "N25", "N35", "N37", "O1", "O4", "O29",
        "P1", "Q1", "Q3", "R1", "R4", "R8", "S1", "S12", "S29", "T1",
        "U1", "U6", "V1", "V4", "V13", "V28", "W1", "W3", "X1", "X8",
        "Y1", "Y3", "Y5", "Z1", "Z2", "Z4", "Z7", "Z11"
    ]
    
    # Generate sample data
    data = []
    
    for i in range(num_samples):
        # Random selection of Gardiner code
        predicted_class = np.random.choice(gardiner_codes)
        
        # Random confidence score (weighted towards higher values)
        confidence = np.random.beta(3, 2)  # Skewed towards higher confidence
        
        # Generate synthetic image path
        image_path = f"test_images/hieroglyph_{i:03d}.jpg"
        
        # Additional metadata
        data.append({
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 3),
            'prediction_id': f"pred_{i:04d}",
            'processing_time_ms': np.random.randint(50, 500),
            'model_version': "v2.1.0",
            'date_processed': f"2025-01-{np.random.randint(1, 31):02d}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by confidence descending
    df = df.sort_values('confidence', ascending=False).reset_index(drop=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f" Generated {len(df)} sample predictions at: {output_path}")
    return df

def run_aku_indexer():
    """Run the AKU data indexer."""
    print(" Running AKU data indexer...")
    
    indexer_script = Path("scripts/aku_data_indexer.py")
    
    if not indexer_script.exists():
        print(f" Indexer script not found: {indexer_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(indexer_script)],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print(" AKU indexer completed successfully")
            print(result.stdout)
            return True
        else:
            print(f" AKU indexer failed with error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f" Error running AKU indexer: {e}")
        return False

def check_aku_data():
    """Check if AKU data is available."""
    aku_path = Path("./external_data/AKU Westcar Scraping")
    
    if not aku_path.exists():
        print(f"  AKU data not found at: {aku_path}")
        print("Please ensure the AKU Westcar Scraping folder is available.")
        return False
    
    json_path = aku_path / "json"
    svg_path = aku_path / "svg"
    
    if not json_path.exists():
        print(f"  JSON folder not found: {json_path}")
        return False
    
    json_files = list(json_path.glob("*.json"))
    svg_files = list(svg_path.glob("*.svg")) if svg_path.exists() else []
    
    print(f" Found {len(json_files)} JSON files and {len(svg_files)} SVG files")
    
    if len(json_files) == 0:
        print(" No JSON files found in AKU database")
        return False
    
    return True

def launch_streamlit_app():
    """Launch the enhanced Streamlit validation app."""
    app_script = Path("streamlit_validation_app_enhanced.py")
    
    if not app_script.exists():
        print(f" Streamlit app not found: {app_script}")
        return False
    
    print(" Launching Streamlit validation app...")
    print("   (This will open in your browser)")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_script),
            "--server.headless", "false"
        ])
        return True
    except Exception as e:
        print(f" Error launching Streamlit app: {e}")
        return False

def main():
    """Main setup function."""
    print(" Setting up Enhanced HieraticAI Validation App")
    print("=" * 50)
    
    # Check if AKU data is available
    if not check_aku_data():
        print("\n Setup cannot continue without AKU data.")
        print("Please ensure the AKU Westcar Scraping folder is properly located.")
        return
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run AKU indexer
    indexer_success = run_aku_indexer()
    
    if not indexer_success:
        print("\n  AKU indexer failed, but continuing with setup...")
    
    # Generate sample prediction data
    sample_predictions_path = "results/predictions_analysis.csv"
    sample_df = create_sample_predictions(sample_predictions_path, num_samples=75)
    
    print("\n Sample predictions summary:")
    print(f"   Total predictions: {len(sample_df)}")
    print(f"   Unique classes: {sample_df['predicted_class'].nunique()}")
    print(f"   Average confidence: {sample_df['confidence'].mean():.3f}")
    print(f"   Top classes: {', '.join(sample_df['predicted_class'].value_counts().head(5).index.tolist())}")
    
    # Check if index was generated
    index_path = Path("data/aku_gardiner_index.json")
    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            gardiner_codes = len(index_data.get('gardiner_index', {}))
            total_records = index_data.get('total_records', 0)
            
            print(f"\n AKU index summary:")
            print(f"   Total records: {total_records}")
            print(f"   Unique Gardiner codes: {gardiner_codes}")
            
        except Exception as e:
            print(f"  Could not read AKU index: {e}")
    
    print("\n Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: streamlit run streamlit_validation_app_enhanced.py")
    print("2. Open the provided URL in your browser")
    print("3. Explore predictions and AKU database integration")
    
    # Ask if user wants to launch the app
    launch = input("\nLaunch the Streamlit app now? (y/n): ").lower().strip()
    
    if launch in ['y', 'yes']:
        launch_streamlit_app()
    else:
        print("\nYou can launch the app later with:")
        print("streamlit run streamlit_validation_app_enhanced.py")

if __name__ == "__main__":
    main()
