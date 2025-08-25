#!/usr/bin/env python3
"""
Interactive web interface for validating HieraticAI model predictions.
Allows users to manually verify predictions, view Gardiner codes, and track validation statistics.
Enriched with AKU Westcar database integration for contextual hieroglyph comparison.

Usage:
    streamlit run scripts/prediction_validator.py

Features:
- Visual prediction display with bounding boxes
- Gardiner code identification
- Manual validation (Correct/Incorrect/Uncertain)
- Real-time validation statistics
- Export validated results
- Confidence-based filtering
- AKU database integration for contextual comparison
- Similar hieroglyph reference display
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
from collections import defaultdict
import re
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="HieraticAI Prediction Validator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PredictionValidator:
    def __init__(self):
        self.project_root = Path(".")
        self.predictions_file = self.project_root / "output"/ "improved_training_20250822_200344"/ "coco_instances_results_FIXED.json"
        self.validation_file = self.project_root / "output"/ "validation_results.json"
        self.images_dir = self.project_root / "hieroglyphs_dataset"
        
        # AKU database paths
        self.aku_index_path = self.project_root / "data"/ "aku_gardiner_index.json"
        self.aku_data_path = Path("./external_data/AKU Westcar Scraping")
        
        # TLA lemma database paths
        self.tla_index_path = self.project_root / "data"/ "tla_lemma_index.json"
        
        # Load AKU and TLA indices on initialization
        self.aku_index = self.load_aku_index()
        self.tla_index = self.load_tla_index()
        
        # Gardiner code mapping with CORRECTED Unicode values from official Unikemet file
        self.gardiner_codes = {
            # Existing codes (corrected Unicode values)
            "A14": {"name": "man with arms in adoration", "category": "Man and his occupations", "unicode": "U+13010"},
            "A1B": {"name": "man seated", "category": "Man and his occupations", "unicode": "U+13000"},
            "A23A": {"name": "king with was-sceptre", "category": "Man and his occupations", "unicode": "U+13668"},
            "A49": {"name": "Syrian", "category": "Man and his occupations", "unicode": "U+1355D"},
            "C9": {"name": "Hathor", "category": "Anthropomorphic deities", "unicode": "U+1387D"},
            "D200": {"name": "eye-paint", "category": "Parts of the human body", "unicode": "U+139AD"},
            "D35": {"name": "arms in negation", "category": "Parts of the human body", "unicode": "U+1309C"},
            "D36": {"name": "forearm", "category": "Parts of the human body", "unicode": "U+1309D"},
            "D3B": {"name": "forearm with bread", "category": "Parts of the human body", "unicode": "U+1396C"},
            "D4": {"name": "eye", "category": "Parts of the human body", "unicode": "U+13079"},
            "D40": {"name": "forearm with stick", "category": "Parts of the human body", "unicode": "U+130A1"},
            "D45": {"name": "arm with wand", "category": "Parts of the human body", "unicode": "U+130A6"},
            "D53": {"name": "phallus", "category": "Parts of the human body", "unicode": "U+130BA"},
            "E7": {"name": "donkey", "category": "Mammals", "unicode": "U+130D8"},
            "F9": {"name": "leopard's head", "category": "Parts of mammals", "unicode": "U+13107"},
            "G14": {"name": "vulture", "category": "Birds", "unicode": "U+13150"},
            "G32": {"name": "heron", "category": "Birds", "unicode": "U+13164"},
            "G40": {"name": "pintail duck", "category": "Birds", "unicode": "U+1316E"},
            "G41": {"name": "pintail duck flying", "category": "Birds", "unicode": "U+1316F"},
            "G7": {"name": "falcon", "category": "Birds", "unicode": "U+13146"},
            "I1": {"name": "gecko", "category": "Amphibious animals, reptiles", "unicode": "U+13188"},
            "I87": {"name": "fish", "category": "Fish", "unicode": "U+13DC8"},
            "L1": {"name": "dung beetle", "category": "Invertebrates", "unicode": "U+131A3"},
            "M14": {"name": "papyrus", "category": "Trees and plants", "unicode": "U+131C6"},
            "M17": {"name": "reed", "category": "Trees and plants", "unicode": "U+131CB"},
            "M22A": {"name": "rush", "category": "Trees and plants", "unicode": "U+13E8C"},
            "M29": {"name": "pod", "category": "Trees and plants", "unicode": "U+131DB"},
            "M5": {"name": "tree", "category": "Trees and plants", "unicode": "U+131B4"},
            "N33": {"name": "grain of sand", "category": "Sky, earth, water", "unicode": "U+13212"},
            "N90": {"name": "water ripple", "category": "Sky, earth, water", "unicode": "U+132E2"},
            "O34": {"name": "bolt", "category": "Buildings and their parts", "unicode": "U+13283"},
            "P6": {"name": "mast", "category": "Ships and parts of ships", "unicode": "U+132A2"},
            "Q19": {"name": "hill over hill", "category": "Domestic and funerary furniture", "unicode": "U+140AB"},
            "R10": {"name": "festival", "category": "Temple furniture and sacred emblems", "unicode": "U+132BB"},
            "S28": {"name": "linen", "category": "Crowns, dress, staves", "unicode": "U+132F3"},
            "S33": {"name": "sandal", "category": "Crowns, dress, staves", "unicode": "U+132F8"},
            "U28": {"name": "fire-drill", "category": "Agriculture, crafts, and professions", "unicode": "U+13351"},
            "U30": {"name": "potter's kiln", "category": "Agriculture, crafts, and professions", "unicode": "U+13354"},
            "U32": {"name": "pestle", "category": "Agriculture, crafts, and professions", "unicode": "U+13356"},
            "U35": {"name": "knife-sharpener", "category": "Agriculture, crafts, and professions", "unicode": "U+1335A"},
            "U9": {"name": "grain-measure", "category": "Agriculture, crafts, and professions", "unicode": "U+1333D"},
            "V102": {"name": "cord", "category": "Rope, fibre, baskets", "unicode": "U+14356"},
            "V19": {"name": "hobble", "category": "Rope, fibre, baskets", "unicode": "U+13385"},
            "V27": {"name": "wick", "category": "Rope, fibre, baskets", "unicode": "U+1339A"},
            "V29A": {"name": "swab", "category": "Rope, fibre, baskets", "unicode": "U+1433F"},
            "V30": {"name": "wickerwork basket", "category": "Rope, fibre, baskets", "unicode": "U+1339F"},
            "V39": {"name": "tethering rope", "category": "Rope, fibre, baskets", "unicode": "U+133AC"},
            "W24": {"name": "bowl", "category": "Vessels of stone and earthenware", "unicode": "U+133CC"},
            "W9": {"name": "stone bowl", "category": "Vessels of stone and earthenware", "unicode": "U+133B8"},
            "X8": {"name": "bread loaf", "category": "Loaves and cakes", "unicode": "U+133D9"},
            "Y4": {"name": "scribe's palette", "category": "Writings, games, music", "unicode": "U+133DF"},
            "Y5": {"name": "draught-board", "category": "Writings, games, music", "unicode": "U+133E0"},
            "Z3A": {"name": "three strokes", "category": "Strokes", "unicode": "U+133EB"},
            "Z9": {"name": "two diagonal strokes", "category": "Strokes", "unicode": "U+133F4"},
            "A32h": {"name": "dancing man", "category": "Man and his occupations", "unicode": "U+135DF"},
            
            # Additional codes found in predictions (with correct Unicode values from official file)
            "A1": {"name": "man", "category": "Man and his occupations", "unicode": "U+13000"},
            "A2": {"name": "man with hand to mouth", "category": "Man and his occupations", "unicode": "U+13001"},
            "B1": {"name": "woman", "category": "Woman and her occupations", "unicode": "U+13050"},
            "D1": {"name": "head in profile", "category": "Parts of the human body", "unicode": "U+13076"},
            "D21": {"name": "mouth", "category": "Parts of the human body", "unicode": "U+1308B"},
            "D46": {"name": "hand", "category": "Parts of the human body", "unicode": "U+130A7"},
            "E8": {"name": "kid", "category": "Mammals", "unicode": "U+130D9"},
            "G1": {"name": "Egyptian vulture", "category": "Birds", "unicode": "U+1313F"},
            "G17": {"name": "owl", "category": "Birds", "unicode": "U+13153"},
            "I10": {"name": "cobra", "category": "Amphibious animals, reptiles", "unicode": "U+13193"},
            "I9": {"name": "horned viper", "category": "Amphibious animals, reptiles", "unicode": "U+13191"},
            "M15": {"name": "clump of papyrus", "category": "Trees and plants", "unicode": "U+131C7"},
            "M18": {"name": "clump of papyrus with flowers", "category": "Trees and plants", "unicode": "U+131CD"},
            "M23": {"name": "sedge", "category": "Trees and plants", "unicode": "U+131D3"},
            "N35": {"name": "ripple of water", "category": "Sky, earth, water", "unicode": "U+13216"},
            "O1": {"name": "house", "category": "Buildings and their parts", "unicode": "U+13250"},
            "Q3": {"name": "stool", "category": "Domestic and funerary furniture", "unicode": "U+132AA"},
            "R11": {"name": "column", "category": "Temple furniture and sacred emblems", "unicode": "U+132BD"},
            "S24": {"name": "girdle knot", "category": "Crowns, dress, staves", "unicode": "U+132ED"},
            "S29": {"name": "folded cloth", "category": "Crowns, dress, staves", "unicode": "U+132F4"},
            "U31": {"name": "knife sharpener", "category": "Agriculture, crafts, and professions", "unicode": "U+13355"},
            "U33": {"name": "pestle and mortar", "category": "Agriculture, crafts, and professions", "unicode": "U+13358"},
            "V1": {"name": "coil of rope", "category": "Rope, fibre, baskets", "unicode": "U+13362"},
            "V2": {"name": "rope", "category": "Rope, fibre, baskets", "unicode": "U+1336C"},
            "V31": {"name": "basket with handle", "category": "Rope, fibre, baskets", "unicode": "U+133A1"},
            "W25": {"name": "jar with handles", "category": "Vessels of stone and earthenware", "unicode": "U+133CE"},
            "X1": {"name": "loaf of bread", "category": "Loaves and cakes", "unicode": "U+133CF"},
            "Y1": {"name": "papyrus roll", "category": "Writings, games, music", "unicode": "U+133DB"},
            "Z4": {"name": "two diagonal strokes", "category": "Strokes", "unicode": "U+133ED"},
            "Aa1": {"name": "placenta", "category": "Unclassified", "unicode": "U+1340D"},
            "Aa15": {"name": "part of human body", "category": "Unclassified", "unicode": "U+1341D"}
        }

    def load_predictions(self):
        """Load model predictions from file."""
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        else:
            st.error(f"Predictions file not found: {self.predictions_file}")
            return []

    def load_validation_results(self):
        """Load existing validation results."""
        if self.validation_file.exists():
            with open(self.validation_file, 'r') as f:
                return json.load(f)
        return {"validated_predictions": {}, "session_stats": {}}

    def save_validation_results(self, validation_data):
        """Save validation results to file."""
        self.validation_file.parent.mkdir(exist_ok=True)
        with open(self.validation_file, 'w') as f:
            json.dump(validation_data, f, indent=2)
    
    @st.cache_data
    def load_aku_index(_self):
        """Load the AKU database index."""
        try:
            if _self.aku_index_path.exists():
                with open(_self.aku_index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Try to generate index if it doesn't exist
                st.warning(f"AKU index not found at {_self.aku_index_path}. Consider running the indexer.")
                return {}
        except Exception as e:
            st.warning(f"Could not load AKU index: {e}")
            return {}
    
    @st.cache_data
    def load_tla_index(_self):
        """Load the TLA lemma index."""
        try:
            if _self.tla_index_path.exists():
                with open(_self.tla_index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                st.warning(f"TLA index not found at {_self.tla_index_path}. Run the TLA indexer to generate it.")
                return {}
        except Exception as e:
            st.warning(f"Could not load TLA index: {e}")
            return {}
    
    def get_similar_aku_signs(self, gardiner_code: str, max_results: int = 3) -> List[Dict]:
        """Get similar signs from AKU database for a given Gardiner code."""
        if not self.aku_index:
            return []
        
        gardiner_index = self.aku_index.get('gardiner_index', {})
        similar_signs = gardiner_index.get(gardiner_code, [])
        
        if not similar_signs:
            return []
        
        # Sort by quality indicators (SVG available, readable, complete)
        sorted_signs = sorted(similar_signs, key=lambda x: (
            x.get('svg_path') is not None,  # SVG available
            x.get('metadata', {}).get('Lesbarkeit', '') == 'ja',  # Readable
            x.get('metadata', {}).get('Zustand', '') == 'vollständig',  # Complete
        ), reverse=True)
        
        return sorted_signs[:max_results]
    
    def get_hieroglyph_from_gardiner(self, gardiner_code: str) -> Optional[str]:
        """Convert Gardiner code to Unicode hieroglyph character."""
        gardiner_info = self.gardiner_codes.get(gardiner_code, {})
        unicode_point = gardiner_info.get('unicode', '')
        
        if unicode_point and unicode_point.startswith('U+'):
            try:
                return chr(int(unicode_point[2:], 16))
            except ValueError:
                return None
        return None
    
    def get_tla_lemmas_for_sign(self, gardiner_code: str) -> List[Dict]:
        """Get TLA lemma information with 100% coverage through fallback strategies."""
        # Strategy 1: Try direct TLA match
        hieroglyph = self.get_hieroglyph_from_gardiner(gardiner_code)
        if hieroglyph and self.tla_index:
            hieroglyph_index = self.tla_index.get('by_hieroglyph', self.tla_index.get('hieroglyph_index', {}))
            lemmas = hieroglyph_index.get(hieroglyph, [])
            if lemmas:
                return sorted(lemmas, key=lambda x: x.get('frequency', 0), reverse=True)
        
        # Strategy 2: Try fallback mapping to similar signs
        fallback_mappings = {
            'A23A': 'A23', 'D3B': 'D3', 'V29A': 'V29', 'A32h': 'A32',
            'D200': 'D4', 'V102': 'V1', 'I87': 'I3', 'M14': 'M15',
            'M15': 'M17', 'M18': 'M17', 'G7': 'G1', 'G32': 'G14',
            'E7': 'E8', 'U30': 'U28', 'U32': 'U31', 'U35': 'U33',
            'V27': 'V1', 'V39': 'V1'
        }
        
        if gardiner_code in fallback_mappings:
            fallback_code = fallback_mappings[gardiner_code]
            fallback_hieroglyph = self.get_hieroglyph_from_gardiner(fallback_code)
            if fallback_hieroglyph and self.tla_index:
                hieroglyph_index = self.tla_index.get('by_hieroglyph', {})
                lemmas = hieroglyph_index.get(fallback_hieroglyph, [])
                if lemmas:
                    # Add fallback indicator
                    enhanced_lemmas = []
                    for lemma in lemmas:
                        enhanced_lemma = lemma.copy()
                        enhanced_lemma['fallback_note'] = f"Data from similar sign {fallback_code}"
                        enhanced_lemmas.append(enhanced_lemma)
                    return sorted(enhanced_lemmas, key=lambda x: x.get('frequency', 0), reverse=True)
        
        # Strategy 3: Manual entries for signs not in TLA
        manual_entries = {
            'A14': [{'hieroglyphs': '', 'transliteration': 'dw', 'translation': 'To worship; to adore', 'frequency': 0, 'lemma_id': 'MANUAL_A14', 'source': 'Gardiner + Manual'}],
            'A49': [{'hieroglyphs': '', 'transliteration': 'mw', 'translation': 'Asiatic; Syrian (determinative)', 'frequency': 0, 'lemma_id': 'MANUAL_A49', 'source': 'Gardiner + Manual'}],
            'C9': [{'hieroglyphs': '', 'transliteration': 'Ḥwt-Ḥr', 'translation': 'Hathor (goddess determinative)', 'frequency': 0, 'lemma_id': 'MANUAL_C9', 'source': 'Gardiner + Manual'}],
            'F9': [{'hieroglyphs': '', 'transliteration': 'pẖr', 'translation': 'Leopard (determinative)', 'frequency': 0, 'lemma_id': 'MANUAL_F9', 'source': 'Gardiner + Manual'}],
            'I1': [{'hieroglyphs': '', 'transliteration': 'sf', 'translation': 'Gecko; lizard (determinative)', 'frequency': 0, 'lemma_id': 'MANUAL_I1', 'source': 'Gardiner + Manual'}],
            'M5': [{'hieroglyphs': '', 'transliteration': 'jšd', 'translation': 'Tree; wood (determinative)', 'frequency': 0, 'lemma_id': 'MANUAL_M5', 'source': 'Gardiner + Manual'}],
            'N90': [{'hieroglyphs': '', 'transliteration': 'mw', 'translation': 'Water; liquid (determinative)', 'frequency': 0, 'lemma_id': 'MANUAL_N90', 'source': 'Gardiner + Manual'}],
            'Q19': [{'hieroglyphs': '', 'transliteration': 'qrrt', 'translation': 'Hill; mound (determinative)', 'frequency': 0, 'lemma_id': 'MANUAL_Q19', 'source': 'Gardiner + Manual'}],
            'R10': [{'hieroglyphs': '', 'transliteration': 'ḥb', 'translation': 'Festival; celebration', 'frequency': 0, 'lemma_id': 'MANUAL_R10', 'source': 'Gardiner + Manual'}],
            'S33': [{'hieroglyphs': '', 'transliteration': 'tbw.t', 'translation': 'Sandal; shoe', 'frequency': 0, 'lemma_id': 'MANUAL_S33', 'source': 'Gardiner + Manual'}],
            'U9': [{'hieroglyphs': '', 'transliteration': 'hnw', 'translation': 'Jar; grain measure', 'frequency': 0, 'lemma_id': 'MANUAL_U9', 'source': 'Gardiner + Manual'}],
            'Y4': [{'hieroglyphs': '', 'transliteration': 'mn', 'translation': 'Gaming piece; draughtsman', 'frequency': 0, 'lemma_id': 'MANUAL_Y4', 'source': 'Gardiner + Manual'}],
            'Z3A': [{'hieroglyphs': '', 'transliteration': '3', 'translation': 'Three (strokes); plural marker', 'frequency': 0, 'lemma_id': 'MANUAL_Z3A', 'source': 'Gardiner + Manual'}],
            'D45': [{'hieroglyphs': '', 'transliteration': 'ḏsr', 'translation': 'Arm with sacred objects', 'frequency': 0, 'lemma_id': 'MANUAL_D45', 'source': 'Gardiner + Manual'}],
            'M22A': [{'hieroglyphs': '', 'transliteration': 'nn', 'translation': 'Rush; reed plant (determinative)', 'frequency': 0, 'lemma_id': 'MANUAL_M22A', 'source': 'Gardiner + Manual'}]
        }
        
        if gardiner_code in manual_entries:
            return manual_entries[gardiner_code]
        
        # Strategy 4: Create basic fallback from Gardiner info
        gardiner_info = self.gardiner_codes.get(gardiner_code, {})
        if gardiner_info:
            return [{
                'hieroglyphs': hieroglyph or gardiner_code,
                'transliteration': f"[{gardiner_code.lower()}]",
                'translation': f"{gardiner_info.get('name', 'Unknown sign')} (Gardiner classification)",
                'frequency': 0,
                'lemma_id': f'FALLBACK_{gardiner_code}',
                'source': 'Gardiner Fallback',
                'note': 'No TLA data available - using Gardiner classification'
            }]
        
        return []
    
    def get_svg_as_html(self, svg_path: str, width: int = 80) -> str:
        """Convert SVG to HTML for display."""
        try:
            # Handle both absolute paths from index and relative paths
            if not Path(svg_path).exists():
                # Try to construct the path relative to our project
                # Extract filename from the original path
                filename = Path(svg_path).name
                local_svg_path = self.aku_data_path / "svg" / filename
                
                if local_svg_path.exists():
                    svg_path = str(local_svg_path)
                else:
                    return f'<div style="color: red;">SVG file not found: {filename}</div>'
            
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # Inject width if not present
            if 'width=' not in svg_content:
                svg_content = svg_content.replace('<svg', f'<svg width="{width}"')
            
            return svg_content
        except Exception as e:
            return f'<div style="color: red;">Error loading SVG: {e}</div>'
    
    def display_aku_reference_signs(self, gardiner_code: str, container):
        """Display AKU reference signs for a given Gardiner code."""
        with container:
            similar_signs = self.get_similar_aku_signs(gardiner_code, max_results=10)
            
            if similar_signs:
                st.markdown(f"### AKU Reference Signs for `{gardiner_code}`")
                st.markdown(f"*Found {len(similar_signs)} similar signs in AKU Westcar database*")
                
                for i, sign in enumerate(similar_signs):
                    with st.container():
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Display SVG if available
                            if sign.get('svg_path'):
                                svg_html = self.get_svg_as_html(sign['svg_path'], width=60)
                                st.components.v1.html(svg_html, height=80)
                            else:
                                st.info("No SVG available")
                        
                        with col2:
                            # Display metadata
                            metadata = sign.get('metadata', {})
                            
                            st.write(f"**AKU-Nr:** {sign.get('aku_nr', 'N/A')}")
                            
                            description = metadata.get('Beschreibung', 'No description')
                            if len(description) > 50:
                                st.write(f"**Description:** {description[:50]}...")
                            else:
                                st.write(f"**Description:** {description}")
                            
                            # Quality indicators
                            quality_indicators = []
                            if metadata.get('Lesbarkeit') == 'ja':
                                quality_indicators.append("Readable")
                            if metadata.get('Zustand') == 'vollständig':
                                quality_indicators.append("Complete")
                            if metadata.get('Farbe') == 'schwarz':
                                quality_indicators.append("Clear ink")
                            
                            if quality_indicators:
                                st.write("".join(quality_indicators))
                            
                            # Show additional metadata in expander
                            with st.expander(f"Full metadata #{i+1}"):
                                for key, value in metadata.items():
                                    if value and str(value).strip():
                                        st.write(f"**{key}:** {value}")
                        
                        if i < len(similar_signs) - 1:
                            st.markdown("---")
                
                # Show database statistics for this code
                stats = self.aku_index.get('statistics', {}).get(gardiner_code, {})
                if stats:
                    st.markdown(f"**Database Stats:** {stats.get('total', 0)} total, "
                              f"{stats.get('with_svg', 0)} with SVG, "
                              f"{stats.get('readable', 0)} readable")
            
            else:
                st.markdown(f"### AKU Reference Signs for `{gardiner_code}`")
                st.info(f"No similar signs found in AKU database for `{gardiner_code}`.")
                
                # Show some available codes for reference
                if self.aku_index:
                    available_codes = list(self.aku_index.get('gardiner_index', {}).keys())[:10]
                    if available_codes:
                        st.write("Available codes (sample): "+ ", ".join(available_codes))
    
    def display_tla_lemma_info(self, gardiner_code: str, container):
        """Display TLA lemma information for a given Gardiner code."""
        with container:
            lemmas = self.get_tla_lemmas_for_sign(gardiner_code)
            
            if lemmas:
                st.markdown(f"### TLA Lemma Information for `{gardiner_code}`")
                st.markdown(f"*Found {len(lemmas)} related lemmas in TLA database (showing top 5)*")
                
                # Show only top 5 most frequent lemmas
                for i, lemma in enumerate(lemmas[:5]):
                    with st.container():
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Display hieroglyphs prominently if available
                            hieroglyphs = lemma.get('hieroglyphs', '')
                            if hieroglyphs:
                                st.markdown(f"**{hieroglyphs}**")
                                st.caption(f"Freq: {lemma.get('frequency', 0)}")
                            else:
                                st.info("No hieroglyphs")
                        
                        with col2:
                            # Display transliteration and translation
                            transliteration = lemma.get('transliteration', '')
                            translation = lemma.get('translation', '')
                            
                            if transliteration:
                                st.markdown(f"**`{transliteration}`**")
                            
                            if translation and not translation.startswith('English translation not found'):
                                st.write(f"**Translation:** {translation}")
                            elif translation.startswith('English translation not found'):
                                st.write(f"**Translation:** *(not available)*")
                            
                            # Show lemma ID for reference in smaller text
                            lemma_id = lemma.get('lemma_id') or lemma.get('id')
                            if lemma_id:
                                st.caption(f"TLA ID: {lemma_id}")
                        
                        if i < min(len(lemmas), 5) - 1:
                            st.markdown("---")
                
                # Show database statistics for this hieroglyph
                hieroglyph = self.get_hieroglyph_from_gardiner(gardiner_code)
                if hieroglyph and self.tla_index:
                    by_hieroglyph = self.tla_index.get('by_hieroglyph', {})
                    total_signs = len(by_hieroglyph)
                    st.markdown(f"**TLA Database Stats:** {len(lemmas)} lemmas for this sign, {total_signs} total signs indexed")
            
            else:
                st.markdown(f"### TLA Lemma Information for `{gardiner_code}`")
                hieroglyph = self.get_hieroglyph_from_gardiner(gardiner_code)
                if hieroglyph:
                    st.info(f"No lemma information found for hieroglyph '{hieroglyph}' ({gardiner_code}) in TLA database.")
                else:
                    st.info(f"Could not convert Gardiner code '{gardiner_code}' to Unicode hieroglyph for TLA lookup.")

    def get_gardiner_info(self, category_id):
        """Get Gardiner code information for a category."""
        # Mapping from model category IDs to actual Gardiner codes (from actual FIXED predictions)
        gardiner_map = {
            1: "A1",
            14: "A2",
            55: "B1",
            76: "D1",
            87: "D21",
            102: "D35",
            109: "D40",
            114: "D46",
            159: "E8",
            209: "G1",
            211: "G17",
            258: "I10",
            274: "I9",
            292: "M15",
            294: "M17",
            295: "M18",
            305: "M23",
            353: "N35",
            371: "O1",
            416: "P6",
            422: "Q3",
            430: "R11",
            459: "S24",
            462: "S29",
            525: "U31",
            527: "U33",
            537: "V1",
            546: "V2",
            561: "V31",
            587: "W25",
            595: "X1",
            602: "Y1",
            606: "Y5",
            615: "Z3A",
            616: "Z4",
            621: "Aa1",
            629: "G7",
            630: "U28",
            631: "Aa15",
        }
        
        code = gardiner_map.get(category_id, f"UNKNOWN_{category_id}")
        info = self.gardiner_codes.get(code, {"name": f"unknown category {category_id}", "category": "Unknown", "unicode": "N/A"})
        return code, info

    def visualize_predictions(self, image_path, predictions, image_id, validated_results):
        """Create visualization of predictions on image."""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            st.error(f"Could not load image: {image_path}")
            return None, []
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(img_rgb)
        ax.set_title(f"HieraticAI Predictions on Westcar Papyrus VIII 5-24 ({image_path.name})", fontsize=14, fontweight='bold')
        
        prediction_data = []
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, pred in enumerate(predictions):
            bbox = pred['bbox']  # [x, y, width, height]
            confidence = pred['score']
            category_id = pred['category_id']
            
            # Get Gardiner code
            gardiner_code, gardiner_info = self.get_gardiner_info(category_id)
            
            # Check if already validated
            pred_key = f"{image_id}_{i}"
            validation_status = validated_results.get(pred_key, "pending")
            
            # Choose color based on validation status - make sure colors match the status
            if validation_status == "correct":
                color = 'lime'  # Bright green for correct
                alpha = 0.9
            elif validation_status == "incorrect":
                color = 'red'  # Red for incorrect
                alpha = 0.9
            elif validation_status == "uncertain":
                color = 'orange'  # Orange for uncertain
                alpha = 0.9
            else:
                color = 'blue'  # Blue for pending validation
                alpha = 0.7
            
            # Draw bounding box
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=color, facecolor='none', alpha=alpha
            )
            ax.add_patch(rect)
            
            # Add label with prediction number, Gardiner code, and confidence
            prediction_number = i + 1
            label_text = f"#{prediction_number}\n{gardiner_code}\n{confidence:.2f}"
            ax.text(bbox[0], bbox[1]-15, label_text, 
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                   color='white', ha='left', va='top')
            
            # Store prediction data
            prediction_data.append({
                'index': i,
                'gardiner_code': gardiner_code,
                'gardiner_name': gardiner_info['name'],
                'category': gardiner_info['category'],
                'confidence': confidence,
                'bbox': bbox,
                'validation_status': validation_status,
                'pred_key': pred_key
            })
        
        ax.axis('off')
        plt.tight_layout()
        
        return fig, prediction_data

def main():
    st.title("HieraticAI Prediction Validator")
    st.markdown("**Interactive validation interface for hieratic character predictions on the Westcar Papyrus**")
    
    # Add usage instructions
    with st.expander("How to use this interface", expanded=False):
        st.markdown("""
        ### Getting Started
        1. **View Predictions**: The left panel shows your HieraticAI model's predictions overlaid on the Westcar Papyrus (VIII 5-24)
        2. **Select Signs**: Use the dropdown in the right panel to select individual predictions for validation
        3. **Review Context**: Each prediction shows:
           - Cropped image of the detected sign
           - Gardiner code classification and Unicode character
           - TLA lemma data (transliteration, translation, frequency)
           - AKU reference signs from the Westcar manuscript database
        4. **Validate**: Mark each prediction as Correct, Incorrect, or Uncertain
        5. **Track Progress**: Monitor validation statistics and export results
        
        ### Understanding the Display
        - **Confidence Threshold**: Use the sidebar slider to filter predictions by confidence level
        - **Database Status**: Check sidebar for AKU and TLA database connectivity
        - **Bounding Boxes**: Color-coded rectangles around detected signs show validation status
        
        ### TLA Integration Features
        - **Complete Coverage**: Every Gardiner code has associated linguistic data
        - **Fallback System**: Missing signs use similar signs or manual entries
        - **Data Sources**: Clearly marked whether data comes from TLA, similar signs, or fallback
        """)
    
    validator = PredictionValidator()
    
    # Load data
    predictions_data = validator.load_predictions()
    validation_results = validator.load_validation_results()
    validated_predictions = validation_results.get("validated_predictions", {})
    
    if not predictions_data:
        st.warning("No predictions found. Please run inference first.")
        return
    
    # Sidebar controls
    st.sidebar.header("Validation Controls")
    
    # AKU Database status
    st.sidebar.markdown("### AKU Database")
    if validator.aku_index:
        total_aku_records = validator.aku_index.get('total_records', 0)
        total_gardiner_codes = validator.aku_index.get('total_gardiner_codes', 0)
        st.sidebar.success(f"Connected: {total_aku_records} records, {total_gardiner_codes} codes")
    else:
        st.sidebar.warning("Not available - Run indexer to enable")
    
    # TLA Database status
    st.sidebar.markdown("### TLA Database")
    if validator.tla_index:
        # Handle both possible index structures
        by_hieroglyph = validator.tla_index.get('by_hieroglyph', {})
        by_lemma = validator.tla_index.get('by_lemma_id', {})
        total_tla_signs = len(by_hieroglyph)
        total_tla_lemmas = len(by_lemma) if by_lemma else sum(len(lemmas) for lemmas in by_hieroglyph.values())
        st.sidebar.success(f"Connected: {total_tla_lemmas} lemmas, {total_tla_signs} signs")
    else:
        st.sidebar.warning("Not available - Run TLA indexer to enable")
    
    # Use the first available image automatically
    available_images = list(set([pred.get('image_id', 'unknown') for pred in predictions_data]))
    if not available_images:
        st.warning("No predictions found.")
        return
    
    selected_image = available_images[0]  # Use first image automatically
    
    # Confidence threshold
    min_confidence = st.sidebar.slider(
        "Minimum Confidence", 
        min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )
    
    # Filter predictions for selected image
    image_predictions = [
        pred for pred in predictions_data 
        if pred.get('image_id') == selected_image and pred.get('score', 0) >= min_confidence
    ]
    
    if not image_predictions:
        st.warning(f"No predictions found for image {selected_image} with confidence >= {min_confidence}")
        return
    
    # Load image ID to filename mapping from annotations and find the image
    image_path = None
    patch_name = f"patch_{selected_image-1:04d}.png"# Default fallback
    
    # Try each split to find both the mapping and the actual image file
    # Prioritize test split as it has the most comprehensive annotations
    for split in ['test', 'val', 'train']:
        try:
            # Load annotations for this split
            with open(validator.images_dir / split / 'annotations.json', 'r') as f:
                annotations = json.load(f)
            
            # Check if our image ID is in this split
            split_image_map = {img_info['id']: img_info['file_name'] for img_info in annotations['images']}
            
            if selected_image in split_image_map:
                patch_name = split_image_map[selected_image]
                potential_path = validator.images_dir / split / 'images' / patch_name
                if potential_path.exists():
                    image_path = potential_path
                    break
        except:
            continue
    
    if image_path is None:
        st.error(f"Image file not found for {selected_image}")
        return
    
    
    # Display image with predictions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Prediction Visualization")
        fig, prediction_data = validator.visualize_predictions(
            image_path, image_predictions, selected_image, validated_predictions
        )
        
        if fig:
            st.pyplot(fig)
            plt.close()  # Clean up memory
    
    with col2:
        st.subheader("Validation Panel")
        
        # Show validation summary
        total_preds = len(prediction_data)
        validated_preds = len([p for p in prediction_data if p['validation_status'] != 'pending'])
        
        st.info(f"**{validated_preds}/{total_preds}** predictions validated")
        
        # Instructions
        st.markdown("""
        **Validation Instructions:**
        
        Select a prediction from the dropdown below to validate it. Review the cropped sign image, 
        Gardiner classification, and linguistic context from TLA and AKU databases before making your decision.
        
        The bounding box colors indicate:
        - 🔵 **Blue**: Pending validation
        - 🟢 **Green**: Validated as correct
        - 🔴 **Red**: Validated as incorrect  
        - 🟠 **Orange**: Validated as uncertain
        """)
        
        # Selected prediction for validation
        if 'selected_prediction' not in st.session_state:
            st.session_state.selected_prediction = None
        
        # Selection dropdown for better UX
        if prediction_data:
            st.markdown("### Select Prediction to Validate")
            
            # Create options for dropdown
            options = []
            for i, pred_info in enumerate(prediction_data):
                status_display = {
                    'pending': 'PENDING',
                    'correct': 'CORRECT',
                    'incorrect': 'INCORRECT',
                    'uncertain': 'UNCERTAIN'
                }.get(pred_info['validation_status'], 'PENDING')
                
                options.append(f"[{status_display}] {i+1}. {pred_info['gardiner_code']} (conf: {pred_info['confidence']:.2f})")
            
            selected_option = st.selectbox(
                "Choose a prediction:",
                options=["Select a prediction..."] + options,
                key="prediction_selector"
            )
            
            if selected_option != "Select a prediction...":
                # Extract the prediction index
                pred_index = int(selected_option.split('.')[0].split()[-1]) - 1
                pred_info = prediction_data[pred_index]
                
                st.markdown("---")
                st.markdown(f"### Validating Prediction #{pred_index + 1}")
                
                # Display cropped image of the selected sign
                try:
                    # Load the original image
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Get bounding box coordinates
                        bbox = pred_info['bbox']
                        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        
                        # Add some padding around the crop
                        padding = 10
                        x_start = max(0, x - padding)
                        y_start = max(0, y - padding)
                        x_end = min(img_rgb.shape[1], x + w + padding)
                        y_end = min(img_rgb.shape[0], y + h + padding)
                        
                        # Crop the image
                        cropped_sign = img_rgb[y_start:y_end, x_start:x_end]
                        
                        if cropped_sign.size > 0:
                            st.image(cropped_sign, caption=f"Sign #{pred_index + 1}: {pred_info['gardiner_code']}", width=150)
                except Exception as e:
                    st.write(f"Could not display cropped image: {e}")
                
                # Display Gardiner code with Unicode
                unicode_point = validator.gardiner_codes.get(pred_info['gardiner_code'], {}).get('unicode', 'N/A')
                if unicode_point != 'N/A' and unicode_point.startswith('U+'):
                    try:
                        unicode_char = chr(int(unicode_point[2:], 16))
                        st.markdown(f"**Gardiner Code:** `{pred_info['gardiner_code']}` → **{unicode_char}** ({unicode_point})")
                    except:
                        st.markdown(f"**Gardiner Code:** `{pred_info['gardiner_code']}` ({unicode_point})")
                else:
                    st.markdown(f"**Gardiner Code:** `{pred_info['gardiner_code']}`")
                
                st.write(f"**Name:** {pred_info['gardiner_name']}")
                st.write(f"**Category:** {pred_info['category']}")
                st.write(f"**Confidence:** {pred_info['confidence']:.3f}")
                st.write(f"**Bounding Box:** [{pred_info['bbox'][0]:.1f}, {pred_info['bbox'][1]:.1f}, {pred_info['bbox'][2]:.1f}, {pred_info['bbox'][3]:.1f}]")
                
                # Current validation status
                current_status = pred_info['validation_status']
                if current_status != "pending":
                    status_display = {
                        'correct': 'Correct',
                        'incorrect': 'Incorrect',
                        'uncertain': 'Uncertain'
                    }.get(current_status, current_status)
                    st.write(f"**Current Status:** {status_display}")
                else:
                    st.write(f"**Current Status:** Pending validation")
                
                # Display TLA lemma information
                st.markdown("---")
                tla_container = st.container()
                validator.display_tla_lemma_info(pred_info['gardiner_code'], tla_container)
                
                # Display AKU reference signs
                st.markdown("---")
                aku_container = st.container()
                validator.display_aku_reference_signs(pred_info['gardiner_code'], aku_container)
                
                st.markdown("---")
                st.markdown("### Validate this prediction:")
                
                # Validation buttons
                col_a, col_b, col_c = st.columns([1, 1, 1])
                
                with col_a:
                    if st.button("Correct", key=f"correct_{pred_info['pred_key']}", use_container_width=True, type="primary"):
                        validated_predictions[pred_info['pred_key']] = "correct"
                        validation_results["validated_predictions"] = validated_predictions
                        validator.save_validation_results(validation_results)
                        st.success(f"Marked {pred_info['gardiner_code']} as correct!")
                        st.rerun()
                
                with col_b:
                    if st.button("Incorrect", key=f"incorrect_{pred_info['pred_key']}", use_container_width=True):
                        validated_predictions[pred_info['pred_key']] = "incorrect"
                        validation_results["validated_predictions"] = validated_predictions
                        validator.save_validation_results(validation_results)
                        st.error(f"Marked {pred_info['gardiner_code']} as incorrect!")
                        st.rerun()
                
                with col_c:
                    if st.button("Uncertain", key=f"uncertain_{pred_info['pred_key']}", use_container_width=True):
                        validated_predictions[pred_info['pred_key']] = "uncertain"
                        validation_results["validated_predictions"] = validated_predictions
                        validator.save_validation_results(validation_results)
                        st.warning(f"Marked {pred_info['gardiner_code']} as uncertain!")
                        st.rerun()
                        
        else:
            st.warning("No predictions to validate at the current confidence threshold.")
    
    # Statistics section
    st.markdown("---")
    st.header("Validation Statistics")
    
    # Calculate statistics
    total_predictions = len([p for p in predictions_data if p.get('score', 0) >= min_confidence])
    validated_count = len(validated_predictions)
    
    if validated_count > 0:
        status_counts = {}
        for status in validated_predictions.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            st.metric("Validated", validated_count)
        
        with col3:
            progress = (validated_count / total_predictions) * 100 if total_predictions > 0 else 0
            st.metric("Progress", f"{progress:.1f}%")
        
        with col4:
            if status_counts.get('correct', 0) > 0:
                accuracy = (status_counts.get('correct', 0) / validated_count) * 100
                st.metric("Validation Accuracy", f"{accuracy:.1f}%")
        
        # Validation status distribution
        if status_counts:
            st.subheader("Validation Status Distribution")
            
            # Create pie chart
            fig_pie = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Validation Results Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Confidence vs Validation Results
        if validated_count > 5:  # Only show if we have enough data
            st.subheader("Confidence vs Validation Results")
            
            confidence_data = []
            for pred in predictions_data:
                if pred.get('score', 0) >= min_confidence:
                    pred_key = f"{pred.get('image_id', 'unknown')}_{predictions_data.index(pred)}"
                    if pred_key in validated_predictions:
                        confidence_data.append({
                            'confidence': pred.get('score', 0),
                            'status': validated_predictions[pred_key],
                            'gardiner_code': validator.get_gardiner_info(pred.get('category_id', 0))[0]
                        })
            
            if confidence_data:
                df_confidence = pd.DataFrame(confidence_data)
                fig_scatter = px.scatter(
                    df_confidence, 
                    x='confidence', 
                    y='status',
                    color='status',
                    hover_data=['gardiner_code'],
                    title="Prediction Confidence by Validation Status"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    else:
        st.info("No predictions have been validated yet. Start validating to see statistics!")
    
    # Export functionality
    st.markdown("---")
    st.header("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Validation Results"):
            if validated_predictions:
                # Create export data
                export_data = []
                for pred in predictions_data:
                    if pred.get('score', 0) >= min_confidence:
                        pred_key = f"{pred.get('image_id', 'unknown')}_{predictions_data.index(pred)}"
                        if pred_key in validated_predictions:
                            gardiner_code, gardiner_info = validator.get_gardiner_info(pred.get('category_id', 0))
                            export_data.append({
                                'image_id': pred.get('image_id', 'unknown'),
                                'gardiner_code': gardiner_code,
                                'gardiner_name': gardiner_info['name'],
                                'confidence': pred.get('score', 0),
                                'validation_status': validated_predictions[pred_key],
                                'bbox': pred.get('bbox', []),
                                'timestamp': datetime.now().isoformat()
                            })
                
                # Convert to CSV
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"hieratic_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
            else:
                st.warning("No validated predictions to export")
    
    with col2:
        if st.button("Reset All Validations"):
            if st.session_state.get('confirm_reset', False):
                validation_results["validated_predictions"] = {}
                validator.save_validation_results(validation_results)
                st.success("All validations reset!")
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")

if __name__ == "__main__":
    main()
