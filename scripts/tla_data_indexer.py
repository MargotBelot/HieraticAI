#!/usr/bin/env python3
"""
TLA Data Indexer
================

Parse and index TLA (Thesaurus Linguae Aegyptiae) lemma data to enrich 
hieroglyph predictions with linguistic information.

This script extracts lemma information including:
- Hieroglyphic writing
- Transliteration 
- English translation/meaning
- Frequency in corpus
- Lemma ID

Authors: Margot Belot <margotbelot@icloud.com>
         Domino Colyer <dominic23@zedat.fu-berlin.de>
Date: August 2025
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import unicodedata

class TLADataIndexer:
    def __init__(self, tla_file_path: str):
        """
        Initialize TLA data indexer.
        
        Args:
            tla_file_path: Path to TLA_Scraping.txt file
        """
        self.tla_file_path = Path(tla_file_path)
        self.lemma_data = {}
        self.gardiner_to_lemmas = defaultdict(list)
        self.hieroglyph_to_lemmas = defaultdict(list)
        
        print(f"🏺 Initializing TLA indexer for: {self.tla_file_path}")
    
    def parse_tla_data(self):
        """Parse the TLA scraping file and extract lemma information."""
        print("📖 Parsing TLA lemma data...")
        
        if not self.tla_file_path.exists():
            print(f"❌ TLA file not found: {self.tla_file_path}")
            return
        
        with open(self.tla_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # The file uses \r\r\n line endings and has empty lines between fields
        # Remove BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]
        
        # Replace line endings
        content = content.replace('\r\r\n', '\n')
        
        # Pattern with empty lines: number) /lemma/id: \n\n\t hieroglyphs \n\n\t transliteration \n\n\t translation \n\n\t Frequency: num
        lemma_pattern = r'(\d+)\) /lemma/(\d+):\n\n\t([^\n]*)\n\n\t([^\n]*)\n\n\t([^\n]*)\n\n\tFrequency: (\d+)'
        
        matches = re.findall(lemma_pattern, content, re.MULTILINE)
        
        processed = 0
        errors = 0
        
        for match in matches:
            try:
                entry_num, lemma_id, hieroglyphs, transliteration, translation, frequency = match
                
                # Clean up the data
                hieroglyphs = hieroglyphs.strip()
                transliteration = transliteration.strip()
                translation = translation.strip()
                frequency = int(frequency)
                
                # Handle missing hieroglyphs
                has_hieroglyphs = hieroglyphs and not hieroglyphs.startswith("Hieroglyph not found")
                
                lemma_record = {
                    "lemma_id": lemma_id,
                    "entry_number": int(entry_num),
                    "hieroglyphs": hieroglyphs if has_hieroglyphs else None,
                    "transliteration": transliteration,
                    "translation": translation,
                    "frequency": frequency,
                    "has_hieroglyphs": has_hieroglyphs
                }
                
                # Store by lemma ID
                self.lemma_data[lemma_id] = lemma_record
                
                # Index by individual hieroglyphs if available
                if has_hieroglyphs:
                    individual_signs = self.extract_individual_signs(hieroglyphs)
                    for sign in individual_signs:
                        self.hieroglyph_to_lemmas[sign].append(lemma_record)
                
                processed += 1
                
            except Exception as e:
                errors += 1
                print(f"   ⚠️  Error processing lemma entry: {e}")
                continue
        
        print(f"✅ Processed {processed} lemma entries ({errors} errors)")
        print(f"📊 Indexed {len(self.lemma_data)} lemmas")
        print(f"🔤 Found {len(self.hieroglyph_to_lemmas)} unique hieroglyphic signs")
    
    def extract_individual_signs(self, hieroglyphs: str) -> List[str]:
        """Extract individual hieroglyphic signs from a lemma's hieroglyphic writing."""
        if not hieroglyphs:
            return []
        
        signs = []
        # Split into individual Unicode characters
        for char in hieroglyphs:
            # Check if it's an Egyptian hieroglyph (Unicode blocks 13000-1342F)
            char_code = ord(char)
            if 0x13000 <= char_code <= 0x1342F:
                signs.append(char)
        
        return list(set(signs))  # Remove duplicates
    
    def get_gardiner_code_from_unicode(self, unicode_char: str) -> Optional[str]:
        """
        Convert Unicode hieroglyph to Gardiner code.
        This is a simplified mapping - in practice, you'd want a comprehensive lookup.
        """
        # This would require a comprehensive Unicode -> Gardiner mapping
        # For now, we'll use Unicode codepoint as identifier
        if '\u13000' <= unicode_char <= '\u1342F':
            return f"U+{ord(unicode_char):04X}"
        return None
    
    def get_lemmas_for_hieroglyph(self, hieroglyph: str, max_results: int = 10) -> List[Dict]:
        """Get lemma information for a specific hieroglyph."""
        return self.hieroglyph_to_lemmas.get(hieroglyph, [])[:max_results]
    
    def get_lemmas_for_gardiner_code(self, gardiner_code: str, max_results: int = 10) -> List[Dict]:
        """Get lemma information for a Gardiner code (if we had the mapping)."""
        # This would require Gardiner -> Unicode mapping
        return []
    
    def create_lemma_lookup_index(self) -> Dict:
        """Create a searchable index for quick lemma lookups."""
        index = {
            'by_hieroglyph': dict(self.hieroglyph_to_lemmas),
            'by_lemma_id': self.lemma_data,
            'statistics': {
                'total_lemmas': len(self.lemma_data),
                'lemmas_with_hieroglyphs': sum(1 for l in self.lemma_data.values() if l['has_hieroglyphs']),
                'unique_hieroglyphic_signs': len(self.hieroglyph_to_lemmas),
                'most_frequent_lemmas': sorted(
                    self.lemma_data.values(), 
                    key=lambda x: x['frequency'], 
                    reverse=True
                )[:20]
            }
        }
        
        return index
    
    def export_index(self, output_path: str):
        """Export the TLA index for use by other tools."""
        index = self.create_lemma_lookup_index()
        
        # Convert defaultdict to regular dict for JSON serialization
        index['by_hieroglyph'] = {k: v for k, v in index['by_hieroglyph'].items()}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Exported TLA index to: {output_path}")
    
    def create_summary_report(self) -> str:
        """Create a summary report of the TLA data."""
        stats = self.create_lemma_lookup_index()['statistics']
        
        report = []
        report.append("🏺 TLA (Thesaurus Linguae Aegyptiae) Summary")
        report.append("=" * 50)
        report.append(f"📊 Total lemmas: {stats['total_lemmas']}")
        report.append(f"🔤 Lemmas with hieroglyphs: {stats['lemmas_with_hieroglyphs']}")
        report.append(f"📝 Unique hieroglyphic signs: {stats['unique_hieroglyphic_signs']}")
        report.append("")
        
        report.append("🔝 Top 15 Most Frequent Lemmas:")
        report.append("-" * 60)
        report.append(f"{'Rank':<4} {'Freq':<6} {'Transliteration':<15} {'Translation'[:30]}")
        report.append("-" * 60)
        
        for i, lemma in enumerate(stats['most_frequent_lemmas'][:15], 1):
            trans = lemma['transliteration'][:14]
            meaning = lemma['translation'][:30]
            report.append(f"{i:<4} {lemma['frequency']:<6} {trans:<15} {meaning}")
        
        report.append("")
        report.append("📈 Coverage by Hieroglyphic Signs:")
        
        # Show most referenced hieroglyphic signs
        sign_counts = {sign: len(lemmas) for sign, lemmas in self.hieroglyph_to_lemmas.items()}
        top_signs = sorted(sign_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for sign, count in top_signs:
            unicode_name = unicodedata.name(sign, f"U+{ord(sign):04X}")
            report.append(f"   {sign} ({unicode_name}): {count} lemmas")
        
        return "\n".join(report)

def main():
    """Main function to build TLA index."""
    tla_file = "/Users/margot/Desktop/ALP_project/TLA_Data/TLA_Scraping.txt"
    
    if not Path(tla_file).exists():
        print(f"❌ TLA file not found: {tla_file}")
        return
    
    # Create indexer
    indexer = TLADataIndexer(tla_file)
    
    # Parse and process TLA data
    indexer.parse_tla_data()
    
    # Export index for use by validation interface
    output_path = "data/tla_lemma_index.json"
    Path(output_path).parent.mkdir(exist_ok=True)
    indexer.export_index(output_path)
    
    # Create and save summary report
    report = indexer.create_summary_report()
    print("\n" + report)
    
    with open("data/tla_lemma_summary.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    # Show some example lookups
    print("\n🔍 Example Hieroglyph Lookups:")
    
    # Test with some hieroglyphic signs if available
    if indexer.hieroglyph_to_lemmas:
        sample_signs = list(indexer.hieroglyph_to_lemmas.keys())[:5]
        for sign in sample_signs:
            lemmas = indexer.get_lemmas_for_hieroglyph(sign, max_results=2)
            print(f"   {sign}: {len(lemmas)} lemma(s) found")
            for lemma in lemmas[:1]:  # Show first match
                print(f"     → {lemma['transliteration']}: {lemma['translation']} (freq: {lemma['frequency']})")

if __name__ == "__main__":
    main()
