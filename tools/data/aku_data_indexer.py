#!/usr/bin/env python3
"""
Index AKU Westcar database signs by Gardiner codes to enrich prediction validation.
Creates a searchable index linking SVGs and metadata for similar hieroglyphs.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Optional
import re

class AKUDataIndexer:
    def __init__(self, aku_data_path: str):
        """
        Initialize the AKU data indexer.
        
        Args:
            aku_data_path: Path to AKU Westcar Scraping folder
        """
        self.aku_data_path = Path(aku_data_path)
        self.json_path = self.aku_data_path / "json"
        self.svg_path = self.aku_data_path / "svg"
        self.txt_path = self.aku_data_path / "txt"
        
        # Index by Gardiner code
        self.gardiner_index = defaultdict(list)
        self.aku_records = {}
        
        print(f"Initializing AKU indexer for: {self.aku_data_path}")
        
    def load_all_records(self):
        """Load all JSON records and create indices."""
        print("Loading AKU database records...")
        
        json_files = list(self.json_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        processed = 0
        errors = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract key information
                record_id = json_file.stem
                aku_nr = data.get("AKU-Nr.", "")
                mdc_code = data.get("Manuel de Codage (MdC)", "").strip()
                
                # Find corresponding SVG file
                svg_file = self.find_svg_file(record_id, aku_nr)
                
                # Create enriched record with relative paths
                json_relative = json_file.relative_to(Path.cwd()) if json_file.is_absolute() else json_file
                svg_relative = svg_file.relative_to(Path.cwd()) if (svg_file and svg_file.is_absolute()) else svg_file
                
                enriched_record = {
                    "id": record_id,
                    "aku_nr": aku_nr,
                    "gardiner_code": mdc_code,
                    "json_path": str(json_relative),
                    "svg_path": str(svg_relative) if svg_relative else None,
                    "metadata": data
                }
                
                # Store in main index
                self.aku_records[record_id] = enriched_record
                
                # Index by Gardiner code if available
                if mdc_code:
                    # Clean up the Gardiner code (remove extra spaces, etc.)
                    clean_code = self.clean_gardiner_code(mdc_code)
                    if clean_code:
                        self.gardiner_index[clean_code].append(enriched_record)
                
                processed += 1
                
            except Exception as e:
                errors += 1
                print(f"  Error processing {json_file}: {e}")
                continue
        
        print(f"Processed {processed} records ({errors} errors)")
        print(f"Indexed {len(self.gardiner_index)} unique Gardiner codes")
        
    def find_svg_file(self, record_id: str, aku_nr: str) -> Optional[Path]:
        """Find corresponding SVG file for a record."""
        # Try different naming patterns
        patterns = [
            f"ht_{record_id}.svg",
            f"{aku_nr}.svg",
            f"{record_id}.svg"
        ]
        
        for pattern in patterns:
            svg_file = self.svg_path / pattern
            if svg_file.exists():
                return svg_file
        
        # Try to find any SVG with the record ID
        matching_svgs = list(self.svg_path.glob(f"*{record_id}*"))
        if matching_svgs:
            return matching_svgs[0]
        
        return None
    
    def clean_gardiner_code(self, code: str) -> str:
        """Clean and standardize Gardiner codes."""
        if not code:
            return ""
        
        # Remove extra whitespace
        code = code.strip()
        
        # Handle common variations
        code = re.sub(r'\s+', '', code)  # Remove all whitespace
        code = code.upper()  # Standardize case
        
        # Validate format (basic check)
        if re.match(r'^[A-Z]+\d+[A-Z]?$', code):
            return code
        
        return code  # Return as-is if doesn't match expected pattern
    
    def get_similar_signs(self, gardiner_code: str, max_results: int = 10) -> List[Dict]:
        """Get similar signs for a given Gardiner code."""
        clean_code = self.clean_gardiner_code(gardiner_code)
        
        similar_signs = self.gardiner_index.get(clean_code, [])
        
        # Sort by some criteria (e.g., completeness, date)
        sorted_signs = sorted(similar_signs, key=lambda x: (
            x['svg_path'] is not None,  # SVG available
            x['metadata'].get('Lesbarkeit', '') == 'ja',  # Readable
            x['metadata'].get('Zustand', '') == 'vollständig',  # Complete
        ), reverse=True)
        
        return sorted_signs[:max_results]
    
    def get_gardiner_statistics(self) -> Dict:
        """Get statistics about the Gardiner code distribution."""
        stats = {}
        
        for code, records in self.gardiner_index.items():
            svg_count = sum(1 for r in records if r['svg_path'] is not None)
            readable_count = sum(1 for r in records if r['metadata'].get('Lesbarkeit', '') == 'ja')
            complete_count = sum(1 for r in records if r['metadata'].get('Zustand', '') == 'vollständig')
            
            stats[code] = {
                'total': len(records),
                'with_svg': svg_count,
                'readable': readable_count,
                'complete': complete_count,
                'sample_metadata': records[0]['metadata'] if records else {}
            }
        
        return stats
    
    def export_index(self, output_path: str):
        """Export the index for use by other tools."""
        export_data = {
            'gardiner_index': {code: records for code, records in self.gardiner_index.items()},
            'statistics': self.get_gardiner_statistics(),
            'total_records': len(self.aku_records),
            'total_gardiner_codes': len(self.gardiner_index)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported index to: {output_path}")
    
    def create_summary_report(self) -> str:
        """Create a summary report of the AKU database."""
        stats = self.get_gardiner_statistics()
        
        report = []
        report.append("AKU Westcar Database Summary")
        report.append("="* 50)
        report.append(f"Total records: {len(self.aku_records)}")
        report.append(f"Unique Gardiner codes: {len(self.gardiner_index)}")
        report.append("")
        
        # Top categories by frequency
        sorted_codes = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        report.append("Top 20 Most Common Gardiner Codes:")
        report.append("-"* 50)
        report.append(f"{'Code':<8} {'Total':<6} {'SVG':<5} {'Read':<5} {'Comp':<5} {'Description'}")
        report.append("-"* 50)
        
        for code, data in sorted_codes[:20]:
            desc = data['sample_metadata'].get('Beschreibung', '')[:30]
            report.append(f"{code:<8} {data['total']:<6} {data['with_svg']:<5} "
                         f"{data['readable']:<5} {data['complete']:<5} {desc}")
        
        report.append("")
        report.append("Coverage Statistics:")
        report.append(f"Records with SVG: {sum(s['with_svg'] for s in stats.values())}")
        report.append(f"Records readable: {sum(s['readable'] for s in stats.values())}")
        report.append(f"Records complete: {sum(s['complete'] for s in stats.values())}")
        
        return "\n".join(report)

def main():
    """Main function to build AKU index."""
    aku_path = "./external_data/AKU Westcar Scraping"
    
    if not os.path.exists(aku_path):
        print(f"AKU data path not found: {aku_path}")
        return
    
    # Create indexer
    indexer = AKUDataIndexer(aku_path)
    
    # Load and process all records
    indexer.load_all_records()
    
    # Export index for use by validation interface
    output_path = "data/aku_gardiner_index.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    indexer.export_index(output_path)
    
    # Create and save summary report
    report = indexer.create_summary_report()
    print("\n"+ report)
    
    with open("data/aku_database_summary.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    # Show some examples
    print("\n Example lookups:")
    
    # Test with some common Gardiner codes
    test_codes = ["A1", "M17", "N35", "G1", "D21"]
    
    for code in test_codes:
        similar = indexer.get_similar_signs(code, max_results=3)
        print(f"{code}: {len(similar)} similar signs found")
        
        if similar:
            for i, sign in enumerate(similar[:2]):
                desc = sign['metadata'].get('Beschreibung', 'No description')
                svg_status = "SVG"if sign['svg_path'] else "No SVG"
                print(f"  {i+1}. {desc[:40]} ({svg_status})")

if __name__ == "__main__":
    main()
