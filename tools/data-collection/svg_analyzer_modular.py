#!/usr/bin/env python3
"""
Modular SVG Analysis Script

This script analyzes SVG files in a directory and provides comprehensive
statistics about their dimensions, complexity, and structure using the
hieroglyph_toolkit modular architecture.

Usage:
    python svg_analyzer_modular.py [--config CONFIG_FILE] [--svg-dir SVG_DIR] [--output OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
import glob

# Add the toolkit to Python path
sys.path.append(str(Path(__file__).parent / "hieroglyph_scraping_toolkit"))

from hieroglyph_scraping_toolkit import (
    get_config_manager, 
    setup_logging,
    create_default_config_file,
    safe_json_save,
    progress_bar
)


def main():
    """
    Main execution function for the SVG analysis script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze SVG files and generate comprehensive statistics"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--svg-dir", 
        type=str, 
        help="Directory containing SVG files to analyze"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output directory for results"
    )
    parser.add_argument(
        "--create-config", 
        action="store_true",
        help="Create a default configuration file and exit"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.svg",
        help="File pattern to match SVG files (default: *.svg)"
    )
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config_file = args.config or "hieroglyph_config.json"
        create_default_config_file(config_file)
        print(f"Default configuration created at: {config_file}")
        print("Edit the configuration file to set your paths, then run the script again.")
        return 0
    
    try:
        # Load configuration
        config_manager = get_config_manager(args.config)
        path_config = config_manager.get_path_config()
        
        # Determine SVG directory
        if args.svg_dir:
            svg_directory = Path(args.svg_dir)
        elif path_config.svg_directory:
            svg_directory = path_config.svg_directory
        else:
            print(f"No SVG directory specified. Use --svg-dir or set svg_directory in config.")
            return 1
        
        if not svg_directory.exists():
            print(f"SVG directory not found: {svg_directory}")
            return 1
        
        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = path_config.output_directory
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else "INFO"
        log_file = output_dir / "svg_analysis.log"
        logger = setup_logging(
            log_file=str(log_file),
            log_level=getattr(__import__('logging'), log_level)
        )
        
        logger.info("Starting SVG analysis process")
        logger.info(f"SVG directory: {svg_directory}")
        logger.info(f"Output directory: {output_dir}")
        
        # Find SVG files
        svg_files = list(svg_directory.glob(args.pattern))
        if not svg_files:
            print(f"No SVG files found in {svg_directory} matching pattern '{args.pattern}'")
            return 1
        
        print(f"Found {len(svg_files)} SVG files to analyze")
        
        # Initialize SVG processor (import here to avoid dependency issues)
        from hieroglyph_scraping_toolkit.svg import SVGProcessor
        processor = SVGProcessor(svg_directory, logger)
        
        # Analyze each SVG file
        all_analyses = []
        filenames = [f.name for f in svg_files]
        
        print(f"\nAnalyzing {len(svg_files)} SVG files...")
        
        for i, svg_file in enumerate(svg_files):
            try:
                # Show progress
                if not args.verbose:
                    progress = progress_bar(i + 1, len(svg_files), 
                                         prefix="Progress", suffix="Complete")
                    print(progress, end='', flush=True)
                
                # Load and analyze SVG
                svg_content = processor.load_svg_file(svg_file.name)
                
                if svg_content:
                    metrics = processor.analyze_svg_metrics(svg_content)
                    
                    # Add filename to metrics
                    analysis = {
                        'filename': svg_file.name,
                        'file_path': str(svg_file),
                        'file_size_bytes': svg_file.stat().st_size,
                        **metrics
                    }
                    
                    all_analyses.append(analysis)
                    
                    if args.verbose:
                        print(f"Analyzed {svg_file.name} ({i+1}/{len(svg_files)})")
                        print(f"Dimensions: {metrics['viewbox']['width']:.1f}x{metrics['viewbox']['height']:.1f}")
                        print(f"Elements: {metrics['complexity']['total_elements']}")
                        print(f"Complexity: {metrics['complexity']['complexity_score']:.2f}")
                else:
                    logger.warning(f"Failed to load SVG file: {svg_file.name}")
                    if args.verbose:
                        print(f"Failed to load {svg_file.name}")
                
            except Exception as e:
                logger.error(f"Error analyzing {svg_file.name}: {e}")
                if args.verbose:
                    print(f"Error analyzing {svg_file.name}: {e}")
                continue
        
        if not args.verbose:
            print()  # New line after progress bar
        
        if not all_analyses:
            print("No SVG files were successfully analyzed")
            return 1
        
        # Calculate aggregate statistics
        print(f"\nCalculating statistics...")
        
        # Basic statistics
        total_files = len(all_analyses)
        avg_file_size = sum(a['file_size_bytes'] for a in all_analyses) / total_files
        
        # Dimension statistics
        widths = [a['viewbox']['width'] for a in all_analyses]
        heights = [a['viewbox']['height'] for a in all_analyses]
        
        dimension_stats = {
            'average_width': sum(widths) / len(widths),
            'average_height': sum(heights) / len(heights),
            'min_width': min(widths),
            'max_width': max(widths),
            'min_height': min(heights),
            'max_height': max(heights)
        }
        
        # Element statistics
        element_counts = {}
        complexity_scores = []
        
        for analysis in all_analyses:
            elements = analysis['elements']
            for elem_type, count in elements.items():
                if elem_type not in element_counts:
                    element_counts[elem_type] = []
                element_counts[elem_type].append(count)
            
            complexity_scores.append(analysis['complexity']['complexity_score'])
        
        element_stats = {}
        for elem_type, counts in element_counts.items():
            if counts:  # Only if we have data
                element_stats[elem_type] = {
                    'total': sum(counts),
                    'average': sum(counts) / len(counts),
                    'max': max(counts),
                    'files_with_element': sum(1 for c in counts if c > 0)
                }
        
        # Compile final statistics
        final_stats = {
            'summary': {
                'total_files_analyzed': total_files,
                'total_files_found': len(svg_files),
                'success_rate_percent': (total_files / len(svg_files)) * 100,
                'average_file_size_bytes': avg_file_size,
                'svg_directory': str(svg_directory),
                'analysis_date': __import__('datetime').datetime.now().isoformat()
            },
            'dimensions': dimension_stats,
            'elements': element_stats,
            'complexity': {
                'average_complexity': sum(complexity_scores) / len(complexity_scores),
                'min_complexity': min(complexity_scores),
                'max_complexity': max(complexity_scores)
            },
            'detailed_analyses': all_analyses
        }
        
        # Save results
        results_file = output_dir / "svg_analysis_results.json"
        if safe_json_save(final_stats, results_file):
            logger.info("Analysis results saved successfully")
            
            # Print summary
            print(f"SVG Analysis Complete!")
            print(f"Analyzed {total_files}/{len(svg_files)} files successfully")
            print(f"Results saved to: {results_file}")
            
            print(f"\nDimension Statistics:")
            print(f"Average size: {dimension_stats['average_width']:.1f} × {dimension_stats['average_height']:.1f}")
            print(f"Size range: {dimension_stats['min_width']:.1f}-{dimension_stats['max_width']:.1f} × {dimension_stats['min_height']:.1f}-{dimension_stats['max_height']:.1f}")
            
            print(f"\nElement Statistics:")
            for elem_type, stats in element_stats.items():
                if stats['total'] > 0:
                    print(f"{elem_type}: {stats['total']} total, avg {stats['average']:.1f} per file")
            
            print(f"\nComplexity:")
            print(f"Average complexity score: {final_stats['complexity']['average_complexity']:.2f}")
            print(f"Complexity range: {final_stats['complexity']['min_complexity']:.2f} - {final_stats['complexity']['max_complexity']:.2f}")
            
            print(f"\nLog file: {log_file}")
            
            # Show cache statistics
            cache_stats = processor.get_cache_statistics()
            print(f"\nCache Statistics:")
            print(f"SVG cache size: {cache_stats['svg_cache_size']}")
            
            return 0
        else:
            logger.error("Failed to save results")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())