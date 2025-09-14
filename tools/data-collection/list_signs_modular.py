#!/usr/bin/env python3
"""
Modular Sign Collection Script

This script collects all unique sign numbers from the AKU-PAL website
and saves them in both JSON and TXT formats using the hieroglyph_toolkit
modular architecture.

Usage:
    python list_signs_modular.py [--config CONFIG_FILE] [--output OUTPUT_DIR]

Author: Margot
Date: September 2024
"""

import argparse
import sys
from pathlib import Path

# Add the toolkit to Python path
sys.path.append(str(Path(__file__).parent / "hieroglyph_scraping_toolkit"))

from hieroglyph_scraping_toolkit import (
    get_config_manager, 
    setup_logging,
    create_default_config_file
)


def main():
    """
    Main execution function for the sign collection script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Collect sign numbers from AKU-PAL website"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
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
        web_config = config_manager.get_web_scraping_config()
        path_config = config_manager.get_path_config()
        
        # Override output directory if specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = path_config.output_directory
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else "INFO"
        log_file = output_dir / "sign_collection.log"
        logger = setup_logging(
            log_file=str(log_file),
            log_level=getattr(__import__('logging'), log_level)
        )
        
        logger.info("Starting sign number collection process")
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize scraper (import here to avoid dependency issues)
        from hieroglyph_scraping_toolkit.scraping import AKUPALScraper
        with AKUPALScraper(web_config, str(log_file)) as scraper:
            logger.info("Initialized AKU-PAL scraper")
            
            # Collect sign numbers
            logger.info("Collecting sign numbers from AKU-PAL...")
            sign_numbers = scraper.collect_all_sign_numbers()
            
            if not sign_numbers:
                logger.error("No sign numbers collected. Check your internet connection and try again.")
                return 1
            
            logger.info(f"Successfully collected {len(sign_numbers)} unique sign numbers")
            
            # Save results
            logger.info("Saving results...")
            success = scraper.save_sign_numbers(sign_numbers, str(output_dir))
            
            if success:
                logger.info("Sign numbers saved successfully")
                
                # Print summary
                print(f"\n✅ Sign Collection Complete!")
                print(f"📊 Collected {len(sign_numbers)} unique sign numbers")
                print(f"💾 Results saved to: {output_dir}")
                print(f"📄 Files created:")
                print(f"   - sign_numbers.json")
                print(f"   - sign_numbers.txt")
                print(f"📋 Log file: {log_file}")
                
                # Get and display scraping statistics
                stats = scraper.get_scraping_statistics()
                print(f"\n📈 Statistics:")
                print(f"   - Success rate: {stats['success_rate']:.1f}%")
                print(f"   - Failed signs: {stats['failed_signs_count']}")
                
                return 0
            else:
                logger.error("Failed to save results")
                return 1
                
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())