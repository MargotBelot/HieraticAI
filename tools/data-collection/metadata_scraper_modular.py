#!/usr/bin/env python3
"""
Modular Metadata Scraper Script

This script scrapes metadata from the AKU-PAL website for specific signs
and saves the metadata in both JSON and TXT formats using the hieroglyph_toolkit
modular architecture.

Usage:
    python metadata_scraper_modular.py [--config CONFIG_FILE] [--signs-file SIGNS_FILE] [--output OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add the toolkit to Python path
sys.path.append(str(Path(__file__).parent / "hieroglyph_scraping_toolkit"))

from hieroglyph_scraping_toolkit import (create_default_config_file,
                                         get_config_manager, progress_bar,
                                         safe_file_read, safe_json_load,
                                         setup_logging)


def load_sign_numbers(signs_file: str) -> List[str]:
    """
    Load sign numbers from various file formats.

    Args:
        signs_file: Path to file containing sign numbers

    Returns:
        List[str]: List of sign numbers to process
    """
    signs_path = Path(signs_file)

    if not signs_path.exists():
        raise FileNotFoundError(f"Signs file not found: {signs_file}")

    if signs_path.suffix.lower() == ".json":
        # Load from JSON file
        data = safe_json_load(signs_path)
        if isinstance(data, list):
            return data
        else:
            raise ValueError("JSON file must contain a list of sign numbers")

    elif signs_path.suffix.lower() == ".txt":
        # Load from text file (one sign per line or URLs)
        content = safe_file_read(signs_path)
        if not content:
            raise ValueError(f"Could not read signs file: {signs_file}")

        lines = content.strip().split("\n")
        signs = []

        for line in lines:
            line = line.strip()
            if line:
                # Extract sign number from URL if needed
                if line.startswith("http"):
                    sign_id = line.split("/")[-1]
                    signs.append(sign_id)
                else:
                    signs.append(line)

        return signs

    else:
        raise ValueError(f"Unsupported file format: {signs_path.suffix}")


def main():
    """
    Main execution function for the metadata scraping script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Scrape metadata from AKU-PAL website for specific signs"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--signs-file",
        type=str,
        required=True,
        help="Path to file containing sign numbers (JSON or TXT format)",
    )
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file and exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of signs to process before saving intermediate results",
    )
    parser.add_argument(
        "--download-svg",
        action="store_true",
        help="Also download SVG files for each sign",
    )

    args = parser.parse_args()

    # Create default config if requested
    if args.create_config:
        config_file = args.config or "hieroglyph_config.json"
        create_default_config_file(config_file)
        print(f"Default configuration created at: {config_file}")
        print(
            "Edit the configuration file to set your paths, then run the script again."
        )
        return 0

    try:
        # Load sign numbers
        print("Loading sign numbers...")
        sign_numbers = load_sign_numbers(args.signs_file)
        print(f"Loaded {len(sign_numbers)} sign numbers")

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
        log_file = output_dir / "metadata_scraping.log"
        logger = setup_logging(
            log_file=str(log_file), log_level=getattr(__import__("logging"), log_level)
        )

        logger.info("Starting metadata scraping process")
        logger.info(f"Signs to process: {len(sign_numbers)}")
        logger.info(f"Output directory: {output_dir}")

        # Initialize scraper (import here to avoid dependency issues)
        from hieroglyph_scraping_toolkit.scraping import AKUPALScraper

        with AKUPALScraper(web_config, str(log_file)) as scraper:
            logger.info("Initialized AKU-PAL scraper")

            all_metadata = []
            processed_count = 0

            print(f"\nProcessing {len(sign_numbers)} signs...")

            for i, sign_id in enumerate(sign_numbers):
                try:
                    # Show progress
                    if not args.verbose:
                        progress = progress_bar(
                            i + 1,
                            len(sign_numbers),
                            prefix="Progress",
                            suffix="Complete",
                        )
                        print(progress, end="", flush=True)

                    # Scrape metadata
                    metadata = scraper.scrape_sign_metadata(sign_id)

                    if metadata:
                        all_metadata.append(metadata)
                        processed_count += 1

                        if args.verbose:
                            print(
                                f"Processed sign {sign_id} ({i+1}/{len(sign_numbers)})"
                            )
                    else:
                        logger.warning(f"Failed to scrape metadata for sign {sign_id}")
                        if args.verbose:
                            print(f"Failed sign {sign_id} ({i+1}/{len(sign_numbers)})")

                    # Download SVG if requested
                    if args.download_svg and metadata:
                        svg_success = scraper.download_sign_svg(
                            sign_id, str(output_dir / "svg")
                        )
                        if svg_success and args.verbose:
                            print(f"Downloaded SVG for {sign_id}")

                    # Save intermediate results
                    if (
                        len(all_metadata) % args.batch_size == 0
                        and len(all_metadata) > 0
                    ):
                        logger.info(
                            f"Saving intermediate results ({len(all_metadata)} metadata records)"
                        )
                        scraper.save_metadata_batch(all_metadata, str(output_dir))

                except KeyboardInterrupt:
                    print("\nOperation cancelled by user")
                    break
                except Exception as e:
                    logger.error(f"Error processing sign {sign_id}: {e}")
                    if args.verbose:
                        print(f"Error with sign {sign_id}: {e}")
                    continue

            if not args.verbose:
                print()  # New line after progress bar

            # Save final results
            if all_metadata:
                logger.info("Saving final results...")
                success = scraper.save_metadata_batch(all_metadata, str(output_dir))

                if success:
                    logger.info("Metadata saved successfully")

                    # Print summary
                    print(f"\nMetadata Scraping Complete!")
                    print(
                        f"Processed {processed_count}/{len(sign_numbers)} signs successfully"
                    )
                    print(f"Results saved to: {output_dir}")
                    print(f"Files created:")
                    print(f"- all_metadata.json ({len(all_metadata)} records)")
                    print(f"- all_metadata.txt")
                    if args.download_svg:
                        print(f"- svg/ (directory with SVG files)")
                    print(f"Log file: {log_file}")

                    # Get and display scraping statistics
                    stats = scraper.get_scraping_statistics()
                    print(f"\nStatistics:")
                    print(f"- Success rate: {stats['success_rate']:.1f}%")
                    print(f"- Failed signs: {stats['failed_signs_count']}")
                    print(f"- Cache hits: {stats.get('metadata_cache_size', 0)}")

                    return 0
                else:
                    logger.error("Failed to save results")
                    return 1
            else:
                print("\nNo metadata was successfully collected")
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
