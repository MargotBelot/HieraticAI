"""
AKU-PAL Metadata Scraper Module

This module provides specialized scraping functionality for the AKU-PAL website,
including sign collection, metadata extraction, and SVG downloading.

Author: Margot
Date: September 2024
"""

import requests
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper
from ..config import WebScrapingConfig
from ..utils import safe_json_save, safe_file_write, clean_filename, timing_decorator


class AKUPALScraper(BaseScraper):
    """
    Specialized scraper for the AKU-PAL website.
    
    This class handles sign discovery, metadata extraction, and SVG file downloading
    from the AKU-PAL (Altaegyptische Kursivschriften - Paläographische Datenbank) website.
    """
    
    def __init__(self, config: WebScrapingConfig, log_file: Optional[str] = None):
        """
        Initialize the AKU-PAL scraper.
        
        Args:
            config: Web scraping configuration
            log_file: Optional log file path
        """
        super().__init__(config, log_file)
        self.collected_signs = []
        self.failed_signs = []
        self.metadata_cache = {}
    
    @timing_decorator
    def collect_all_sign_numbers(self) -> List[str]:
        """
        Collect all unique sign numbers from the AKU-PAL graphemes page.
        
        Returns:
            List[str]: List of sign numbers found on the site
        """
        self.logger.info("Starting sign number collection from AKU-PAL")
        
        if not self.navigate_to_url(self.config.graphemes_url):
            self.logger.error("Failed to navigate to graphemes page")
            return []
        
        # Wait for the page to load and collect all unique IDs
        id_links = self.wait_for_elements((By.CSS_SELECTOR, "a.result-link"), timeout=15)
        
        if not id_links:
            self.logger.warning("No result links found on graphemes page")
            return []
        
        # Extract URLs with ID parameters
        id_urls = []
        for link in id_links:
            href = self.extract_attribute_safely(link, "href")
            if href and "id=" in href:
                id_urls.append(href)
        
        self.logger.info(f"Found {len(id_urls)} unique ID URLs")
        
        all_sign_numbers = []
        
        # Process each ID URL to extract signs
        for i, id_url in enumerate(id_urls):
            self.logger.info(f"Processing ID URL {i+1}/{len(id_urls)}: {id_url}")
            
            current_id = id_url.split("id=")[-1].split("&")[0]
            signs = self._extract_signs_from_id_page(id_url, current_id)
            
            if signs:
                all_sign_numbers.extend(signs)
                self.logger.info(f"Found {len(signs)} signs for ID {current_id}")
            else:
                self.logger.warning(f"No signs found for ID {current_id}")
            
            self.wait_between_requests()
        
        # Remove duplicates while preserving order
        unique_signs = []
        seen = set()
        for sign in all_sign_numbers:
            if sign not in seen:
                unique_signs.append(sign)
                seen.add(sign)
        
        self.collected_signs = unique_signs
        self.logger.info(f"Collected {len(unique_signs)} unique sign numbers")
        
        return unique_signs
    
    def _extract_signs_from_id_page(self, id_url: str, current_id: str) -> List[str]:
        """
        Extract sign numbers from a specific ID page.
        
        Args:
            id_url: URL of the ID page
            current_id: Current ID being processed
            
        Returns:
            List[str]: List of sign numbers found on the page
        """
        if not self.navigate_to_url(id_url):
            return []
        
        # Wait for hieratogramm images to load
        sign_images = self.wait_for_elements(
            (By.CSS_SELECTOR, "img[alt^='Hieratogramm']"), 
            timeout=10
        )
        
        signs = []
        for img in sign_images:
            alt_text = self.extract_attribute_safely(img, "alt")
            if alt_text and "Hieratogramm" in alt_text:
                # Extract the number from alt text
                number = alt_text.split("Hieratogramm")[-1].strip()
                if number:
                    signs.append(number)
        
        return signs
    
    @timing_decorator
    def scrape_sign_metadata(self, sign_id: str) -> Optional[Dict[str, Any]]:
        """
        Scrape metadata for a specific sign ID.
        
        Args:
            sign_id: The sign ID to scrape metadata for
            
        Returns:
            Dict[str, Any] or None: Scraped metadata or None if failed
        """
        # Check cache first
        if sign_id in self.metadata_cache:
            return self.metadata_cache[sign_id]
        
        sign_url = f"{self.config.base_url_signs}{sign_id}"
        self.logger.debug(f"Scraping metadata for sign {sign_id} from {sign_url}")
        
        if not self.navigate_to_url(sign_url):
            self.logger.error(f"Failed to navigate to sign page: {sign_url}")
            self.failed_signs.append(sign_id)
            return None
        
        # Wait for metadata table to load
        metadata_elements = self.wait_for_elements((By.CSS_SELECTOR, ".table tr td"), timeout=15)
        
        if not metadata_elements:
            self.logger.warning(f"No metadata table found for sign {sign_id}")
            return None
        
        # Extract metadata text
        metadata_texts = [self.extract_text_safely(elem) for elem in metadata_elements]
        filtered_metadata = self._filter_unwanted_metadata(metadata_texts)
        
        # Structure the metadata
        metadata_dict = self._structure_metadata(filtered_metadata, sign_id)
        
        # Cache the result
        self.metadata_cache[sign_id] = metadata_dict
        
        return metadata_dict
    
    def _filter_unwanted_metadata(self, metadata_list: List[str]) -> List[str]:
        """
        Filter out unwanted metadata entries.
        
        Args:
            metadata_list: List of metadata strings
            
        Returns:
            List[str]: Filtered metadata list
        """
        unwanted_keywords = ['Zitierhinweis:', 'Grafiken -', 'Graphem -']
        return [
            item for item in metadata_list 
            if not any(keyword in item for keyword in unwanted_keywords)
        ]
    
    def _structure_metadata(self, metadata_list: List[str], sign_id: str) -> Dict[str, Any]:
        """
        Structure metadata into a dictionary with predefined keys.
        
        Args:
            metadata_list: List of metadata values
            sign_id: Sign ID being processed
            
        Returns:
            Dict[str, Any]: Structured metadata dictionary
        """
        metadata_keys = [
            "AKU-Nr.", "Manuel de Codage (MdC)", "Kategorie", "Subkategorie", 
            "Beschreibung", "Text", "Kolumne", "Zeile", "Schriftrichtung", 
            "Schriftart", "Material", "Zustand", "Breite", "Höhe", "Lesbarkeit", 
            "Zeitraum", "Textart", "Kommentar", "Retro-Digitalisat", 
            "Publikation", "Publikationsdatum", "Autor", "Lizenz"
        ]
        
        metadata_dict = {"sign_id": sign_id}
        
        for i, key in enumerate(metadata_keys):
            if i < len(metadata_list):
                metadata_dict[key] = metadata_list[i]
            else:
                metadata_dict[key] = ""
        
        return metadata_dict
    
    @timing_decorator
    def download_sign_svg(self, sign_id: str, output_directory: Optional[str] = None) -> bool:
        """
        Download SVG file for a specific sign.
        
        Args:
            sign_id: Sign ID to download SVG for
            output_directory: Optional output directory (uses config default if None)
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        svg_url = f"https://aku-pal.uni-mainz.de/img/data/ht/svg/ht_{sign_id}.svg"
        output_dir = output_directory or self.config.output_svg_dir
        
        try:
            self.logger.debug(f"Downloading SVG for sign {sign_id} from {svg_url}")
            
            response = requests.get(svg_url, timeout=30)
            response.raise_for_status()
            
            # Create safe filename
            filename = clean_filename(f"ht_{sign_id}.svg")
            svg_path = Path(output_dir) / filename
            
            # Ensure output directory exists
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write SVG content
            with open(svg_path, 'wb') as svg_file:
                svg_file.write(response.content)
            
            self.logger.info(f"SVG saved: {svg_path}")
            return True
            
        except requests.RequestException as e:
            self.logger.warning(f"Failed to download SVG for sign {sign_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading SVG for sign {sign_id}: {e}")
            return False
    
    @timing_decorator
    def scrape_hieratogramm_metadata(self, sign_url: str) -> Dict[str, Any]:
        """
        Scrape detailed metadata from Hieratogramm pages using BeautifulSoup.
        
        Args:
            sign_url: URL of the sign page to scrape
            
        Returns:
            Dict[str, Any]: Scraped metadata dictionary
        """
        if not self.navigate_to_url(sign_url):
            return {}
        
        # Get page source and parse with BeautifulSoup
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        
        sign_id = sign_url.strip().split("/")[-1]
        sign_data = {
            "sign_id": sign_id,
            "url": sign_url
        }
        
        # Extract metadata from different sections
        sections_to_extract = [
            "Graphem", 
            "Informationen zum Hieratogramm", 
            "Metadaten", 
            "Lizenzhinweis"
        ]
        
        for section in sections_to_extract:
            section_data = self._extract_metadata_section(soup, section)
            sign_data.update(section_data)
        
        return sign_data
    
    def _extract_metadata_section(self, soup: BeautifulSoup, section_title: str) -> Dict[str, str]:
        """
        Extract metadata from a specific section of the Hieratogramm page.
        
        Args:
            soup: BeautifulSoup parsed HTML
            section_title: Title of the section to extract
            
        Returns:
            Dict[str, str]: Extracted metadata as key-value pairs
        """
        data = {}
        header = soup.find("h2", string=section_title)
        
        if header:
            table = header.find_next("table")
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all(["th", "td"])
                    if len(cells) >= 2:
                        key = f"{section_title} | {cells[0].get_text(strip=True)}"
                        value = cells[1].get_text(strip=True)
                        data[key] = value
        
        return data
    
    def save_sign_numbers(self, sign_numbers: List[str], output_directory: str) -> bool:
        """
        Save collected sign numbers to JSON and TXT files.
        
        Args:
            sign_numbers: List of sign numbers to save
            output_directory: Directory to save files in
            
        Returns:
            bool: True if both files saved successfully
        """
        try:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON file
            json_success = safe_json_save(
                sign_numbers, 
                output_dir / "sign_numbers.json"
            )
            
            # Save TXT file with full URLs
            txt_content = "\n".join(
                f"{self.config.base_url_signs}{number}" 
                for number in sign_numbers
            )
            txt_success = safe_file_write(
                output_dir / "sign_numbers.txt", 
                txt_content
            )
            
            if json_success and txt_success:
                self.logger.info(f"Sign numbers saved to {output_dir}")
                return True
            else:
                self.logger.error("Failed to save one or both sign number files")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving sign numbers: {e}")
            return False
    
    def save_metadata_batch(self, metadata_list: List[Dict[str, Any]], output_directory: str) -> bool:
        """
        Save batch metadata to JSON and TXT files.
        
        Args:
            metadata_list: List of metadata dictionaries
            output_directory: Directory to save files in
            
        Returns:
            bool: True if files saved successfully
        """
        try:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save comprehensive JSON file
            json_success = safe_json_save(
                metadata_list, 
                output_dir / "all_metadata.json"
            )
            
            # Save human-readable TXT file
            txt_content = ""
            for item in metadata_list:
                txt_content += "=" * 60 + "\n"
                for key, value in item.items():
                    txt_content += f"{key}: {value}\n"
                txt_content += "\n"
            
            txt_success = safe_file_write(
                output_dir / "all_metadata.txt", 
                txt_content
            )
            
            if json_success and txt_success:
                self.logger.info(f"Batch metadata saved to {output_dir}")
                return True
            else:
                self.logger.error("Failed to save metadata files")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving batch metadata: {e}")
            return False
    
    def get_scraping_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive scraping statistics.
        
        Returns:
            Dict[str, Any]: Statistics about the scraping session
        """
        return {
            "total_signs_collected": len(self.collected_signs),
            "failed_signs_count": len(self.failed_signs),
            "metadata_cache_size": len(self.metadata_cache),
            "failed_signs": self.failed_signs,
            "success_rate": (
                (len(self.collected_signs) - len(self.failed_signs)) / 
                max(1, len(self.collected_signs)) * 100
            )
        }