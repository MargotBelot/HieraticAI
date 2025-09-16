"""
Configuration Management Module

This module provides configuration management for the hieroglyph toolkit,
handling paths, settings, and environment-specific configurations.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class WebScrapingConfig:
    """Configuration for web scraping operations."""
    
    base_url_aku: str = "https://aku-pal.uni-mainz.de"
    base_url_signs: str = "https://aku-pal.uni-mainz.de/signs/"
    graphemes_url: str = "https://aku-pal.uni-mainz.de/graphemes#dating=Zweite%20Zwischenzeit%2C%2017.%20Dynastie"
    
    # Selenium configuration
    headless_mode: bool = True
    implicit_wait: int = 10
    page_load_timeout: int = 30
    request_delay: float = 2.0
    
    # Output directories
    output_json_dir: str = "output/json"
    output_txt_dir: str = "output/txt"
    output_svg_dir: str = "output/svg"


@dataclass
class LayoutConfiguration:
    """Configuration parameters for spatial encoding and layout system."""
    
    # Base spacing parameters
    base_horizontal_spacing: float = 2.5
    base_vertical_spacing: float = 3.0
    
    # Scaling factors for different contexts
    word_spacing_factor: float = 1.6
    line_spacing_factor: float = 1.4
    group_spacing_factor: float = 0.8
    
    # Typography parameters
    baseline_alignment: bool = True
    optical_spacing: bool = True
    size_normalization: bool = True
    
    # Quality parameters
    min_glyph_size: float = 8.0
    max_glyph_size: float = 48.0
    target_line_height: float = 28.0
    
    # Performance parameters
    enable_caching: bool = True
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class PathConfig:
    """Configuration for file and directory paths."""
    
    # Base directories - these should be set by the user
    project_root: Path = field(default_factory=lambda: Path.home())
    svg_directory: Optional[Path] = None
    gardiner_mapping_file: Optional[Path] = None
    input_document: Optional[Path] = None
    output_directory: Path = field(default_factory=lambda: Path.home() / "hieroglyph_output")
    
    def __post_init__(self):
        """Ensure all paths are Path objects and create output directory if needed."""
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
        if self.svg_directory and isinstance(self.svg_directory, str):
            self.svg_directory = Path(self.svg_directory)
        if self.gardiner_mapping_file and isinstance(self.gardiner_mapping_file, str):
            self.gardiner_mapping_file = Path(self.gardiner_mapping_file)
        if self.input_document and isinstance(self.input_document, str):
            self.input_document = Path(self.input_document)
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)
        
        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """
    Manages configuration for the hieroglyph toolkit.
    
    This class handles loading configuration from files, environment variables,
    and provides defaults for all configuration parameters.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Optional path to configuration file (JSON format)
        """
        self.config_file = Path(config_file) if config_file else None
        self._config_data = {}
        
        # Load configuration
        if self.config_file and self.config_file.exists():
            self.load_from_file()
        else:
            self.load_defaults()
    
    def load_from_file(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self._config_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
            self.load_defaults()
    
    def load_defaults(self) -> None:
        """Load default configuration values."""
        self._config_data = {
            "paths": {},
            "layout": {},
            "web_scraping": {}
        }
    
    def save_to_file(self, config_file: Optional[str] = None) -> None:
        """
        Save current configuration to JSON file.
        
        Args:
            config_file: Optional path to save configuration to 
        """
        file_path = Path(config_file) if config_file else self.config_file
        if not file_path:
            raise ValueError("No configuration file specified")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert Path objects to strings for JSON serialization
        config_to_save = self._serialize_config(self._config_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
    
    def _serialize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Path objects to strings for JSON serialization."""
        serialized = {}
        for key, value in config.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_config(value)
            elif isinstance(value, Path):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized
    
    def get_path_config(self) -> PathConfig:
        """
        Get path configuration.
        
        Returns:
            PathConfig: Configured paths
        """
        path_data = self._config_data.get("paths", {})
        
        # Set defaults from environment or user home
        home_dir = Path.home()
        
        return PathConfig(
            project_root=Path(path_data.get("project_root", home_dir)),
            svg_directory=Path(path_data["svg_directory"]) if path_data.get("svg_directory") else None,
            gardiner_mapping_file=Path(path_data["gardiner_mapping_file"]) if path_data.get("gardiner_mapping_file") else None,
            input_document=Path(path_data["input_document"]) if path_data.get("input_document") else None,
            output_directory=Path(path_data.get("output_directory", home_dir / "hieroglyph_output"))
        )
    
    def get_layout_config(self) -> LayoutConfiguration:
        """
        Get layout configuration.
        
        Returns:
            LayoutConfiguration: Layout settings
        """
        layout_data = self._config_data.get("layout", {})
        return LayoutConfiguration(**layout_data)
    
    def get_web_scraping_config(self) -> WebScrapingConfig:
        """
        Get web scraping configuration.
        
        Returns:
            WebScrapingConfig: Web scraping settings
        """
        scraping_data = self._config_data.get("web_scraping", {})
        return WebScrapingConfig(**scraping_data)
    
    def set_paths(self, **kwargs) -> None:
        """
        Set path configuration.
        
        Args:
            **kwargs: Path parameters to set
        """
        if "paths" not in self._config_data:
            self._config_data["paths"] = {}
        
        for key, value in kwargs.items():
            if isinstance(value, Path):
                self._config_data["paths"][key] = str(value)
            else:
                self._config_data["paths"][key] = value
    
    def update_layout_config(self, **kwargs) -> None:
        """
        Update layout configuration.
        
        Args:
            **kwargs: Layout parameters to update
        """
        if "layout" not in self._config_data:
            self._config_data["layout"] = {}
        
        self._config_data["layout"].update(kwargs)
    
    def update_web_scraping_config(self, **kwargs) -> None:
        """
        Update web scraping configuration.
        
        Args:
            **kwargs: Web scraping parameters to update
        """
        if "web_scraping" not in self._config_data:
            self._config_data["web_scraping"] = {}
        
        self._config_data["web_scraping"].update(kwargs)


def create_default_config_file(file_path: str) -> None:
    """
    Create a default configuration file with example settings.
    
    Args:
        file_path: Path where to create the configuration file
    """
    home_dir = str(Path.home())
    icloud_path = f"{home_dir}/Library/Mobile Documents/com~apple~CloudDocs"
    
    default_config = {
        "paths": {
            "project_root": f"{icloud_path}/FU/ALP/Project",
            "svg_directory": f"{icloud_path}/FU/ALP/Project/aku_scraping/svg",
            "gardiner_mapping_file": f"{icloud_path}/FU/ALP/Project/Outputs/gardiner_to_svgs.json",
            "input_document": f"{icloud_path}/FU/ALP/Project/Lines.docx",
            "output_directory": f"{home_dir}/Desktop/hieroglyph_output"
        },
        "layout": {
            "base_horizontal_spacing": 2.5,
            "base_vertical_spacing": 3.0,
            "word_spacing_factor": 1.6,
            "line_spacing_factor": 1.4,
            "baseline_alignment": True,
            "optical_spacing": True,
            "size_normalization": True,
            "target_line_height": 28.0,
            "enable_caching": True,
            "parallel_processing": True,
            "max_workers": 4
        },
        "web_scraping": {
            "headless_mode": True,
            "implicit_wait": 10,
            "page_load_timeout": 30,
            "request_delay": 2.0,
            "output_json_dir": "output/json",
            "output_txt_dir": "output/txt",
            "output_svg_dir": "output/svg"
        }
    }
    
    config_path = Path(file_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)


# Convenience function to get a configured manager
def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """
    Get a configured ConfigManager instance.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        ConfigManager: Configured manager instance
    """
    if config_file is None:
        # Look for config file in common locations
        possible_configs = [
            Path.cwd() / "config.json",
            Path.home() / ".hieroglyph_config.json",
            Path(__file__).parent / "config.json"
        ]
        
        for config_path in possible_configs:
            if config_path.exists():
                config_file = str(config_path)
                break
    
    return ConfigManager(config_file)