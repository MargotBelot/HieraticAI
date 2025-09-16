"""
Utilities Module

This module provides common utility functions for the hieroglyph toolkit,
including file operations, logging setup, and error handling utilities.
"""

import os
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps
import xml.etree.ElementTree as ET


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    include_console: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging for the application.
    
    Args:
        log_file: Optional path to log file
        log_level: Logging level (default: INFO)
        include_console: Whether to include console output
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("hieroglyph_scraping_toolkit")
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path: The ensured directory path
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_file_read(
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    fallback_content: Optional[str] = None
) -> Optional[str]:
    """
    Safely read a file with error handling.
    
    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
        fallback_content: Content to return if reading fails
        
    Returns:
        str or None: File content or fallback content
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger = logging.getLogger("hieroglyph_scraping_toolkit")
        logger.warning(f"Failed to read file {file_path}: {e}")
        return fallback_content


def safe_file_write(
    file_path: Union[str, Path],
    content: str,
    encoding: str = 'utf-8',
    create_dirs: bool = True
) -> bool:
    """
    Safely write content to a file with error handling.
    
    Args:
        file_path: Path to the file to write
        content: Content to write
        encoding: File encoding (default: utf-8)
        create_dirs: Whether to create parent directories
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger = logging.getLogger("hieroglyph_scraping_toolkit")
        logger.error(f"Failed to write file {file_path}: {e}")
        return False


def safe_json_load(
    file_path: Union[str, Path],
    fallback_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Safely load JSON data from a file with error handling.
    
    Args:
        file_path: Path to the JSON file
        fallback_data: Data to return if loading fails
        
    Returns:
        dict: Loaded JSON data or fallback data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger = logging.getLogger("hieroglyph_scraping_toolkit")
        logger.warning(f"Failed to load JSON from {file_path}: {e}")
        return fallback_data or {}


def safe_json_save(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    indent: int = 2,
    create_dirs: bool = True
) -> bool:
    """
    Safely save data to a JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to the JSON file
        indent: JSON indentation level
        create_dirs: Whether to create parent directories
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logger = logging.getLogger("hieroglyph_scraping_toolkit")
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Callable: Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger("hieroglyph_scraping_toolkit")
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.3f}s")
        
        return result
    return wrapper


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator to retry function execution on specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Callable: Wrapped function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("hieroglyph_scraping_toolkit")
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}")
                    if delay > 0:
                        time.sleep(delay)
            
        return wrapper
    return decorator


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        str: Generated cache key
    """
    key_string = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_string.encode()).hexdigest()


def validate_svg_content(svg_content: str) -> bool:
    """
    Validate SVG content by attempting to parse it.
    
    Args:
        svg_content: SVG content string
        
    Returns:
        bool: True if valid SVG, False otherwise
    """
    try:
        ET.fromstring(svg_content)
        return True
    except ET.ParseError:
        return False


def extract_svg_dimensions(svg_content: str) -> Optional[tuple]:
    """
    Extract dimensions from SVG content.
    
    Args:
        svg_content: SVG content string
        
    Returns:
        tuple or None: (width, height) if found, None otherwise
    """
    try:
        root = ET.fromstring(svg_content)
        
        # Try viewBox first
        viewbox = root.attrib.get('viewBox')
        if viewbox:
            vb_x, vb_y, vb_width, vb_height = map(float, viewbox.strip().split())
            return vb_width, vb_height
        
        # Fall back to width/height attributes
        width = root.attrib.get('width')
        height = root.attrib.get('height')
        
        if width and height:
            width = float(width.replace('px', ''))
            height = float(height.replace('px', ''))
            return width, height
            
    except Exception:
        pass
    
    return None


def clean_filename(filename: str, max_length: int = 255) -> str:
    """
    Clean a filename to be filesystem-safe.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        str: Cleaned filename
    """
    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Trim to maximum length
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext
    
    return filename


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        dict: File information
    """
    path = Path(file_path)
    
    if not path.exists():
        return {"exists": False, "path": str(path)}
    
    stat = path.stat()
    
    return {
        "exists": True,
        "path": str(path),
        "name": path.name,
        "stem": path.stem,
        "suffix": path.suffix,
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "modified_time": stat.st_mtime,
        "created_time": stat.st_ctime,
        "is_file": path.is_file(),
        "is_directory": path.is_dir()
    }


def progress_bar(
    current: int,
    total: int,
    prefix: str = "Progress",
    suffix: str = "Complete",
    length: int = 50
) -> str:
    """
    Generate a simple progress bar string.
    
    Args:
        current: Current progress value
        total: Total progress value
        prefix: Prefix text
        suffix: Suffix text
        length: Progress bar length
        
    Returns:
        str: Formatted progress bar string
    """
    if total == 0:
        percent = 100
    else:
        percent = min(100, int(100 * current / total))
    
    filled_length = int(length * percent / 100)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    return f'\r{prefix} |{bar}| {percent:.1f}% {suffix}'


class PerformanceTracker:
    """
    Simple performance tracking utility.
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.timings = {}
        self.counters = {}
        self.start_times = {}
    
    def start_timer(self, name: str) -> None:
        """
        Start a named timer.
        
        Args:
            name: Timer name
        """
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """
        End a named timer and record the duration.
        
        Args:
            name: Timer name
            
        Returns:
            float: Elapsed time in seconds
        """
        if name not in self.start_times:
            return 0.0
        
        elapsed = time.time() - self.start_times[name]
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)
        
        del self.start_times[name]
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """
        Increment a named counter.
        
        Args:
            name: Counter name
            value: Value to increment by
        """
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        stats = {"counters": self.counters.copy(), "timings": {}}
        
        for name, times in self.timings.items():
            if times:
                stats["timings"][name] = {
                    "count": len(times),
                    "total": sum(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times)
                }
        
        return stats
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.timings.clear()
        self.counters.clear()
        self.start_times.clear()


# Global performance tracker instance
performance_tracker = PerformanceTracker()