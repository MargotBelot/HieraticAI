"""
SVG Processor Module

This module provides comprehensive SVG processing functionality including
parsing, analysis, manipulation, and conversion for hieroglyphic content.
"""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cairosvg

from ..utils import (
    extract_svg_dimensions,
    safe_file_read,
    safe_file_write,
    timing_decorator,
    validate_svg_content,
)


class SVGProcessor:
    """
    Comprehensive SVG processor for hieroglyphic content.

    This class provides functionality for parsing, analyzing, manipulating,
    and converting SVG files used in hieroglyphic text reconstruction.
    """

    def __init__(
        self, svg_directory: Union[str, Path], logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SVG processor.

        Args:
            svg_directory: Directory containing SVG files
            logger: Optional logger instance
        """
        self.svg_directory = Path(svg_directory)
        self.logger = logger or logging.getLogger("hieroglyph_scraping_toolkit")
        self.svg_cache = {}
        self.metrics_cache = {}

        self.logger.info(
            f"SVG processor initialized with directory: {self.svg_directory}"
        )

    @timing_decorator
    def load_svg_file(self, filename: str) -> Optional[str]:
        """
        Load SVG content from a file with caching.

        Args:
            filename: Name of the SVG file to load

        Returns:
            str or None: SVG content or None if loading failed
        """
        if filename in self.svg_cache:
            return self.svg_cache[filename]

        svg_path = self.svg_directory / filename

        if not svg_path.exists():
            self.logger.warning(f"SVG file not found: {svg_path}")
            return None

        svg_content = safe_file_read(svg_path)

        if svg_content and validate_svg_content(svg_content):
            self.svg_cache[filename] = svg_content
            return svg_content
        else:
            self.logger.error(f"Invalid SVG content in file: {svg_path}")
            return None

    @timing_decorator
    def extract_inner_content(self, svg_content: str) -> str:
        """
        Extract inner SVG content, removing outer <svg> tags.

        This method preserves all child elements while removing the root SVG
        container, allowing for flexible composition in larger SVG documents.

        Args:
            svg_content: Complete SVG content string

        Returns:
            str: Inner SVG content without root tags

        Raises:
            ValueError: If SVG format is invalid
        """
        svg_content = svg_content.strip()

        if not (svg_content.startswith("<svg") and svg_content.endswith("</svg>")):
            raise ValueError(
                "Invalid SVG format: must start with <svg> and end with </svg>"
            )

        try:
            root = ET.fromstring(svg_content)
            inner_elements = []

            for child in root:
                inner_elements.append(ET.tostring(child, encoding="unicode"))

            return "\n".join(inner_elements)

        except ET.ParseError as e:
            raise ValueError(f"Failed to parse SVG content: {e}")

    @timing_decorator
    def analyze_svg_metrics(self, svg_content: str) -> Dict[str, Any]:
        """
        Analyze SVG content to extract comprehensive metrics.

        This method performs sophisticated SVG analysis to determine:
        - Exact bounding box dimensions
        - Path complexity metrics
        - Element distribution
        - Coordinate ranges

        Args:
            svg_content: SVG content to analyze

        Returns:
            dict: Comprehensive metrics dictionary
        """
        try:
            root = ET.fromstring(svg_content)

            # Extract basic dimensions
            viewbox = root.attrib.get("viewBox")
            if viewbox:
                vb_x, vb_y, vb_width, vb_height = map(float, viewbox.strip().split())
            else:
                vb_width = float(root.attrib.get("width", "22.66").replace("px", ""))
                vb_height = float(root.attrib.get("height", "22.04").replace("px", ""))
                vb_x, vb_y = 0, 0

            # Analyze different element types
            elements_analysis = self._analyze_svg_elements(root)

            # Calculate actual content bounds
            content_bounds = self._calculate_content_bounds(
                elements_analysis["coordinates"]
            )

            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(elements_analysis)

            return {
                "viewbox": {
                    "x": vb_x,
                    "y": vb_y,
                    "width": vb_width,
                    "height": vb_height,
                },
                "content_bounds": content_bounds,
                "elements": elements_analysis["counts"],
                "complexity": complexity_metrics,
                "coordinates_range": {
                    "x_range": content_bounds["x_max"] - content_bounds["x_min"],
                    "y_range": content_bounds["y_max"] - content_bounds["y_min"],
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze SVG metrics: {e}")
            return self._get_fallback_metrics()

    def _analyze_svg_elements(self, root: ET.Element) -> Dict[str, Any]:
        """
        Analyze SVG elements to extract coordinates and element counts.

        Args:
            root: SVG root element

        Returns:
            dict: Analysis results with coordinates and element counts
        """
        coordinates = []
        element_counts = {
            "path": 0,
            "rect": 0,
            "circle": 0,
            "ellipse": 0,
            "line": 0,
            "polyline": 0,
            "polygon": 0,
            "text": 0,
        }

        # Find all elements with SVG namespace
        svg_ns = {"svg": "http://www.w3.org/2000/svg"}

        # Process path elements
        for path in root.findall(".//svg:path", svg_ns):
            element_counts["path"] += 1
            d_attr = path.attrib.get("d", "")
            if d_attr:
                path_coords = self._extract_path_coordinates(d_attr)
                coordinates.extend(path_coords)

        # Process rectangles
        for rect in root.findall(".//svg:rect", svg_ns):
            element_counts["rect"] += 1
            x = float(rect.attrib.get("x", 0))
            y = float(rect.attrib.get("y", 0))
            width = float(rect.attrib.get("width", 0))
            height = float(rect.attrib.get("height", 0))
            coordinates.extend([(x, y), (x + width, y + height)])

        # Process circles
        for circle in root.findall(".//svg:circle", svg_ns):
            element_counts["circle"] += 1
            cx = float(circle.attrib.get("cx", 0))
            cy = float(circle.attrib.get("cy", 0))
            r = float(circle.attrib.get("r", 0))
            coordinates.extend([(cx - r, cy - r), (cx + r, cy + r)])

        # Process ellipses
        for ellipse in root.findall(".//svg:ellipse", svg_ns):
            element_counts["ellipse"] += 1
            cx = float(ellipse.attrib.get("cx", 0))
            cy = float(ellipse.attrib.get("cy", 0))
            rx = float(ellipse.attrib.get("rx", 0))
            ry = float(ellipse.attrib.get("ry", 0))
            coordinates.extend([(cx - rx, cy - ry), (cx + rx, cy + ry)])

        # Process lines
        for line in root.findall(".//svg:line", svg_ns):
            element_counts["line"] += 1
            x1 = float(line.attrib.get("x1", 0))
            y1 = float(line.attrib.get("y1", 0))
            x2 = float(line.attrib.get("x2", 0))
            y2 = float(line.attrib.get("y2", 0))
            coordinates.extend([(x1, y1), (x2, y2)])

        # Process polylines and polygons
        for poly in root.findall(".//svg:polyline", svg_ns) + root.findall(
            ".//svg:polygon", svg_ns
        ):
            if poly.tag.endswith("polyline"):
                element_counts["polyline"] += 1
            else:
                element_counts["polygon"] += 1

            points = poly.attrib.get("points", "")
            if points:
                poly_coords = self._extract_points_coordinates(points)
                coordinates.extend(poly_coords)

        # Count text elements (don't extract coordinates as they're different)
        element_counts["text"] = len(root.findall(".//svg:text", svg_ns))

        return {"coordinates": coordinates, "counts": element_counts}

    def _extract_path_coordinates(self, path_data: str) -> List[Tuple[float, float]]:
        """
        Extract coordinate pairs from SVG path data.

        This method uses regex to parse SVG path commands and extract
        all coordinate information for bounding box calculation.

        Args:
            path_data: SVG path data string

        Returns:
            List[Tuple[float, float]]: List of coordinate pairs
        """
        # Extract all numeric values from path data
        numbers = re.findall(r"-?\d+\.?\d*", path_data)

        # Convert to float and pair up as coordinates
        coords = []
        for i in range(0, len(numbers) - 1, 2):
            try:
                x, y = float(numbers[i]), float(numbers[i + 1])
                coords.append((x, y))
            except (ValueError, IndexError):
                continue

        return coords

    def _extract_points_coordinates(
        self, points_data: str
    ) -> List[Tuple[float, float]]:
        """
        Extract coordinates from polyline/polygon points attribute.

        Args:
            points_data: Points attribute string

        Returns:
            List[Tuple[float, float]]: List of coordinate pairs
        """
        coords = []
        numbers = re.findall(r"-?\d+\.?\d*", points_data)

        for i in range(0, len(numbers) - 1, 2):
            try:
                x, y = float(numbers[i]), float(numbers[i + 1])
                coords.append((x, y))
            except (ValueError, IndexError):
                continue

        return coords

    def _calculate_content_bounds(
        self, coordinates: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Calculate the bounding box from a list of coordinates.

        Args:
            coordinates: List of coordinate pairs

        Returns:
            dict: Bounding box information
        """
        if not coordinates:
            return {
                "x_min": 0,
                "y_min": 0,
                "x_max": 0,
                "y_max": 0,
                "width": 0,
                "height": 0,
            }

        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": x_max - x_min,
            "height": y_max - y_min,
        }

    def _calculate_complexity_metrics(
        self, elements_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate complexity metrics based on element analysis.

        Args:
            elements_analysis: Results from element analysis

        Returns:
            dict: Complexity metrics
        """
        element_counts = elements_analysis["counts"]
        coordinate_count = len(elements_analysis["coordinates"])

        # Calculate total elements
        total_elements = sum(element_counts.values())

        # Calculate complexity score (weighted by element type)
        complexity_weights = {
            "path": 3.0,
            "polygon": 2.0,
            "polyline": 2.0,
            "circle": 1.5,
            "ellipse": 1.5,
            "rect": 1.0,
            "line": 0.5,
            "text": 1.0,
        }

        weighted_complexity = sum(
            count * complexity_weights.get(element_type, 1.0)
            for element_type, count in element_counts.items()
        )

        return {
            "total_elements": total_elements,
            "coordinate_count": coordinate_count,
            "weighted_complexity": weighted_complexity,
            "complexity_score": min(
                weighted_complexity / 10.0, 5.0
            ),  # Normalize to 0-5 scale
            "element_distribution": element_counts,
        }

    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """
        Get fallback metrics when analysis fails.

        Returns:
            dict: Default metrics
        """
        return {
            "viewbox": {"x": 0, "y": 0, "width": 22.66, "height": 22.04},
            "content_bounds": {
                "x_min": 0,
                "y_min": 0,
                "x_max": 22.66,
                "y_max": 22.04,
                "width": 22.66,
                "height": 22.04,
            },
            "elements": {"path": 0, "rect": 0, "circle": 0, "ellipse": 0},
            "complexity": {
                "total_elements": 1,
                "coordinate_count": 0,
                "weighted_complexity": 1.0,
                "complexity_score": 1.0,
            },
        }

    @timing_decorator
    def calculate_average_dimensions(
        self, svg_filenames: List[str]
    ) -> Dict[str, float]:
        """
        Calculate average dimensions across multiple SVG files.

        Args:
            svg_filenames: List of SVG filenames to analyze

        Returns:
            dict: Average dimensions and statistics
        """
        widths, heights = [], []
        processed_count = 0

        for filename in svg_filenames:
            svg_content = self.load_svg_file(filename)
            if svg_content:
                dimensions = extract_svg_dimensions(svg_content)
                if dimensions:
                    width, height = dimensions
                    widths.append(width)
                    heights.append(height)
                    processed_count += 1

        if not widths or not heights:
            self.logger.warning("No valid SVG dimensions found")
            return {
                "average_width": 22.66,
                "average_height": 22.04,
                "processed_count": 0,
            }

        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)

        return {
            "average_width": avg_width,
            "average_height": avg_height,
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "processed_count": processed_count,
            "total_files": len(svg_filenames),
        }

    @timing_decorator
    def convert_svg_to_png(
        self, svg_path: Union[str, Path], png_path: Union[str, Path]
    ) -> bool:
        """
        Convert SVG file to PNG format.

        Args:
            svg_path: Path to source SVG file
            png_path: Path to output PNG file

        Returns:
            bool: True if conversion was successful
        """
        try:
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
            self.logger.debug(f"SVG converted to PNG: {png_path}")
            return True
        except Exception as e:
            self.logger.error(f"SVG to PNG conversion failed: {e}")
            return False

    @timing_decorator
    def create_svg_sprite(
        self, svg_files: List[str], output_path: Union[str, Path]
    ) -> bool:
        """
        Create an SVG sprite from multiple SVG files.

        Args:
            svg_files: List of SVG filenames to include
            output_path: Path to output sprite file

        Returns:
            bool: True if sprite creation was successful
        """
        try:
            sprite_content = ['<svg xmlns="http://www.w3.org/2000/svg">']
            sprite_content.append("<defs>")

            x_offset = 0
            total_width = 0
            max_height = 0

            for i, filename in enumerate(svg_files):
                svg_content = self.load_svg_file(filename)
                if not svg_content:
                    continue

                # Extract dimensions
                dimensions = extract_svg_dimensions(svg_content)
                if not dimensions:
                    continue

                width, height = dimensions
                max_height = max(max_height, height)

                # Extract inner content
                try:
                    inner_content = self.extract_inner_content(svg_content)
                    sprite_content.append(
                        f'<g id="glyph_{i}" transform="translate({x_offset}, 0)">'
                    )
                    sprite_content.append(inner_content)
                    sprite_content.append("</g>")

                    x_offset += width
                    total_width += width

                except ValueError as e:
                    self.logger.warning(f"Skipping invalid SVG {filename}: {e}")
                    continue

            sprite_content.append("</defs>")
            sprite_content.append(
                f'<rect width="{total_width}" height="{max_height}" fill="white"/>'
            )
            sprite_content.append("</svg>")

            sprite_svg = "\n".join(sprite_content)

            return safe_file_write(output_path, sprite_svg)

        except Exception as e:
            self.logger.error(f"Failed to create SVG sprite: {e}")
            return False

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get SVG processor cache statistics.

        Returns:
            dict: Cache statistics
        """
        return {
            "svg_cache_size": len(self.svg_cache),
            "metrics_cache_size": len(self.metrics_cache),
            "svg_directory": str(self.svg_directory),
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.svg_cache.clear()
        self.metrics_cache.clear()
        self.logger.info("SVG processor caches cleared")
