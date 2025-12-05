"""
Base Web Scraper Module

This module provides the base web scraper class with common functionality
for all web scraping operations in the hieroglyph toolkit.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from ..config import WebScrapingConfig
from ..utils import (
    ensure_directory,
    retry_on_exception,
    setup_logging,
    timing_decorator,
)


class BaseScraper:
    """
    Base web scraper class providing common functionality for all scrapers.

    This class handles browser setup, common navigation patterns, error handling,
    and provides utilities for robust web scraping operations.
    """

    def __init__(self, config: WebScrapingConfig, log_file: Optional[str] = None):
        """
        Initialize the base scraper.

        Args:
            config: Web scraping configuration
            log_file: Optional log file path
        """
        self.config = config
        self.logger = (
            setup_logging(log_file)
            if log_file
            else logging.getLogger("hieroglyph_scraping_toolkit")
        )
        self.driver: Optional[webdriver.Chrome] = None
        self._setup_driver()

        # Create output directories
        ensure_directory(self.config.output_json_dir)
        ensure_directory(self.config.output_txt_dir)
        if hasattr(self.config, "output_svg_dir"):
            ensure_directory(self.config.output_svg_dir)

    def _setup_driver(self) -> None:
        """Set up the Chrome WebDriver with appropriate options."""
        try:
            chrome_options = Options()

            if self.config.headless_mode:
                chrome_options.add_argument("--headless")

            # Additional Chrome options for stability
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(
                "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)

            # Set timeouts
            self.driver.implicitly_wait(self.config.implicit_wait)
            self.driver.set_page_load_timeout(self.config.page_load_timeout)

            self.logger.info("Chrome WebDriver initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    @retry_on_exception(
        max_retries=3, delay=2.0, exceptions=(TimeoutException, WebDriverException)
    )
    def navigate_to_url(self, url: str) -> bool:
        """
        Navigate to a URL with retry logic.

        Args:
            url: URL to navigate to

        Returns:
            bool: True if navigation was successful
        """
        try:
            self.logger.debug(f"Navigating to: {url}")
            self.driver.get(url)
            return True
        except Exception as e:
            self.logger.warning(f"Navigation failed for {url}: {e}")
            return False

    def wait_for_element(
        self, locator: tuple, timeout: int = None, condition=None
    ) -> Optional[Any]:
        """
        Wait for an element to be present with optional condition.

        Args:
            locator: Selenium locator tuple (By.TYPE, "selector")
            timeout: Wait timeout (uses config default if None)
            condition: Expected condition (defaults to presence_of_element_located)

        Returns:
            WebElement or None: Found element or None if timeout
        """
        timeout = timeout or self.config.implicit_wait
        condition = condition or EC.presence_of_element_located

        try:
            wait = WebDriverWait(self.driver, timeout)
            element = wait.until(condition(locator))
            return element
        except TimeoutException:
            self.logger.warning(f"Element not found within {timeout}s: {locator}")
            return None

    def wait_for_elements(
        self, locator: tuple, timeout: int = None, condition=None
    ) -> List[Any]:
        """
        Wait for multiple elements to be present.

        Args:
            locator: Selenium locator tuple (By.TYPE, "selector")
            timeout: Wait timeout (uses config default if None)
            condition: Expected condition (defaults to presence_of_all_elements_located)

        Returns:
            List[WebElement]: Found elements (empty list if timeout)
        """
        timeout = timeout or self.config.implicit_wait
        condition = condition or EC.presence_of_all_elements_located

        try:
            wait = WebDriverWait(self.driver, timeout)
            elements = wait.until(condition(locator))
            return elements
        except TimeoutException:
            self.logger.warning(f"Elements not found within {timeout}s: {locator}")
            return []

    def safe_find_element(self, locator: tuple) -> Optional[Any]:
        """
        Safely find a single element without waiting.

        Args:
            locator: Selenium locator tuple (By.TYPE, "selector")

        Returns:
            WebElement or None: Found element or None if not found
        """
        try:
            return self.driver.find_element(*locator)
        except Exception:
            return None

    def safe_find_elements(self, locator: tuple) -> List[Any]:
        """
        Safely find multiple elements without waiting.

        Args:
            locator: Selenium locator tuple (By.TYPE, "selector")

        Returns:
            List[WebElement]: Found elements (empty list if none found)
        """
        try:
            return self.driver.find_elements(*locator)
        except Exception:
            return []

    def extract_text_safely(self, element: Any, strip: bool = True) -> str:
        """
        Safely extract text from an element.

        Args:
            element: WebElement to extract text from
            strip: Whether to strip whitespace

        Returns:
            str: Extracted text (empty string if extraction fails)
        """
        try:
            text = element.text
            return text.strip() if strip else text
        except Exception:
            return ""

    def extract_attribute_safely(self, element: Any, attribute: str) -> Optional[str]:
        """
        Safely extract an attribute from an element.

        Args:
            element: WebElement to extract attribute from
            attribute: Attribute name to extract

        Returns:
            str or None: Attribute value or None if extraction fails
        """
        try:
            return element.get_attribute(attribute)
        except Exception:
            return None

    def wait_between_requests(self) -> None:
        """Add delay between requests to be respectful to the server."""
        if self.config.request_delay > 0:
            time.sleep(self.config.request_delay)

    @timing_decorator
    def scrape_with_retries(
        self, scrape_function: callable, *args, max_retries: int = 3, **kwargs
    ) -> Any:
        """
        Execute a scraping function with retry logic.

        Args:
            scrape_function: Function to execute
            *args: Function arguments
            max_retries: Maximum retry attempts
            **kwargs: Function keyword arguments

        Returns:
            Any: Function result or None if all retries failed
        """
        for attempt in range(max_retries + 1):
            try:
                return scrape_function(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    self.logger.error(
                        f"Scraping failed after {max_retries} retries: {e}"
                    )
                    return None

                self.logger.warning(f"Scraping attempt {attempt + 1} failed: {e}")
                self.wait_between_requests()

    def get_current_url(self) -> str:
        """
        Get the current page URL.

        Returns:
            str: Current URL
        """
        try:
            return self.driver.current_url
        except Exception:
            return ""

    def get_page_title(self) -> str:
        """
        Get the current page title.

        Returns:
            str: Page title
        """
        try:
            return self.driver.title
        except Exception:
            return ""

    def save_page_source(self, file_path: str) -> bool:
        """
        Save the current page source to a file.

        Args:
            file_path: Path to save the page source

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save page source: {e}")
            return False

    def take_screenshot(self, file_path: str) -> bool:
        """
        Take a screenshot of the current page.

        Args:
            file_path: Path to save the screenshot

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return self.driver.save_screenshot(file_path)
        except Exception as e:
            self.logger.error(f"Failed to take screenshot: {e}")
            return False

    def scroll_to_element(self, element: Any) -> bool:
        """
        Scroll to bring an element into view.

        Args:
            element: WebElement to scroll to

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.driver.execute_script("arguments[0].scrollIntoView();", element)
            return True
        except Exception as e:
            self.logger.error(f"Failed to scroll to element: {e}")
            return False

    def execute_javascript(self, script: str) -> Any:
        """
        Execute JavaScript in the current page context.

        Args:
            script: JavaScript code to execute

        Returns:
            Any: JavaScript execution result
        """
        try:
            return self.driver.execute_script(script)
        except Exception as e:
            self.logger.error(f"JavaScript execution failed: {e}")
            return None

    def get_page_performance_metrics(self) -> Dict[str, Any]:
        """
        Get basic page performance metrics.

        Returns:
            dict: Performance metrics
        """
        try:
            navigation_timing = self.execute_javascript(
                "return window.performance.timing"
            )

            if navigation_timing:
                load_time = navigation_timing.get(
                    "loadEventEnd", 0
                ) - navigation_timing.get("navigationStart", 0)
                dom_ready = navigation_timing.get(
                    "domContentLoadedEventEnd", 0
                ) - navigation_timing.get("navigationStart", 0)

                return {
                    "page_load_time_ms": load_time,
                    "dom_ready_time_ms": dom_ready,
                    "current_url": self.get_current_url(),
                    "page_title": self.get_page_title(),
                }
        except Exception:
            pass

        return {}

    def cleanup(self) -> None:
        """Clean up resources and close the browser."""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("WebDriver cleaned up successfully")
            except Exception as e:
                self.logger.error(f"Error during WebDriver cleanup: {e}")
            finally:
                self.driver = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def __del__(self):
        """Destructor with cleanup."""
        self.cleanup()
