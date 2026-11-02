# src/services/extractor.py
"""
Extraction Engine (Module 2.1 Extension).
Fast, local DOM parsing using selectolax (lexbor backend) and a domain-rule registry.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

try:
    from selectolax.lexbor import LexborHTMLParser
except ImportError:
    raise ImportError("selectolax is required for the Extraction Engine. Run: pip install selectolax")

from src.services.stealth_engine import ProductRecord

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractionRule",
    "ExtractorError",
    "Extractor",
    "DomainRuleRegistry",
    "PriceNormalizer",
]


# ============================================================================
# SCHEMAS & EXCEPTIONS
# ============================================================================

@dataclass(frozen=True)
class ExtractionRule:
    """CSS selector rules for a specific domain."""
    domain: str
    product_container: str
    name_selector: str
    price_selector: str
    brand_selector: Optional[str] = None
    url_selector: Optional[str] = None
    url_attribute: str = "href"
    asin_selector: Optional[str] = None


class ExtractorError(Exception):
    """Module-level exception for extraction failures."""
    def __init__(self, message: str, error_code: str = "EXTRACTION_FAILURE", original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error


# ============================================================================
# REGISTRY & NORMALIZERS
# ============================================================================

class DomainRuleRegistry:
    """Registry of extraction rules for known domains. Extensible without touching core logic."""
    
    # MVP Rules: Target high-signal aggregators first, not just raw Amazon
    _RULES: Dict[str, ExtractionRule] = {
        "camelcamelcamel.com": ExtractionRule(
            domain="camelcamelcamel.com",
            product_container="div.product",
            name_selector="h2.product-title a",
            price_selector="span.price",
            url_selector="h2.product-title a",
            url_attribute="href",
        ),
        "pricewatcha.com": ExtractionRule(
            domain="pricewatcha.com",
            product_container="div.pw-product-card",
            name_selector="h3.pw-product-name",
            price_selector="span.pw-current-price",
            brand_selector="span.pw-brand",
            url_selector="a.pw-product-link",
            url_attribute="href",
        ),
        "example.com": ExtractionRule(
            domain="example.com",
            product_container="div.product-card",
            name_selector="h2.product-title",
            price_selector="span.price",
            brand_selector="span.brand",
            url_selector="a.product-link",
            url_attribute="href",
        ),
    }

    def get_rule(self, domain: str) -> Optional[ExtractionRule]:
        """Retrieve the extraction rule for a domain."""
        return self._RULES.get(domain)


class PriceNormalizer:
    """Strictly normalizes raw price strings into floats and detects currency."""
    
    # Regex to extract the entire numeric block, including all dots and commas
    _PRICE_REGEX = re.compile(r"([\d.,]+)")
    _CURRENCY_MAP = {"$": "USD", "£": "GBP", "€": "EUR"}

    @classmethod
    def parse(cls, raw_price: Optional[str]) -> Tuple[Optional[float], str]:
        """
        Parse a raw price string into a float and currency code.
        Returns (None, "USD") if unparseable.
        """
        if not raw_price:
            return None, "USD"
        
        raw_price = raw_price.strip()
        if raw_price.lower() in ("free", "n/a", "none", ""):
            return (0.0, "USD") if raw_price.lower() == "free" else (None, "USD")

        # Detect currency
        currency = "USD"
        for symbol, code in cls._CURRENCY_MAP.items():
            if symbol in raw_price:
                currency = code
                break

        match = cls._PRICE_REGEX.search(raw_price)
        if not match:
            return None, currency
        
        numeric_str = match.group(1)
        
        dots = numeric_str.count('.')
        commas = numeric_str.count(',')
        
        if commas == 1 and dots == 0:
            # European decimal (e.g., "1200,50")
            numeric_str = numeric_str.replace(',', '.')
        elif dots == 1 and commas == 0:
            # Could be US decimal ("1.2") or European thousands ("1.200")
            parts = numeric_str.split('.')
            if len(parts) == 2 and len(parts[1]) == 3:
                # Likely European thousands separator
                numeric_str = numeric_str.replace('.', '')
            # Else: US decimal, keep as is
        elif commas > 0 and dots > 0:
            if numeric_str.rfind(',') > numeric_str.rfind('.'):
                # European: "1.200,50"
                numeric_str = numeric_str.replace('.', '').replace(',', '.')
            else:
                # US/UK: "1,200.50"
                numeric_str = numeric_str.replace(',', '')
        else:
            # Fallback for malformed strings: remove all commas, keep dots (US bias)
            numeric_str = numeric_str.replace(',', '')
            
        try:
            return float(numeric_str), currency
        except ValueError:
            return None, currency


# ============================================================================
# CORE EXTRACTOR
# ============================================================================

class Extractor:
    """
    Fast, local DOM parser that extracts ProductRecords from raw HTML.
    
    Args:
        registry: Optional custom DomainRuleRegistry instance (defaults to new instance).
    """

    def __init__(self, registry: Optional[DomainRuleRegistry] = None) -> None:
        self._registry = registry or DomainRuleRegistry()

    def extract(self, html_content: str, domain: str, query: str) -> List[ProductRecord]:
        """
        Extract products from raw HTML based on domain rules.
        
        Args:
            html_content: Raw HTML string from Stealth Engine.
            domain: Canonical target domain.
            query: Original search query (used for logging/context, not extraction).
            
        Returns:
            List of valid ProductRecords. Empty list if no rules or no matches.
        """
        rule = self._registry.get_rule(domain)
        if not rule:
            logger.info("No extraction rule for domain %s. Returning NO_RESULTS.", domain)
            return []

        try:
            tree = LexborHTMLParser(html_content)
        except Exception as e:
            raise ExtractorError(f"Failed to parse HTML for {domain}", original_error=e) from e

        containers = tree.css(rule.product_container)
        if not containers:
            logger.info("No product containers found for %s using selector '%s'", domain, rule.product_container)
            return []

        products = []
        for node in containers:
            try:
                record = self._extract_single_product(node, rule, domain)
                if record:
                    products.append(record)
            except Exception as e:
                # Node-level fault tolerance: skip bad nodes, preserve the batch
                logger.warning("Failed to extract product node in %s: %s", domain, type(e).__name__)
                continue

        logger.info("Extracted %d valid products from %s", len(products), domain)
        return products

    def _extract_single_product(self, node: Any, rule: ExtractionRule, domain: str) -> Optional[ProductRecord]:
        """Extract and validate a single product node."""
        name_node = node.css_first(rule.name_selector)
        price_node = node.css_first(rule.price_selector)
        
        if not name_node or not price_node:
            return None

        raw_name = name_node.text(strip=True)
        raw_price = price_node.text(strip=True)
        
        if not raw_name:
            return None

        price, currency = PriceNormalizer.parse(raw_price)
        if price is None:
            # Spec: Placeholder/invalid products are rejected to pass guard_scout_artifact
            return None

        brand_node = node.css_first(rule.brand_selector) if rule.brand_selector else None
        brand = brand_node.text(strip=True) if brand_node else "Unknown"

        url_node = node.css_first(rule.url_selector) if rule.url_selector else None
        url = ""
        if url_node:
            raw_url = url_node.attributes.get(rule.url_attribute, "")
            # Use urljoin for safe relative/absolute URL resolution
            url = urljoin(f"https://{domain}/", raw_url)

        asin_node = node.css_first(rule.asin_selector) if rule.asin_selector else None
        asin = asin_node.text(strip=True) if asin_node else ""

        return ProductRecord(
            product_name=raw_name,
            brand=brand,
            price=price,
            currency=currency,
            url=url,
            asin=asin,
        )