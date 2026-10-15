"""
Product Scraper: Daemon-safe web scraping using Camofox stealth browser.
"""
import logging
import math
import re
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass
from typing import List

import requests

from src.utils.automation_guard import human_delay, retry_with_backoff

logger = logging.getLogger(__name__)

__all__ = ["ProductData", "ScrapingError", "search_products"]


@dataclass(frozen=True)
class ProductData:
    """Strongly typed representation of a scraped product."""
    product_name: str
    brand: str
    price: float
    currency: str
    url: str
    asin: str


class ScrapingError(Exception):
    """Custom exception for scraping failures."""
    pass


CAMOFOX_BASE_URL = "http://localhost:9377"


def _parse_price(price_str: str) -> float:
    """Parses a locale-aware price string into a finite float."""
    try:
        cleaned = re.sub(r"[^\d.,]", "", price_str).replace(",", ".")
        price = float(cleaned)
        return price if math.isfinite(price) else 0.0
    except (ValueError, TypeError):
        return 0.0


def _check_robots_txt(url: str) -> bool:
    """Checks if the URL is allowed by robots.txt. Fails closed on any error."""
    try:
        parsed = urllib.parse.urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch("*", url)
    except Exception as e:
        logger.warning("Robots.txt check failed (failing closed): %s", e)
        return False


def _check_camofox_available() -> bool:
    """Checks if Camofox service is running."""
    try:
        response = requests.get(f"{CAMOFOX_BASE_URL}/health", timeout=2.0)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@retry_with_backoff
def _extract_products_via_js(tab_id: str) -> List[ProductData]:
    """
    Uses Camofox evaluate endpoint to run JavaScript that extracts product data.
    This is much more reliable than parsing accessibility snapshots.
    """
    js_code = """
    (function() {
        const products = [];
        const items = document.querySelectorAll('[data-asin]');
        
        items.forEach(item => {
            const asin = item.getAttribute('data-asin');
            if (!asin || asin.length !== 10) return;
            
            const titleElem = item.querySelector('.a-text-normal, h2 a span');
            const priceElem = item.querySelector('.a-price .a-offscreen, .a-price-whole');
            const brandElem = item.querySelector('.a-size-small.a-color-secondary, .a-size-base-plus');
            const linkElem = item.querySelector('a.a-link-normal[href*="/dp/"]');
            
            if (!titleElem || !priceElem) return;
            
            const title = titleElem.textContent.trim();
            const priceText = priceElem.textContent.trim();
            const brand = brandElem ? brandElem.textContent.trim() : 'Unknown';
            const url = linkElem ? linkElem.href : 'N/A';
            
            products.push({
                product_name: title,
                brand: brand,
                price_text: priceText,
                url: url,
                asin: asin
            });
        });
        
        return products.slice(0, 10);
    })()
    """
    
    response = requests.post(
        f"{CAMOFOX_BASE_URL}/tabs/{tab_id}/evaluate",
        json={
            "userId": "scout_agent",
            "expression": js_code
        },
        timeout=10.0
    )
    response.raise_for_status()
    
    result = response.json()
    if not result.get("success"):
        raise ScrapingError(f"JavaScript evaluation failed: {result.get('error')}")
    
    raw_products = result.get("result", [])
    
    products = []
    for p in raw_products:
        price = _parse_price(p.get("price_text", ""))
        products.append(ProductData(
            product_name=p.get("product_name", "Unknown"),
            brand=p.get("brand", "Unknown"),
            price=price,
            currency="USD",
            url=p.get("url", "N/A"),
            asin=p.get("asin", "N/A")
        ))
    
    return products


def search_products(query: str) -> List[ProductData]:
    """
    Searches Amazon using Camofox stealth browser.
    Returns a list of ProductData instances. Returns empty list on failure.
    """
    search_url = f"https://www.amazon.com/s?k={urllib.parse.quote(query)}"
    
    # Check robots.txt first (Pillar 1.2 compliance)
    if not _check_robots_txt(search_url):
        logger.warning("Scraping aborted: robots.txt disallows %s", search_url)
        return []
    
    if not _check_camofox_available():
        logger.error("Camofox service is not available at %s", CAMOFOX_BASE_URL)
        return []
    
    tab_id = None
    try:
        # Create a new tab
        response = requests.post(
            f"{CAMOFOX_BASE_URL}/tabs",
            json={
                "userId": "scout_agent",
                "sessionKey": f"search_{int(time.time())}",
                "url": search_url
            },
            timeout=10.0
        )
        response.raise_for_status()
        tab_id = response.json().get("tabId")
        
        if not tab_id:
            raise ScrapingError("Failed to create Camofox tab")
        
        # Wait for page to load
        time.sleep(3.0)
        
        # Extract products via JavaScript
        products = _extract_products_via_js(tab_id)
        
        human_delay()  # Rate limiting
        
        return products
        
    except requests.exceptions.RequestException as e:
        logger.warning("Camofox request failed for query '%s': %s", query, e)
        return []
    except ScrapingError as e:
        logger.warning("Scraping failed for query '%s': %s", query, e)
        return []
    except Exception as e:
        logger.error("Unexpected error during Camofox scraping for query '%s': %s", query, e)
        return []
    finally:
        # Clean up tab only if it was created
        if tab_id:
            try:
                requests.delete(
                    f"{CAMOFOX_BASE_URL}/tabs/{tab_id}",
                    params={"userId": "scout_agent"},
                    timeout=5.0
                )
            except Exception as e:
                logger.warning("Failed to close Camofox tab %s: %s", tab_id, e)