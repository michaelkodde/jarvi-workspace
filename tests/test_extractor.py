# tests/test_extractor.py
"""
Unit tests for the Extraction Engine.
Validates DOM parsing, price normalization, and node-level fault tolerance.
"""
import pytest
from src.services.extractor import Extractor, DomainRuleRegistry, PriceNormalizer
from src.services.stealth_engine import ProductRecord


@pytest.fixture
def extractor():
    return Extractor()


def test_price_normalizer_multi_currency():
    """Verify US, UK, and EU price formats are parsed correctly with currency detection."""
    assert PriceNormalizer.parse("$29.99") == (29.99, "USD")
    assert PriceNormalizer.parse("£1,200.50") == (1200.50, "GBP")
    assert PriceNormalizer.parse("€ 1.200,50") == (1200.50, "EUR")
    assert PriceNormalizer.parse("Free") == (0.0, "USD")
    assert PriceNormalizer.parse("N/A") == (None, "USD")
    assert PriceNormalizer.parse("15.00") == (15.00, "USD") # Default fallback


def test_extract_success_multi_domain(extractor):
    """Verify successful extraction using CamelCamelCamel rules."""
    html = """
    <div class="product">
        <h2 class="product-title"><a href="/product/123">Eco Pan</a></h2>
        <span class="price">£29.99</span>
    </div>
    <div class="product">
        <h2 class="product-title"><a href="https://other.com/456">Bamboo Spoon</a></h2>
        <span class="price">€ 1.200,50</span>
    </div>
    """
    products = extractor.extract(html, "camelcamelcamel.com", "eco pans")
    
    assert len(products) == 2
    assert products[0].product_name == "Eco Pan"
    assert products[0].price == 29.99
    assert products[0].currency == "GBP"
    assert products[0].url == "https://camelcamelcamel.com/product/123"
    
    # Verify urljoin handles absolute URLs correctly
    assert products[1].price == 1200.50
    assert products[1].currency == "EUR"
    assert products[1].url == "https://other.com/456"


def test_extract_skips_invalid_nodes(extractor):
    """Verify that nodes with missing prices or names are skipped, not crashed."""
    html = """
    <div class="product">
        <h2 class="product-title"><a href="/valid">Valid Pan</a></h2>
        <span class="price">$10.00</span>
    </div>
    <div class="product">
        <h2 class="product-title"><a href="/missing-price">Missing Price</a></h2>
    </div>
    <div class="product">
        <span class="price">$5.00</span>
    </div>
    """
    products = extractor.extract(html, "camelcamelcamel.com", "pans")
    
    assert len(products) == 1
    assert products[0].product_name == "Valid Pan"


def test_extract_no_rule_returns_empty(extractor):
    """Verify that unknown domains return an empty list gracefully."""
    html = "<html><body>Test</body></html>"
    products = extractor.extract(html, "unknown-domain.com", "test")
    assert products == []


def test_extract_empty_html(extractor):
    """Verify graceful handling of empty or malformed HTML."""
    products = extractor.extract("", "camelcamelcamel.com", "test")
    assert products == []


def test_registry_injection():
    """Verify that a custom registry can be injected for testing."""
    custom_registry = DomainRuleRegistry()
    # Monkey-patch for test
    custom_registry._RULES["test.com"] = custom_registry._RULES["example.com"]
    
    ext = Extractor(registry=custom_registry)
    html = '<div class="product-card"><h2 class="product-title">Test</h2><span class="price">$5</span></div>'
    products = ext.extract(html, "test.com", "test")
    
    assert len(products) == 1
    assert products[0].product_name == "Test"