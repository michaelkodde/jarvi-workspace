# tests/test_stealth_engine.py
"""
Unit tests for the Native Stealth Engine.
Validates tiered retrieval, circuit breaker integration, cooldown logic, and PII safety.
"""
import time
import pytest
from unittest.mock import MagicMock, patch

from src.services.stealth_engine import (
    StealthEngine,
    Outcome,
    SiteProfile,
    ProductRecord,
)
from src.utils.llm_client import CircuitBreaker, CircuitState


class MockMemoryEngine:
    def __init__(self):
        self.profiles = {}
        self.telemetry = []

    def get_site_profile(self, domain: str):
        return self.profiles.get(domain)

    def save_site_profile(self, profile: SiteProfile):
        self.profiles[profile.domain] = profile

    def log_telemetry(self, domain: str, tier: int, outcome: str, duration: float):
        self.telemetry.append({"domain": domain, "tier": tier, "outcome": outcome, "duration": duration})


@pytest.fixture
def memory_engine():
    return MockMemoryEngine()


@pytest.fixture
def circuit_breaker():
    return CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)


@pytest.fixture
def stealth_engine(memory_engine, circuit_breaker):
    return StealthEngine(
        memory_engine=memory_engine,
        circuit_breaker=circuit_breaker,
    )


def test_policy_gate_prohibited_domain(stealth_engine):
    """Verify that prohibited domains are immediately DISALLOWED."""
    envelope = stealth_engine.search_products("test query", "localhost")
    assert envelope.outcome == Outcome.DISALLOWED
    assert envelope.reason == "Domain prohibited"
    assert envelope.tier_used == 0


def test_policy_gate_active_cooldown(stealth_engine, memory_engine):
    """Verify that domains in active cooldown are DISALLOWED."""
    profile = SiteProfile(
        domain="example.com",
        cooldown_expiry=time.time() + 3600  # 1 hour from now
    )
    memory_engine.save_site_profile(profile)
    
    envelope = stealth_engine.search_products("test query", "example.com")
    assert envelope.outcome == Outcome.DISALLOWED
    assert "Active cooldown" in envelope.reason


@patch("src.services.stealth_engine.curl_requests")
def test_tier1_success(mock_curl, stealth_engine, memory_engine):
    """Verify successful Tier 1 retrieval and profile update."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.text = "<html><body>Mock Product Data</body></html>"
    mock_session.get.return_value.raise_for_status = MagicMock()
    mock_curl.Session.return_value = mock_session

    # Mock extraction to return a product
    mock_strategy = MagicMock()
    mock_strategy.extract.return_value = [
        ProductRecord(product_name="Pan", brand="EcoBrand", price=29.99, currency="USD", url="http://example.com/pan", asin="B012345678")
    ]
    stealth_engine._extraction_strategy = mock_strategy

    envelope = stealth_engine.search_products("eco pan", "example.com")
    
    assert envelope.outcome == Outcome.RETRIEVED
    assert envelope.tier_used == 1
    assert len(envelope.products) == 1
    assert memory_engine.profiles["example.com"].last_success_timestamp is not None
    assert memory_engine.profiles["example.com"].consecutive_failures == 0


@patch("src.services.stealth_engine.curl_requests")
def test_tier1_blocked_triggers_cooldown(mock_curl, stealth_engine, memory_engine):
    """Verify that 403/429 responses trigger BLOCKED outcome and escalating cooldown."""
    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.get.return_value.status_code = 403
    mock_curl.Session.return_value = mock_session

    envelope = stealth_engine.search_products("eco pan", "example.com")
    
    assert envelope.outcome == Outcome.BLOCKED
    assert envelope.tier_used == 1
    
    profile = memory_engine.profiles["example.com"]
    assert profile.consecutive_failures == 1
    assert profile.cooldown_level == 1
    assert profile.cooldown_expiry is not None
    assert profile.cooldown_expiry > time.time()


def test_circuit_breaker_open_bypasses_network(stealth_engine, circuit_breaker):
    """Verify that an OPEN circuit breaker returns UNAVAILABLE instantly."""
    # Force circuit breaker open
    circuit_breaker.state = CircuitState.OPEN
    circuit_breaker.last_failure_time = time.time()
    
    envelope = stealth_engine.search_products("test", "example.com")
    
    assert envelope.outcome == Outcome.UNAVAILABLE
    assert "Circuit breaker OPEN" in envelope.reason
    assert envelope.elapsed_seconds < 1.0  # Should be near-instant


def test_memory_engine_failure_graceful_degradation():
    """Verify that MemoryEngine failures do not crash the engine."""
    broken_memory_engine = MagicMock()
    broken_memory_engine.get_site_profile.side_effect = Exception("DB Locked")
    broken_memory_engine.save_site_profile.side_effect = Exception("DB Locked")
    broken_memory_engine.log_telemetry.side_effect = Exception("DB Locked")
    
    failing_engine = StealthEngine(
        memory_engine=broken_memory_engine,
        circuit_breaker=CircuitBreaker(),
    )
    
    # Should not raise, but return UNAVAILABLE due to mocked network failure
    with patch("src.services.stealth_engine.curl_requests") as mock_curl:
        mock_curl.Session.side_effect = Exception("Network fail")
        envelope = failing_engine.search_products("test", "example.com")
        
    assert envelope.outcome == Outcome.UNAVAILABLE


def test_no_pii_in_logs(stealth_engine, caplog):
    """Verify that the search query is NEVER logged."""
    with patch("src.services.stealth_engine.curl_requests") as mock_curl:
        mock_curl.Session.side_effect = Exception("Fail")
        with caplog.at_level("INFO"):
            stealth_engine.search_products("super secret proprietary product name", "example.com")
            
    for record in caplog.records:
        assert "super secret proprietary product name" not in record.message
        assert "secret" not in record.message.lower()