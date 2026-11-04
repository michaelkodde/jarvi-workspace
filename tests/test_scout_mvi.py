# tests/test_scout_mvi.py
"""
Unit tests for ScoutMVI agent.
Validates MVI pipeline orchestration, error handling, and dependency injection.
"""
import json
import pytest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.scout_mvi import ScoutMVI
from src.services.health_monitor import HealthMonitor, HealthReport, HealthStatus
from src.services.stealth_engine import Outcome, ProductRecord, RetrievalEnvelope, StealthEngine
from src.services.step_adapters import AdapterError


@pytest.fixture
def mock_stealth_engine():
    engine = MagicMock(spec=StealthEngine)
    engine._extraction_strategy = MagicMock()
    engine._extraction_strategy._registry = MagicMock()
    engine._extraction_strategy._registry._RULES = {
        "example.com": MagicMock(),
        "camelcamelcamel.com": MagicMock(),
    }
    engine._extraction_strategy._registry.get_rule = lambda domain: engine._extraction_strategy._registry._RULES.get(domain)
    return engine


@pytest.fixture
def mock_health_monitor():
    monitor = MagicMock(spec=HealthMonitor)
    monitor.evaluate.return_value = HealthReport(
        domain="example.com",
        health_status=HealthStatus.HEALTHY,
        total_attempts=10,
        successful_attempts=8,
        strategic_consecutive_failures=0,
        success_rate=0.8,
        reason="Within acceptable parameters",
    )
    return monitor


@pytest.fixture
def playbook_path(tmp_path: Path) -> Path:
    """Create a real temporary playbook file so BaseAgent._load_playbook() succeeds."""
    playbook = tmp_path / "scout.md"
    playbook.write_text("# Scout Playbook\nYou are the Scout agent.", encoding="utf-8")
    return playbook


@pytest.fixture
def scout_mvi(mock_stealth_engine, mock_health_monitor, playbook_path):
    return ScoutMVI(
        llm_client=MagicMock(),
        token_ledger=MagicMock(),
        playbook_path=str(playbook_path),
        agent_name="scout",
        model_name="qwen2.5:7b",
        stealth_engine=mock_stealth_engine,
        health_monitor=mock_health_monitor,
    )


def test_execute_success(scout_mvi, mock_stealth_engine):
    """Verify successful MVI pipeline execution."""
    products = [
        ProductRecord(
            product_name="Eco Pan",
            brand="GreenCo",
            price=29.99,
            currency="USD",
            url="https://example.com/pan",
            asin="B012345678",
        )
    ]
    mock_stealth_engine.search_products.return_value = RetrievalEnvelope(
        outcome=Outcome.RETRIEVED,
        domain="example.com",
        products=products,
        tier_used=1,
        attempt_count=1,
        elapsed_seconds=2.5,
        reason="Success",
        profile_timestamp=None,
    )
    
    context = json.dumps({"query": "eco pans", "domain": "example.com"})
    result = scout_mvi.execute("Find eco pans", context)
    
    assert result.artifact is not None
    parsed = json.loads(result.artifact)
    assert len(parsed) == 1
    assert parsed[0]["product_name"] == "Eco Pan"
    assert result.agent_name == "scout"
    mock_stealth_engine.search_products.assert_called_once_with("eco pans", "example.com")


def test_execute_missing_context(scout_mvi):
    """Verify AdapterError raised when context is None."""
    with pytest.raises(AdapterError) as exc_info:
        scout_mvi.execute("Find pans", None)
    
    assert "Context is required" in str(exc_info.value)


def test_execute_invalid_json_context(scout_mvi):
    """Verify AdapterError raised when context is not valid JSON."""
    with pytest.raises(AdapterError) as exc_info:
        scout_mvi.execute("Find pans", "not json")
    
    assert "Context must be valid JSON" in str(exc_info.value)


def test_execute_missing_context_fields(scout_mvi):
    """Verify AdapterError raised when context missing query or domain."""
    context = json.dumps({"query": "pans"})
    
    with pytest.raises(AdapterError) as exc_info:
        scout_mvi.execute("Find pans", context)
    
    assert "missing required fields" in str(exc_info.value)


def test_execute_unsupported_domain(scout_mvi):
    """Verify AdapterError raised when domain not in registry."""
    context = json.dumps({"query": "pans", "domain": "unsupported.com"})
    
    with pytest.raises(AdapterError) as exc_info:
        scout_mvi.execute("Find pans", context)
    
    assert "not supported" in str(exc_info.value)
    assert "unsupported.com" in str(exc_info.value)


def test_execute_blocked_outcome(scout_mvi, mock_stealth_engine):
    """Verify AdapterError raised when StealthEngine returns BLOCKED."""
    mock_stealth_engine.search_products.return_value = RetrievalEnvelope(
        outcome=Outcome.BLOCKED,
        domain="example.com",
        products=[],
        tier_used=1,
        attempt_count=1,
        elapsed_seconds=1.0,
        reason="Access denied (403/429)",
        profile_timestamp=None,
    )
    
    context = json.dumps({"query": "pans", "domain": "example.com"})
    
    with pytest.raises(AdapterError) as exc_info:
        scout_mvi.execute("Find pans", context)
    
    assert "Retrieval failed" in str(exc_info.value)
    assert "Access denied" in str(exc_info.value)


def test_execute_no_results(scout_mvi, mock_stealth_engine):
    """Verify empty artifact returned when StealthEngine returns NO_RESULTS."""
    mock_stealth_engine.search_products.return_value = RetrievalEnvelope(
        outcome=Outcome.NO_RESULTS,
        domain="example.com",
        products=[],
        tier_used=1,
        attempt_count=1,
        elapsed_seconds=2.0,
        reason="No products matched",
        profile_timestamp=None,
    )
    
    context = json.dumps({"query": "rare item", "domain": "example.com"})
    result = scout_mvi.execute("Find rare item", context)
    
    assert result.artifact == "[]"


def test_execute_disallowed_outcome(scout_mvi, mock_stealth_engine):
    """Verify empty artifact returned when StealthEngine returns DISALLOWED."""
    mock_stealth_engine.search_products.return_value = RetrievalEnvelope(
        outcome=Outcome.DISALLOWED,
        domain="example.com",
        products=[],
        tier_used=0,
        attempt_count=0,
        elapsed_seconds=0.0,
        reason="Domain prohibited",
        profile_timestamp=None,
    )
    
    context = json.dumps({"query": "pans", "domain": "example.com"})
    result = scout_mvi.execute("Find pans", context)
    
    assert result.artifact == "[]"


def test_execute_health_monitor_failure(scout_mvi, mock_stealth_engine, mock_health_monitor):
    """Verify job continues even if HealthMonitor fails."""
    mock_stealth_engine.search_products.return_value = RetrievalEnvelope(
        outcome=Outcome.RETRIEVED,
        domain="example.com",
        products=[
            ProductRecord(
                product_name="Pan",
                brand="Brand",
                price=10.0,
                currency="USD",
                url="https://example.com/pan",
                asin="B012345678",
            )
        ],
        tier_used=1,
        attempt_count=1,
        elapsed_seconds=2.0,
        reason="Success",
        profile_timestamp=None,
    )
    
    mock_health_monitor.evaluate.side_effect = Exception("DB Locked")
    
    context = json.dumps({"query": "pans", "domain": "example.com"})
    result = scout_mvi.execute("Find pans", context)
    
    assert result.artifact is not None
    assert "Pan" in result.artifact


def test_execute_empty_products_after_retrieval(scout_mvi, mock_stealth_engine):
    """Verify empty artifact returned when StealthEngine returns RETRIEVED but no products."""
    mock_stealth_engine.search_products.return_value = RetrievalEnvelope(
        outcome=Outcome.RETRIEVED,
        domain="example.com",
        products=[],
        tier_used=1,
        attempt_count=1,
        elapsed_seconds=2.0,
        reason="Success",
        profile_timestamp=None,
    )
    
    context = json.dumps({"query": "pans", "domain": "example.com"})
    result = scout_mvi.execute("Find pans", context)
    
    assert result.artifact == "[]"