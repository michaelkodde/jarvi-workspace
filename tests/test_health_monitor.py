# tests/test_health_monitor.py
"""
Unit tests for the Scraper Health Monitor.
Validates health status transitions, telemetry analysis, and graceful degradation.
"""
import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

from src.services.health_monitor import (
    HealthMonitor,
    HealthMonitorError,
    HealthStatus,
    HealthReport,
)


@dataclass
class MockProfile:
    domain: str
    health_status: Optional[str] = None
    consecutive_failures: int = 0  # Should NOT be modified by HealthMonitor
    cooldown_expiry: Optional[float] = None  # Should NOT be modified by HealthMonitor


class MockMemoryEngine:
    def __init__(self, telemetry_data=None, profile=None):
        self.telemetry_data = telemetry_data or []
        self.profile = profile
        self.profile_saved = False

    def get_site_profile(self, domain: str):
        return self.profile

    def save_site_profile(self, profile):
        self.profile = profile
        self.profile_saved = True

    def get_recent_telemetry(self, domain: str, days: int, limit: int):
        return self.telemetry_data[:limit]


def test_evaluate_healthy_domain():
    """Verify that a domain with good success rate is marked HEALTHY."""
    telemetry = [
        {"outcome": "RETRIEVED"},
        {"outcome": "RETRIEVED"},
        {"outcome": "NO_RESULTS"},  # Not a strategic failure
        {"outcome": "RETRIEVED"},
    ]
    profile = MockProfile(domain="example.com", health_status="HEALTHY")
    memory_engine = MockMemoryEngine(telemetry_data=telemetry, profile=profile)
    monitor = HealthMonitor(memory_engine=memory_engine)
    
    report = monitor.evaluate("example.com")
    
    assert report.health_status == HealthStatus.HEALTHY
    assert report.total_attempts == 4
    assert report.successful_attempts == 3
    assert report.strategic_consecutive_failures == 0
    assert memory_engine.profile_saved is True
    assert memory_engine.profile.consecutive_failures == 0  # Verify tactical state untouched


def test_evaluate_degraded_domain():
    """Verify that a domain with <50% success rate over min attempts is DEGRADED."""
    telemetry = [
        {"outcome": "UNAVAILABLE"},
        {"outcome": "UNAVAILABLE"},
        {"outcome": "UNAVAILABLE"},
        {"outcome": "RETRIEVED"},
        {"outcome": "UNAVAILABLE"},
    ]
    profile = MockProfile(domain="example.com", health_status="HEALTHY")
    memory_engine = MockMemoryEngine(telemetry_data=telemetry, profile=profile)
    monitor = HealthMonitor(memory_engine=memory_engine)
    
    report = monitor.evaluate("example.com")
    
    assert report.health_status == HealthStatus.DEGRADED
    assert report.success_rate == 0.2  # 1 out of 5
    assert report.strategic_consecutive_failures == 1  # Last one was UNAVAILABLE
    assert memory_engine.profile_saved is True
    assert memory_engine.profile.health_status == "DEGRADED"


def test_evaluate_blocked_domain_triggers_notification():
    """Verify that >10 consecutive strategic failures marks domain BLOCKED and triggers notification."""
    telemetry = [{"outcome": "BLOCKED"} for _ in range(12)]
    profile = MockProfile(domain="example.com", health_status="DEGRADED")
    memory_engine = MockMemoryEngine(telemetry_data=telemetry, profile=profile)
    
    notification_mock = MagicMock()
    monitor = HealthMonitor(memory_engine=memory_engine, notification_callback=notification_mock)
    
    report = monitor.evaluate("example.com")
    
    assert report.health_status == HealthStatus.BLOCKED
    assert report.strategic_consecutive_failures == 12
    assert memory_engine.profile_saved is True
    assert memory_engine.profile.health_status == "BLOCKED"
    assert memory_engine.profile.consecutive_failures == 0  # Verify tactical state untouched
    
    notification_mock.assert_called_once_with("example.com", "Domain blocked: DOM may have changed, update parsing rules.")


def test_no_results_does_not_trigger_blocking():
    """Verify that NO_RESULTS is not counted as a strategic failure."""
    telemetry = [{"outcome": "NO_RESULTS"} for _ in range(15)]
    profile = MockProfile(domain="example.com", health_status="HEALTHY")
    memory_engine = MockMemoryEngine(telemetry_data=telemetry, profile=profile)
    monitor = HealthMonitor(memory_engine=memory_engine)
    
    report = monitor.evaluate("example.com")
    
    # Should be HEALTHY (success rate is technically 0, but NO_RESULTS doesn't count as failure for blocking)
    # Actually, success_rate = 0/15 = 0.0, which is < 0.5, so it will be DEGRADED, but NOT BLOCKED.
    assert report.health_status == HealthStatus.DEGRADED
    assert report.strategic_consecutive_failures == 0


def test_notification_only_on_transition():
    """Verify that notification is only fired when status changes to BLOCKED."""
    telemetry_blocked = [{"outcome": "BLOCKED"} for _ in range(12)]
    
    # First evaluation: transition to BLOCKED
    profile = MockProfile(domain="example.com", health_status="HEALTHY")
    memory_engine = MockMemoryEngine(telemetry_data=telemetry_blocked, profile=profile)
    notification_mock = MagicMock()
    monitor = HealthMonitor(memory_engine=memory_engine, notification_callback=notification_mock)
    
    monitor.evaluate("example.com")
    assert notification_mock.call_count == 1  # First transition
    
    # Second evaluation: remains BLOCKED
    notification_mock.reset_mock()
    profile.health_status = "BLOCKED"  # Previous status is now BLOCKED
    
    monitor.evaluate("example.com")
    assert notification_mock.call_count == 0  # No transition, no notification


def test_evaluate_graceful_degradation_on_memory_failure():
    """Verify that MemoryEngine failures do not crash the monitor."""
    broken_engine = MagicMock()
    broken_engine.get_recent_telemetry.side_effect = Exception("DB Locked")
    
    monitor = HealthMonitor(memory_engine=broken_engine)
    
    report = monitor.evaluate("example.com")
    
    assert report.health_status == HealthStatus.HEALTHY
    assert report.reason == "Telemetry fetch failed, defaulting to healthy"
    assert report.total_attempts == 0


def test_missing_dependency_raises_error():
    """Verify that missing MemoryEngine raises HealthMonitorError at startup."""
    with pytest.raises(HealthMonitorError) as exc_info:
        HealthMonitor(memory_engine=None)
    
    assert exc_info.value.error_code == "MISSING_DEPENDENCY"