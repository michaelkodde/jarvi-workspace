# src/services/health_monitor.py
"""
Scraper Health Monitor (Module 7).
Evaluates domain retrieval health over a rolling window and updates site profiles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

__all__ = [
    "HealthStatus",
    "HealthReport",
    "HealthMonitorError",
    "HealthMonitor",
    "MemoryEngineProtocol",
]

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CONSTANTS (Zero Hardcoding)
# ============================================================================
MIN_ATTEMPTS_FOR_DEGRADED = 5
DEGRADED_SUCCESS_RATE_THRESHOLD = 0.50
BLOCKED_CONSECUTIVE_FAILURES = 10
TELEMETRY_ROLLING_WINDOW_DAYS = 7
TELEMETRY_QUERY_LIMIT = 100


# ============================================================================
# SCHEMAS & EXCEPTIONS
# ============================================================================

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class HealthReport:
    """MCP-serializable report of a domain's retrieval health."""
    domain: str
    health_status: HealthStatus
    total_attempts: int
    successful_attempts: int
    strategic_consecutive_failures: int
    success_rate: float
    reason: str


class HealthMonitorError(Exception):
    """Module-level exception for health monitor failures."""
    def __init__(self, message: str, error_code: str = "HEALTH_MONITOR_FAILURE", original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error


# ============================================================================
# PROTOCOLS
# ============================================================================

class MemoryEngineProtocol(Protocol):
    def get_site_profile(self, domain: str) -> Optional[Any]: ...
    def save_site_profile(self, profile: Any) -> None: ...
    def get_recent_telemetry(self, domain: str, days: int, limit: int) -> List[Dict[str, Any]]: ...


# ============================================================================
# CORE MONITOR
# ============================================================================

class HealthMonitor:
    """
    Evaluates domain retrieval health and updates site profiles.
    
    Args:
        memory_engine: Injected MemoryEngine for telemetry and profile persistence.
        notification_callback: Optional callable(domain: str, message: str) for alerts.
    """

    def __init__(
        self,
        memory_engine: MemoryEngineProtocol,
        notification_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        if memory_engine is None:
            raise HealthMonitorError("MemoryEngine is required", error_code="MISSING_DEPENDENCY")
        
        self._memory_engine = memory_engine
        self._notification_callback = notification_callback

    def evaluate(self, domain: str) -> HealthReport:
        """
        Evaluate the health of a domain based on recent telemetry.
        
        Args:
            domain: The canonical target domain to evaluate.
            
        Returns:
            HealthReport detailing the domain's current health status.
        """
        try:
            telemetry = self._memory_engine.get_recent_telemetry(
                domain=domain, 
                days=TELEMETRY_ROLLING_WINDOW_DAYS, 
                limit=TELEMETRY_QUERY_LIMIT
            )
        except Exception as e:
            logger.warning("MemoryEngine telemetry fetch failed for %s: %s. Assuming healthy.", domain, type(e).__name__)
            return self._build_report(domain, [], HealthStatus.HEALTHY, "Telemetry fetch failed, defaulting to healthy")

        total_attempts = len(telemetry)
        if total_attempts == 0:
            return self._build_report(domain, telemetry, HealthStatus.HEALTHY, "No recent attempts")

        successful_attempts = sum(1 for t in telemetry if t.get("outcome") == "RETRIEVED")
        success_rate = successful_attempts / total_attempts
        
        # Count consecutive strategic failures from the most recent attempt backwards.
        # NO_RESULTS is explicitly ignored (it is a valid retrieval outcome).
        strategic_consecutive_failures = 0
        for t in reversed(telemetry):
            outcome = t.get("outcome")
            if outcome in ("BLOCKED", "UNAVAILABLE"):
                strategic_consecutive_failures += 1
            elif outcome == "RETRIEVED":
                break
            # NO_RESULTS is ignored per spec

        # Determine Health Status
        if strategic_consecutive_failures >= BLOCKED_CONSECUTIVE_FAILURES:
            status = HealthStatus.BLOCKED
            reason = f"{strategic_consecutive_failures} consecutive strategic failures"
        elif total_attempts >= MIN_ATTEMPTS_FOR_DEGRADED and success_rate < DEGRADED_SUCCESS_RATE_THRESHOLD:
            status = HealthStatus.DEGRADED
            reason = f"Success rate {success_rate:.1%} over {total_attempts} attempts"
        else:
            status = HealthStatus.HEALTHY
            reason = "Within acceptable parameters"

        # Apply side effects (Update Profile - Strategic State Only)
        previous_status_str = self._update_profile_safe(domain, status)

        # Trigger notification ONLY on state transition to BLOCKED (prevents spam)
        if status == HealthStatus.BLOCKED and previous_status_str != HealthStatus.BLOCKED.value:
            self._notify_safe(domain, "Domain blocked: DOM may have changed, update parsing rules.")

        return self._build_report(
            domain, telemetry, status, reason, success_rate, strategic_consecutive_failures
        )

    def _update_profile_safe(self, domain: str, new_status: HealthStatus) -> Optional[str]:
        """
        Update the SiteProfile health_status field only, degrading gracefully on failure.
        Returns the previous health status string for transition detection.
        """
        try:
            profile = self._memory_engine.get_site_profile(domain)
            if profile is None:
                logger.warning("Site profile not found for %s. Skipping health status update.", domain)
                return None
            
            # Read previous status for transition detection
            previous_status_str = getattr(profile, "health_status", None)
            
            # CRITICAL: Update ONLY the health_status field to prevent race conditions 
            # with Stealth Engine's tactical state (consecutive_failures, cooldown_expiry).
            profile.health_status = new_status.value
            
            self._memory_engine.save_site_profile(profile)
            return previous_status_str
            
        except Exception as e:
            logger.warning("Failed to update site profile health for %s: %s", domain, type(e).__name__)
            return None

    def _notify_safe(self, domain: str, message: str) -> None:
        """Trigger notification callback, degrading gracefully on failure."""
        if self._notification_callback:
            try:
                self._notification_callback(domain, message)
            except Exception as e:
                logger.error("Notification callback failed for %s: %s", domain, type(e).__name__)

    def _build_report(
        self, 
        domain: str, 
        telemetry: List[Dict[str, Any]], 
        status: HealthStatus, 
        reason: str,
        success_rate: float = 1.0,
        strategic_consecutive_failures: int = 0
    ) -> HealthReport:
        """Construct the final health report."""
        total = len(telemetry)
        successful = sum(1 for t in telemetry if t.get("outcome") == "RETRIEVED")
        
        return HealthReport(
            domain=domain,
            health_status=status,
            total_attempts=total,
            successful_attempts=successful,
            strategic_consecutive_failures=strategic_consecutive_failures,
            success_rate=success_rate,
            reason=reason
        )