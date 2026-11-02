# src/services/stealth_engine.py
"""
Jarvi Native Stealth Engine (Module 2.1).
Adaptive, tiered web retrieval engine with TLS fingerprint rotation, 
circuit breaker integration, and graceful degradation.
"""
from __future__ import annotations

import logging
import random
import time
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol

from src.utils.llm_client import CircuitBreaker

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None  # Will fail fast at startup validation if required

logger = logging.getLogger(__name__)

__all__ = [
    "Outcome",
    "ProductRecord",
    "SiteProfile",
    "RetrievalEnvelope",
    "StealthError",
    "StealthEngine",
    "ExtractionProtocol",
]

# ============================================================================
# SCHEMAS & EXCEPTIONS
# ============================================================================

class Outcome(Enum):
    RETRIEVED = "RETRIEVED"
    NO_RESULTS = "NO_RESULTS"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    DISALLOWED = "DISALLOWED"


@dataclass(frozen=True)
class ProductRecord:
    """MCP-serializable product record."""
    product_name: str
    brand: str
    price: float
    currency: str
    url: str
    asin: str


@dataclass
class SiteProfile:
    """Persisted site profile state."""
    domain: str
    preferred_tier: int = 1
    consecutive_failures: int = 0
    last_success_timestamp: Optional[float] = None
    cooldown_expiry: Optional[float] = None
    cooldown_level: int = 0  # 0=none, 1=1h, 2=6h, 3=24h
    confidence: float = 1.0


@dataclass(frozen=True)
class RetrievalEnvelope:
    """Primary output of the Stealth Engine."""
    outcome: Outcome
    domain: str
    products: List[ProductRecord]
    tier_used: int
    attempt_count: int
    elapsed_seconds: float
    reason: str
    profile_timestamp: Optional[float]


class StealthError(Exception):
    """Module-level exception for stealth engine failures."""
    def __init__(self, message: str, tier: int = 0, error_code: str = "STEALTH_FAILURE", original_error: Optional[Exception] = None):
        super().__init__(message)
        self.tier = tier
        self.error_code = error_code
        self.original_error = original_error


# ============================================================================
# PROTOCOLS
# ============================================================================

class MemoryEngineProtocol(Protocol):
    def get_site_profile(self, domain: str) -> Optional[SiteProfile]: ...
    def save_site_profile(self, profile: SiteProfile) -> None: ...
    def log_telemetry(self, domain: str, tier: int, outcome: str, duration: float) -> None: ...


class ExtractionProtocol(Protocol):
    def extract(self, html_content: str, domain: str, query: str) -> List[ProductRecord]: ...


# ============================================================================
# CONSTANTS
# ============================================================================

FINGERPRINT_POOL = ["chrome124", "chrome120", "safari17_0", "edge101"]
PROHIBITED_DOMAINS = {"localhost", "127.0.0.1", "internal.jarvi.local"}
COOLDOWN_DURATIONS = {0: 0, 1: 3600, 2: 21600, 3: 86400}
MAX_TIER1_TIMEOUT = 15.0
MAX_TIER2_TIMEOUT = 30.0


# ============================================================================
# CORE ENGINE
# ============================================================================

class StealthEngine:
    """
    Adaptive, tiered web retrieval engine.
    
    Args:
        memory_engine: Injected MemoryEngine for profile/telemetry persistence.
        circuit_breaker: Injected CircuitBreaker for network degradation handling.
        extraction_strategy: Optional injected strategy for HTML parsing.
    """

    def __init__(
        self,
        memory_engine: MemoryEngineProtocol,
        circuit_breaker: CircuitBreaker,
        extraction_strategy: Optional[ExtractionProtocol] = None,
    ) -> None:
        if memory_engine is None:
            raise StealthError("MemoryEngine is required", error_code="MISSING_DEPENDENCY")
        if circuit_breaker is None:
            raise StealthError("CircuitBreaker is required", error_code="MISSING_DEPENDENCY")
        if curl_requests is None:
            raise StealthError("curl-cffi is required for Tier 1 retrieval", error_code="MISSING_DEPENDENCY")

        self._memory_engine = memory_engine
        self._circuit_breaker = circuit_breaker
        self._extraction_strategy = extraction_strategy
        self._browser_runtime_available = False  # Feature flag for Tier 2

    def search_products(self, query: str, domain: str) -> RetrievalEnvelope:
        """
        Execute adaptive retrieval for a given query and domain.
        
        Args:
            query: Product search query (NOT logged).
            domain: Canonical target domain.
            
        Returns:
            RetrievalEnvelope containing outcome and extracted products.
        """
        start_time = time.time()
        attempt_count = 0
        
        # 1. Policy Gate
        if domain in PROHIBITED_DOMAINS:
            return self._build_envelope(Outcome.DISALLOWED, domain, [], 0, attempt_count, start_time, "Domain prohibited")
        
        if not self._circuit_breaker.allow_request():
            return self._build_envelope(Outcome.UNAVAILABLE, domain, [], 0, attempt_count, start_time, "Circuit breaker OPEN")

        # 2. Profile Lookup
        profile = self._get_profile_safe(domain)
        now = time.time()
        if profile.cooldown_expiry and now < profile.cooldown_expiry:
            return self._build_envelope(Outcome.DISALLOWED, domain, [], profile.preferred_tier, attempt_count, start_time, "Active cooldown")

        # 3. Tier 1 Attempt
        attempt_count += 1
        envelope = self._attempt_tier1(query, domain, profile, start_time)
        
        if envelope.outcome in (Outcome.RETRIEVED, Outcome.NO_RESULTS):
            self._update_profile(profile, success=True)
            self._log_telemetry_safe(domain, 1, envelope.outcome.value, time.time() - start_time)
            return envelope

        # 4. Tier 2 Escalation (if available and warranted)
        if self._browser_runtime_available and envelope.outcome == Outcome.BLOCKED:
            attempt_count += 1
            envelope_t2 = self._attempt_tier2(query, domain, profile, start_time)
            if envelope_t2.outcome in (Outcome.RETRIEVED, Outcome.NO_RESULTS):
                self._update_profile(profile, success=True)
                self._log_telemetry_safe(domain, 2, envelope_t2.outcome.value, time.time() - start_time)
                return envelope_t2

        # 5. Graceful Degradation
        is_blocked = (envelope.outcome == Outcome.BLOCKED)
        self._update_profile(profile, success=False, is_blocked=is_blocked)
        self._log_telemetry_safe(domain, profile.preferred_tier, envelope.outcome.value, time.time() - start_time)
        
        # CRITICAL FIX: Preserve BLOCKED outcome so the caller knows to respect the cooldown.
        # Do not overwrite it with UNAVAILABLE.
        final_outcome = Outcome.BLOCKED if is_blocked else Outcome.UNAVAILABLE
        final_reason = envelope.reason or "All tiers failed"
        
        return self._build_envelope(final_outcome, domain, [], profile.preferred_tier, attempt_count, start_time, final_reason)

    def _attempt_tier1(self, query: str, domain: str, profile: SiteProfile, start_time: float) -> RetrievalEnvelope:
        """Execute Tier 1 fingerprinted HTTP retrieval."""
        attempt_count = 1
        fingerprint = random.choice(FINGERPRINT_POOL)
        
        try:
            # Scoped session for this specific retrieval operation
            with curl_requests.Session(impersonate=fingerprint, timeout=MAX_TIER1_TIMEOUT) as session:
                # Mimic initial handshake for cookie establishment
                session.get(f"https://{domain}", timeout=5.0)
                
                # Execute actual search query
                search_url = f"https://{domain}/search?q={urllib.parse.quote(query)}"
                response = session.get(search_url, timeout=MAX_TIER1_TIMEOUT)
                
                if response.status_code in (403, 429):
                    self._circuit_breaker.record_failure()
                    return self._build_envelope(Outcome.BLOCKED, domain, [], 1, attempt_count, start_time, "Access denied (403/429)")
                
                response.raise_for_status()
                self._circuit_breaker.record_success()
                
                return self._extract_and_envelope(response.text, domain, query, 1, attempt_count, start_time)
                
        except Exception as e:
            self._circuit_breaker.record_failure()
            reason = "Tier 1 timeout" if "timeout" in str(e).lower() else "Tier 1 transport failure"
            return self._build_envelope(Outcome.UNAVAILABLE, domain, [], 1, attempt_count, start_time, reason)

    def _attempt_tier2(self, query: str, domain: str, profile: SiteProfile, start_time: float) -> RetrievalEnvelope:
        """Execute Tier 2 ephemeral JS rendering (stubbed for absent-safe behavior)."""
        attempt_count = 2
        try:
            # Placeholder for crawl4ai integration
            logger.warning("Tier 2 runtime not implemented/available. Skipping.")
            return self._build_envelope(Outcome.UNAVAILABLE, domain, [], 2, attempt_count, start_time, "Tier 2 runtime absent")
        except Exception as e:
            return self._build_envelope(Outcome.UNAVAILABLE, domain, [], 2, attempt_count, start_time, f"Tier 2 runtime failure: {e}")

    def _extract_and_envelope(self, html_content: str, domain: str, query: str, tier: int, attempt_count: int, start_time: float) -> RetrievalEnvelope:
        """Run extraction strategy and build envelope."""
        try:
            if self._extraction_strategy:
                products = self._extraction_strategy.extract(html_content, domain, query)
            else:
                # Fallback: basic heuristic or defer to agent
                products = []
            
            if not products:
                return self._build_envelope(Outcome.NO_RESULTS, domain, [], tier, attempt_count, start_time, "No products matched")
            
            return self._build_envelope(Outcome.RETRIEVED, domain, products, tier, attempt_count, start_time, "Success")
        except Exception as e:
            return self._build_envelope(Outcome.NO_RESULTS, domain, [], tier, attempt_count, start_time, f"Extraction failure: {e}")

    def _get_profile_safe(self, domain: str) -> SiteProfile:
        """Retrieve profile with graceful degradation."""
        try:
            profile = self._memory_engine.get_site_profile(domain)
            if profile:
                # Check expiry (7 days = 604800 seconds)
                if profile.last_success_timestamp and (time.time() - profile.last_success_timestamp) > 604800:
                    logger.warning("Site profile expired for %s, resetting to Tier 1", domain)
                    profile.preferred_tier = 1
                    profile.consecutive_failures = 0
                    profile.cooldown_level = 0
                    profile.cooldown_expiry = None
                return profile
        except Exception as e:
            logger.warning("MemoryEngine profile lookup failed for %s: %s. Using defaults.", domain, type(e).__name__)
        
        return SiteProfile(domain=domain)

    def _update_profile(self, profile: SiteProfile, success: bool, is_blocked: bool = False) -> None:
        """Update profile state based on outcome."""
        now = time.time()
        if success:
            profile.last_success_timestamp = now
            profile.consecutive_failures = 0
            profile.cooldown_level = 0
            profile.cooldown_expiry = None
            profile.confidence = min(1.0, profile.confidence + 0.1)
        else:
            profile.consecutive_failures += 1
            profile.confidence = max(0.0, profile.confidence - 0.1)
            
            if is_blocked or profile.consecutive_failures >= 3:
                # Escalating cooldown: 1h -> 6h -> 24h
                profile.cooldown_level = min(3, profile.cooldown_level + 1)
                profile.cooldown_expiry = now + COOLDOWN_DURATIONS[profile.cooldown_level]
                logger.warning("Domain %s entering cooldown level %d (%d hours)", profile.domain, profile.cooldown_level, COOLDOWN_DURATIONS[profile.cooldown_level] // 3600)

        try:
            self._memory_engine.save_site_profile(profile)
        except Exception as e:
            logger.warning("Failed to save site profile for %s: %s", profile.domain, type(e).__name__)

    def _log_telemetry_safe(self, domain: str, tier: int, outcome: str, duration: float) -> None:
        """Append telemetry with graceful degradation."""
        try:
            self._memory_engine.log_telemetry(domain, tier, outcome, duration)
        except Exception as e:
            logger.warning("Telemetry write failed for %s: %s", domain, type(e).__name__)

    def _build_envelope(
        self, outcome: Outcome, domain: str, products: List[ProductRecord], 
        tier: int, attempt_count: int, start_time: float, reason: str
    ) -> RetrievalEnvelope:
        """Construct the final retrieval envelope."""
        profile = self._get_profile_safe(domain)
        return RetrievalEnvelope(
            outcome=outcome,
            domain=domain,
            products=products,
            tier_used=tier,
            attempt_count=attempt_count,
            elapsed_seconds=time.time() - start_time,
            reason=reason,
            profile_timestamp=profile.last_success_timestamp,
        )