# src/agents/scout_mvi.py
"""
ScoutMVI Agent: Orchestrates the MVI (Minimum Viable Income) product research pipeline.
Integrates Stealth Engine, Extractor, and Health Monitor for end-to-end product retrieval.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Callable

from src.agents.base_agent import AgentResult, BaseAgent
from src.services.health_monitor import HealthMonitor
from src.services.stealth_engine import Outcome, StealthEngine
from src.services.step_adapters import AdapterError, guard_scout_artifact
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger

logger = logging.getLogger(__name__)

__all__ = ["ScoutMVI"]


class ScoutMVI(BaseAgent):
    """
    Scout agent specialized for MVI product research.
    
    Orchestrates: Stealth Engine → Extractor → Guard → Health Monitor.
    Receives injected dependencies for testability and zero global state.
    
    Args:
        llm_client: LLM client callable.
        token_ledger: Token usage tracking.
        playbook_path: Path to agent playbook.
        agent_name: Name of the agent.
        model_name: LLM model name.
        stealth_engine: Injected StealthEngine instance.
        health_monitor: Injected HealthMonitor instance.
    """

    def __init__(
        self,
        llm_client: Callable[..., OllamaResponse],
        token_ledger: TokenLedger,
        playbook_path: str,
        agent_name: str,
        model_name: str,
        stealth_engine: StealthEngine,
        health_monitor: HealthMonitor,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            token_ledger=token_ledger,
            playbook_path=playbook_path,
            agent_name=agent_name,
            model_name=model_name,
        )
        self.stealth_engine = stealth_engine
        self.health_monitor = health_monitor

    def execute(self, task_prompt: str, context: str | None = None) -> AgentResult:
        """
        Execute the MVI product research pipeline.
        
        Args:
            task_prompt: Natural language task description.
            context: JSON string containing 'query' and 'domain' parameters.
            
        Returns:
            AgentResult with artifact containing JSON-serialized ProductRecord list.
            
        Raises:
            AdapterError: If domain not supported, retrieval fails, or guard validation fails.
        """
        # 1. Parse context to extract query and domain
        if not context:
            raise AdapterError("Context is required and must contain 'query' and 'domain'")
        
        try:
            context_dict = json.loads(context)
        except json.JSONDecodeError as e:
            raise AdapterError(f"Context must be valid JSON: {e}") from e
        
        query = context_dict.get("query")
        domain = context_dict.get("domain")
        
        if not query or not domain:
            raise AdapterError(
                f"Context missing required fields. Expected 'query' and 'domain', got: {list(context_dict.keys())}"
            )
        
        # 2. Validate domain in registry
        extractor = getattr(self.stealth_engine, "_extraction_strategy", None)
        if extractor is None:
            raise AdapterError("StealthEngine not configured with extraction strategy")
        
        registry = getattr(extractor, "_registry", None)
        if registry is None or not registry.get_rule(domain):
            supported = list(registry._RULES.keys()) if registry else []
            raise AdapterError(
                f"Domain '{domain}' not supported. Supported domains: {supported}"
            )
        
        logger.info("ScoutMVI executing: query='%s', domain='%s'", query, domain)
        
        # 3. Call StealthEngine (internally uses Extractor)
        try:
            envelope = self.stealth_engine.search_products(query, domain)
        except Exception as e:
            raise AdapterError(f"StealthEngine failed: {e}") from e
        
        # 4. Handle outcomes
        if envelope.outcome == Outcome.DISALLOWED:
            logger.warning("Domain %s disallowed: %s", domain, envelope.reason)
            return self._build_result([])
        
        if envelope.outcome == Outcome.NO_RESULTS:
            logger.info("No products found for query '%s' on %s", query, domain)
            return self._build_result([])
        
        if envelope.outcome in (Outcome.BLOCKED, Outcome.UNAVAILABLE):
            raise AdapterError(
                f"Retrieval failed for {domain}: {envelope.reason} "
                f"(tier={envelope.tier_used}, attempts={envelope.attempt_count})"
            )
        
        # 5. Validate products via guard
        products = envelope.products
        if not products:
            logger.info("StealthEngine returned RETRIEVED but no products extracted for %s", domain)
            return self._build_result([])
        
        # Serialize products to JSON for guard validation
        products_json = json.dumps([asdict(p) for p in products], indent=2)
        
        try:
            guard_scout_artifact(products_json)
        except AdapterError as e:
            raise AdapterError(f"Guard validation failed: {e}") from e
        
        # 6. Update domain health (graceful failure)
        try:
            health_report = self.health_monitor.evaluate(domain)
            logger.info(
                "Domain %s health: %s (success_rate=%.1f%%, consecutive_failures=%d)",
                domain,
                health_report.health_status.value,
                health_report.success_rate * 100,
                health_report.strategic_consecutive_failures,
            )
        except Exception as e:
            logger.warning("HealthMonitor evaluation failed for %s: %s", domain, type(e).__name__)
        
        # 7. Return artifact
        logger.info("ScoutMVI completed: %d products extracted from %s", len(products), domain)
        return self._build_result(products)

    def _build_result(self, products: list) -> AgentResult:
        """Build AgentResult with JSON-serialized product list."""
        products_json = json.dumps([asdict(p) for p in products], indent=2)
        return AgentResult(
            agent_name=self.agent_name,
            artifact=products_json,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )