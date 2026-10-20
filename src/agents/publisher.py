"""
Publisher Agent: Last-mile delivery router for content distribution.

Validates Creator's ContentEnvelope, inserts affiliate links, routes content
to files/agents/orchestrator, logs to publication ledger, and returns a report.
Does NOT call LLM - routing is 100% deterministic Python.
"""
import dataclasses
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.agents.base_agent import AgentResult, BaseAgent
from src.utils.llm_client import OllamaResponse
from src.utils.publication_ledger import LedgerError, PublicationLedger, PublicationRecord
from src.utils.token_ledger import TokenLedger

logger = logging.getLogger(__name__)

__all__ = ["Publisher", "PublicationError"]


class PublicationError(Exception):
    """Raised when structural validation of the content envelope fails."""
    pass


@dataclass(frozen=True)
class DeliveryResult:
    """Result of delivering a single variant."""
    variant_index: int
    destination: str
    target: str
    status: str
    message: str
    content_hash: str
    affiliate_links_inserted: int
    formatted_content: Optional[str] = None


@dataclass(frozen=True)
class PublicationReport:
    """Report of all delivery attempts for a publication."""
    publication_id: str
    timestamp: str
    status: str
    deliveries: Tuple[DeliveryResult, ...]
    total_affiliate_links_inserted: int
    next_actions: Tuple[str, ...]


ALLOWED_DESTINATIONS = frozenset(["file", "agent", "publisher", "orchestrator"])


class Publisher(BaseAgent):
    """
    Last-mile delivery router for content distribution.
    
    Inherits BaseAgent for architectural consistency but does NOT call _call_llm.
    Routing is 100% deterministic Python. The playbook is documentation only.
    
    Args:
        llm_client: Callable returning OllamaResponse (not used by Publisher).
        token_ledger: TokenLedger for usage tracking (not used by Publisher).
        model_name: Ollama model identifier (not used by Publisher).
        playbook_path: Path to publisher playbook markdown (documentation only).
        publication_ledger_path: Path to publication ledger SQLite database.
    
    Raises:
        LedgerError: If ledger initialization fails.
    """

    def __init__(
        self,
        llm_client: Callable[..., OllamaResponse],
        token_ledger: TokenLedger,
        model_name: str,
        playbook_path: str = "src/agents/playbooks/publisher.md",
        publication_ledger_path: str = "dashboard/publications.db",
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            token_ledger=token_ledger,
            playbook_path=playbook_path,
            agent_name="Publisher",
            model_name=model_name,
        )
        self._ledger = PublicationLedger(db_path=publication_ledger_path)

    def execute(self, task_prompt: str, context: Optional[str] = None) -> AgentResult:
        """
        Execute content publication routing.
        
        Args:
            task_prompt: JSON string of Creator's ContentEnvelope.
            context: Optional JSON string with affiliate_link_config, product_urls, publish_immediately.
        
        Returns:
            AgentResult with artifact = JSON-serialized PublicationReport.
        
        Raises:
            PublicationError: If envelope validation fails.
        """
        publication_id = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Parse and validate envelope
        try:
            envelope = json.loads(task_prompt)
        except json.JSONDecodeError as e:
            raise PublicationError(f"task_prompt is not valid JSON: {e.__class__.__name__}") from e

        self._validate_envelope(envelope)

        # Parse context
        affiliate_config: dict = {}
        product_urls: List[str] = []
        if context:
            try:
                context_dict = json.loads(context)
                if isinstance(context_dict, dict):
                    affiliate_config = context_dict.get("affiliate_link_config", {})
                    product_urls = context_dict.get("product_urls", [])
                    if not isinstance(affiliate_config, dict):
                        logger.warning("affiliate_link_config is not a dict; skipping affiliate insertion.")
                        affiliate_config = {}
                    if not isinstance(product_urls, list) or not product_urls:
                        logger.warning("product_urls is missing or empty; skipping affiliate insertion.")
                        product_urls = []
            except json.JSONDecodeError as e:
                logger.warning("context is not valid JSON: %s; skipping affiliate insertion.", e.__class__.__name__)
                affiliate_config = {}
                product_urls = []

        variants = envelope.get("variants", [])
        deliveries: List[DeliveryResult] = []

        for variant_idx, variant in enumerate(variants):
            output_metadata = variant.get("output_metadata", {})
            routing = output_metadata.get("routing", {})
            destination = routing.get("destination", "orchestrator")
            content_hash = output_metadata.get("content_hash", "")
            validation_passed = routing.get("validation_passed", True)

            # Check validation_passed
            if not validation_passed:
                delivery = DeliveryResult(
                    variant_index=variant_idx,
                    destination=destination,
                    target=routing.get("file_path") or routing.get("next_agent") or "orchestrator",
                    status="skipped",
                    message="Creator validation failed",
                    content_hash=content_hash,
                    affiliate_links_inserted=0,
                )
                deliveries.append(delivery)
                self._log_to_ledger(publication_id, timestamp, delivery)
                continue

            # Check duplicate
            if self._check_duplicate(content_hash):
                delivery = DeliveryResult(
                    variant_index=variant_idx,
                    destination=destination,
                    target=routing.get("file_path") or routing.get("next_agent") or "orchestrator",
                    status="skipped",
                    message="Duplicate publication",
                    content_hash=content_hash,
                    affiliate_links_inserted=0,
                )
                deliveries.append(delivery)
                self._log_to_ledger(publication_id, timestamp, delivery)
                continue

            # Insert affiliate links
            variant_content = json.dumps(variant, ensure_ascii=False)
            modified_content, links_inserted = self._insert_affiliate_links(
                variant_content, product_urls, affiliate_config
            )

            # Route based on destination
            target = routing.get("file_path") or routing.get("next_agent") or "orchestrator"
            status = "success"
            message = ""
            formatted_content: Optional[str] = None

            if destination == "file":
                file_path = routing.get("file_path", "")
                if not file_path:
                    status = "failed"
                    message = "File path not specified"
                else:
                    try:
                        self._write_file(file_path, modified_content)
                        status = "success"
                        message = "File written successfully"
                    except OSError as e:
                        logger.error("File write failed for %s: %s", file_path, e)
                        status = "failed"
                        message = "File write failed"
                formatted_content = None

            elif destination == "agent":
                downstream_contract = routing.get("downstream_contract", {})
                try:
                    formatted_content = self._format_for_agent(variant, downstream_contract)
                    status = "routed"
                    message = "Formatted for agent routing"
                except (TypeError, ValueError) as e:
                    logger.error("Failed to format content for agent: %s", e)
                    status = "failed"
                    message = "Failed to format content for agent"
                    formatted_content = None

            elif destination == "orchestrator":
                formatted_content = modified_content
                status = "routed"
                message = "Returned to orchestrator"

            elif destination == "publisher":
                status = "deferred"
                message = "Phase 2+ integration required"
                formatted_content = None

            delivery = DeliveryResult(
                variant_index=variant_idx,
                destination=destination,
                target=target,
                status=status,
                message=message,
                content_hash=content_hash,
                affiliate_links_inserted=links_inserted,
                formatted_content=formatted_content,
            )
            deliveries.append(delivery)
            self._log_to_ledger(publication_id, timestamp, delivery)

        # Build report
        report_status = self._determine_report_status(deliveries)
        next_actions = self._determine_next_actions(deliveries)
        total_affiliate = sum(d.affiliate_links_inserted for d in deliveries)

        report = PublicationReport(
            publication_id=publication_id,
            timestamp=timestamp,
            status=report_status,
            deliveries=tuple(deliveries),
            total_affiliate_links_inserted=total_affiliate,
            next_actions=tuple(next_actions),
        )

        return AgentResult(
            agent_name="Publisher",
            artifact=json.dumps(dataclasses.asdict(report)),
            token_usage={"total_tokens": 0},
        )

    def _validate_envelope(self, envelope: Any) -> None:
        """
        Validate the content envelope structure.
        
        Args:
            envelope: Parsed envelope dict.
        
        Raises:
            PublicationError: If validation fails.
        """
        if not isinstance(envelope, dict):
            raise PublicationError("envelope is not a dict")

        variants = envelope.get("variants")
        if not isinstance(variants, list) or len(variants) == 0:
            raise PublicationError("variants is missing or empty")

        for idx, variant in enumerate(variants):
            if not isinstance(variant, dict):
                raise PublicationError(f"variant[{idx}] is not a dict")

            output_metadata = variant.get("output_metadata")
            if not isinstance(output_metadata, dict):
                raise PublicationError(f"variant[{idx}].output_metadata is missing")

            routing = output_metadata.get("routing")
            if not isinstance(routing, dict):
                raise PublicationError(f"variant[{idx}].output_metadata.routing is missing")

            destination = routing.get("destination")
            if destination not in ALLOWED_DESTINATIONS:
                raise PublicationError(
                    f"variant[{idx}].output_metadata.routing.destination is invalid: {destination}"
                )

    def _check_duplicate(self, content_hash: str) -> bool:
        """
        Check if content hash has been published before.
        
        Args:
            content_hash: The content hash to check.
        
        Returns:
            True if duplicate, False otherwise.
        """
        try:
            return self._ledger.check_duplicate(content_hash)
        except LedgerError as e:
            logger.warning("Ledger duplicate check failed; proceeding as not duplicate: %s", e)
            return False

    def _insert_affiliate_links(
        self, content: str, product_urls: List[str], config: dict
    ) -> Tuple[str, int]:
        """
        Insert affiliate tracking parameters into product URLs.
        
        Args:
            content: The content string to modify.
            product_urls: List of verified product URLs.
            config: Dict of query parameters to append.
        
        Returns:
            Tuple of (modified content, count of insertions).
        """
        if not product_urls or not config:
            return content, 0

        count = 0
        for url in product_urls:
            if url not in content:
                continue

            try:
                parsed = urlparse(url)
                existing_params = parse_qsl(parsed.query, keep_blank_values=True)
                new_params = existing_params + list(config.items())
                new_query = urlencode(new_params)
                new_url = urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    new_query,
                    parsed.fragment,
                ))
                content = content.replace(url, new_url)
                count += 1
            except Exception as e:
                logger.warning("Failed to insert affiliate link for %s: %s", url, e)

        return content, count

    def _write_file(self, file_path: str, content: str) -> None:
        """
        Write content to file atomically.
        
        Args:
            file_path: Path to write to.
            content: Content to write.
        
        Raises:
            OSError: If write fails.
        """
        path = os.path.abspath(file_path)
        if os.path.exists(path):
            logger.warning("File already exists, overwriting: %s", path)

        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(temp_path, path)
        except OSError:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    def _format_for_agent(self, variant: dict, downstream_contract: dict) -> str:
        """
        Format content for downstream agent consumption.
        
        Args:
            variant: The variant dict (currently unused, reserved for future use).
            downstream_contract: The downstream contract parameters.
        
        Returns:
            JSON string of formatted content.
        
        Raises:
            TypeError: If downstream_contract is not JSON-serializable.
            ValueError: If downstream_contract is invalid.
        """
        return json.dumps(downstream_contract, ensure_ascii=False)

    def _log_to_ledger(self, publication_id: str, timestamp: str, delivery: DeliveryResult) -> None:
        """
        Log a delivery attempt to the publication ledger.
        
        Args:
            publication_id: The publication ID.
            timestamp: The timestamp.
            delivery: The delivery result.
        """
        record = PublicationRecord(
            publication_id=f"{publication_id}_{delivery.variant_index}",
            timestamp=timestamp,
            content_hash=delivery.content_hash,
            destination=delivery.destination,
            target=delivery.target,
            status=delivery.status,
            affiliate_links_inserted=delivery.affiliate_links_inserted,
            error_message=delivery.message if delivery.status in ("failed", "skipped") else None,
        )
        try:
            self._ledger.log_publication(record)
        except LedgerError as e:
            logger.error("Failed to log publication to ledger: %s", e)

    def _determine_report_status(self, deliveries: List[DeliveryResult]) -> str:
        """
        Determine the overall report status based on delivery statuses.
        
        Args:
            deliveries: List of delivery results.
        
        Returns:
            Report status string.
        """
        if not deliveries:
            return "failed"

        statuses = [d.status for d in deliveries]

        if all(s in ("success", "routed") for s in statuses):
            return "delivered"
        elif all(s == "failed" for s in statuses):
            return "failed"
        elif all(s == "deferred" for s in statuses):
            return "deferred"
        elif all(s == "skipped" for s in statuses):
            return "failed"
        else:
            return "partial"

    def _determine_next_actions(self, deliveries: List[DeliveryResult]) -> List[str]:
        """
        Determine next actions based on delivery results.
        
        Args:
            deliveries: List of delivery results.
        
        Returns:
            List of next action strings.
        """
        actions = []

        if any(d.status == "failed" for d in deliveries):
            actions.append("retry_failed_deliveries")

        if any(d.status == "skipped" and d.message == "Creator validation failed" for d in deliveries):
            actions.append("notify_hitl_for_review")

        if any(d.status == "routed" and d.destination == "agent" for d in deliveries):
            actions.append("proceed_to_next_agent")

        if any(d.status == "deferred" for d in deliveries):
            actions.append("await_external_api_integration")

        return actions

    def execute(self, task_prompt: str, context: str | None = None) -> "AgentResult":
        """
        Executes the agent's primary task using inherited helpers.
        """
        from src.agents.base_agent import AgentResult
        
        # 1. Call the LLM using the inherited helper (playbook is auto-applied as system prompt)
        response = self._call_llm(prompt=task_prompt, context=context)
        
        # 2. Clean the output using the inherited helper
        cleaned_content = self._strip_markdown_fences(response.content)
        
        # 3. Return the standardized result
        return AgentResult(
            artifact=cleaned_content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )